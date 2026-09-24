"""Independent process: no original video, source frames or sender training cache."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",required=True)
    parser.add_argument("--enhancement")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    cfg = json.loads((HERE/"config.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu_uuid"]
    import torch
    from codec import load_models, decode_base, dpb_hash
    from enhancement import Enhancement, FactorizedEntropy
    from evaluation import read_container, rgb01
    def source_guard(event,arguments):
        if event=="open" and isinstance(arguments[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(arguments[0])).resolve()
            if path.is_relative_to(HERE/"data") or path.is_relative_to(Path(cfg["dataset_root"])) or (path.parent.name=="base" and path.suffix==".pt"):
                raise RuntimeError(f"Receiver attempted to access non-shared source/cache: {path}")
    sys.addaudithook(source_guard)
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.enhancement is None:
        # No enhancement bitstream: no enhancement model construction or calls.
        i,p,_=load_models(cfg)
        rows=decode_base(i,p,args.base)
        with Path(args.output).open("xb") as f:
            torch.save({"frames":[rgb01(row["x_base"],cfg["valid_hw"]) for row in rows],
                        "no_enhancement_bypass":True,"source_access_guard":True},f)
        return
    if args.checkpoint is None:
        raise ValueError("Enhancement stream requires a shared enhancement checkpoint")
    ck = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    enh = Enhancement(cfg["architecture"]).cuda().eval()
    entropy = FactorizedEntropy(cfg["architecture"]["enhancement_channels"])
    enh.load_state_dict(ck["enhancement"],strict=True)
    entropy.load_state_dict(ck["entropy"],strict=True)
    hw, streams = read_container(args.enhancement,args.base)
    i,p,_ = load_models(cfg)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    rows = decode_base(i,p,args.base)
    # Independent float32 generator loaded through fresh checkpoint instance.
    from src.models.video_model_gvcrt import DMC
    from codec import strict_load, ROOT
    gen_model = DMC()
    strict_load(gen_model,ROOT/cfg["checkpoint_p"],"P")
    generator = gen_model.recon_generation_net.decoder.cuda().eval().requires_grad_(False)
    assert len(streams)==len(rows)
    frames, timings, symbol_hashes = [],[],[]
    before = dpb_hash(p)
    import hashlib
    with torch.no_grad():
        for row,stream in zip(rows,streams):
            torch.cuda.synchronize()
            t = time.perf_counter()
            if not stream:
                frames.append(rgb01(row["x_base"],hw))
                symbol_hashes.append(None)
            else:
                if row["type"] != "P":
                    raise ValueError("I-frame enhancement forbidden")
                symbols,delta,level = entropy.decode(stream)
                symbol_hashes.append(hashlib.sha256(symbols.numpy().tobytes()).hexdigest())
                ell,q = row["ell"].cuda().float(),row["q_recon"].cuda().float()
                corrected = ell + enh.decoder(symbols.cuda().float()*delta,ell)
                frames.append(rgb01(generator(corrected,q),hw).cpu())
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-t)
    assert before == dpb_hash(p)
    torch.cuda.synchronize()
    wall = time.perf_counter()-start
    result = {"frames":frames,"symbol_hashes":symbol_hashes,"enhancement_seconds":timings,
              "base_decode_seconds":[r["decode_seconds"] for r in rows],"wall_seconds":wall,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"dpb_unchanged":True,"source_access_guard":True,
              "timing_note":"wall includes base decoder, DPB hashes, CPU copies, and fresh FP32 generator loading; model construction before timer excluded"}
    with Path(args.output).open("xb") as f:
        torch.save(result,f)


if __name__ == "__main__":
    main()
