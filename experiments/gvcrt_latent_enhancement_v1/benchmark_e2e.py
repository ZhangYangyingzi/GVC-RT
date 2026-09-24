"""Measured complete sender/receiver pipeline, no latent cache. Single configured GPU."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/"config.json").read_text())
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["gpu_uuid"]
sys.path.insert(0,str(HERE.parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--level",type=int,default=1)
    args = parser.parse_args()
    import torch
    from torch.nn import functional as F
    from codec import load_models,encode_base,decode_base,source_frame,dpb_hash,strict_load,ROOT
    from enhancement import Enhancement,FactorizedEntropy
    from evaluation import write_container,read_container,rgb01
    from src.models.video_model_gvcrt import DMC
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    folder = HERE/args.output
    folder.mkdir(exist_ok=False)
    entry = json.loads((HERE/"manifest.json").read_text())["train"][0]
    ck = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    qp,delta = ck["base_qp"],CFG["training"]["deltas"][args.level]
    i,p,_ = load_models(CFG)
    ir,pr,_ = load_models(CFG)
    e = Enhancement(CFG["architecture"]).cuda().eval()
    e.load_state_dict(ck["enhancement"],strict=True)
    ent = FactorizedEntropy(CFG["architecture"]["enhancement_channels"])
    ent.load_state_dict(ck["entropy"],strict=True)
    rec_ent = FactorizedEntropy(CFG["architecture"]["enhancement_channels"])
    rec_ent.load_state_dict(ck["entropy"],strict=True)
    m = DMC()
    strict_load(m,ROOT/CFG["checkpoint_p"],"P")
    g = m.recon_generation_net.decoder.cuda().eval().requires_grad_(False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    enc = encode_base(i,p,CFG,entry,qp,folder/"base.bin")
    torch.cuda.synchronize()
    t1=time.perf_counter()
    sender_rows=decode_base(i,p,folder/"base.bin")
    torch.cuda.synchronize()
    t2=time.perf_counter()
    packets=[]
    with torch.no_grad():
        for row in sender_rows:
            if row["type"]=="I":
                packets.append(b"")
                continue
            x=source_frame(entry["video_id"],row["frame"])
            ell=row["ell"].cuda().float()
            symbols=(e.encoder(F.pad(x,(0,0,0,8),mode="replicate"),ell)/delta).round()
            packets.append(ent.encode(symbols,delta,args.level))
    enhancement_bits=write_container(folder/"enhancement.gle",folder/"base.bin",entry["valid_hw"],packets)
    torch.cuda.synchronize()
    t3=time.perf_counter()
    rows=decode_base(ir,pr,folder/"base.bin")
    assert all(torch.equal(a["x_base"],b["x_base"]) and
               (a["type"]=="I" or (torch.equal(a["ell"],b["ell"]) and torch.equal(a["q_recon"],b["q_recon"])))
               for a,b in zip(sender_rows,rows))
    state=dpb_hash(pr)
    torch.cuda.synchronize()
    t4=time.perf_counter()
    hw,packets=read_container(folder/"enhancement.gle",folder/"base.bin")
    with torch.no_grad():
        for row,packet in zip(rows,packets):
            if packet:
                s,d,_=rec_ent.decode(packet)
                ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
                output=rgb01(g(ell+e.decoder(s.cuda().float()*d,ell),q),hw)
            else:
                output=rgb01(row["x_base"],hw)
    assert state==dpb_hash(pr)
    torch.cuda.synchronize()
    end=time.perf_counter()
    result={"frames":entry["frames"],"qp":qp,"delta":delta,
            "base_encode_wall_seconds":t1-begin,"sender_base_latent_acquisition_seconds":t2-t1,
            "sender_analysis_actual_entropy_seconds":t3-t2,"receiver_base_decode_seconds":t4-t3,
            "receiver_actual_entropy_synthesis_generation_seconds":end-t4,
            "total_seconds":end-begin,"total_ms_per_frame":1000*(end-begin)/entry["frames"],
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"base_bits":sum(v["base_bits"] for v in enc),
            "enhancement_bits":enhancement_bits,"dpb_unchanged":True,"sender_receiver_base_latent_exact":True,
            "included":"PNG reads twice, padding, actual base coding, sender local base decode to obtain ell, CPU DPB hashes/clones, actual enhancement entropy coding, file I/O, independent receiver base decode, enhancement entropy decode and RGB synthesis",
            "excluded":"model loading/CDF initialization of base codec, physical network transfer, metrics, output video rendering; no warm-up frames discarded",
            "cache":"No latent/reconstruction cache; OS filesystem cache may be warm. Diagnostic decode computes original RGB before enhanced RGB; both costs included."}
    (folder/"timing.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
