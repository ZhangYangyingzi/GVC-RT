"""Independent receiver. Source videos, teacher/oracle tensors and base training caches are forbidden."""
import argparse
import time
from support import *
from model import ResidualCodec
import bitstream


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--stream",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    configure()
    source_root=Path(json.loads((V1/"config.json").read_text())["dataset_root"])
    def guard(event,arguments):
        if event=="open" and isinstance(arguments[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(arguments[0])).resolve()
            # No recursive config reads inside an audit hook.
            if path.is_relative_to(V1/"data") or path.is_relative_to(V11/"results_v1/oracle") or "teachers_train6" in path.parts or "sender_oracles" in path.parts or (path.parent.name=="base" and path.suffix==".pt") or path.is_relative_to(source_root):
                raise RuntimeError(f"Receiver source/cache access forbidden: {path}")
    sys.addaudithook(guard)
    ck=torch.load(args.model,map_location="cpu",weights_only=False)
    assert ck["baseline_checkpoint_sha256"]==sha(V1/CFG["baseline_checkpoint"])
    if ck["method"]=="direct":
        entropy=FactorizedEntropy(18)
        entropy.load_state_dict(ck["entropy"],strict=True)
        net=None
    else:
        net=ResidualCodec(ck["residual_scale"]).cuda().eval().requires_grad_(False)
        net.load_state_dict(ck["model"],strict=True)
        entropy=net.entropy
    t=time.perf_counter()
    decoded=bitstream.decode(args.stream,args.base,args.model,entropy)
    assert decoded["method"]==(0 if ck["method"]=="direct" else 1)
    assert decoded["shape"]==((18,68,120) if net is None else (8,17,30))
    if ck.get("deltas"):
        assert decoded["level"]<len(ck["deltas"]) and decoded["delta"]==ck["deltas"][decoded["level"]]
    entropy_seconds=time.perf_counter()-t
    fixed=FixedModels()
    i,pnet,_=load_models(json.loads((V1/"config.json").read_text()))
    torch.cuda.synchronize()
    t=time.perf_counter()
    rows=decode_base(i,pnet,args.base)
    torch.cuda.synchronize()
    base_seconds=time.perf_counter()-t
    before=dpb_hash(pnet)
    frames=[]
    symbols_hash=[]
    times=[]
    zero_checks=[]
    torch.cuda.reset_peak_memory_stats()
    assert len(rows)==len(decoded["frames"])
    with torch.no_grad():
        for row,s in zip(rows,decoded["frames"]):
            torch.cuda.synchronize()
            t=time.perf_counter()
            if row["type"]=="I":
                assert s is None
                y=rgb01(row["x_base"])
                symbols_hash.append(None)
            else:
                assert s is not None
                ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
                ec=fixed.ell_c(ell)
                u=s.cuda().float()*decoded["delta"]
                r=u if net is None else net.synthesize(u,ec)
                if torch.count_nonzero(s)==0:
                    assert torch.count_nonzero(r)==0
                    xc=fixed.g(ec,q)
                    raw=fixed.g(ec+r,q)
                    assert torch.equal(raw,xc)
                    zero_checks.append(row["frame"])
                else:
                    raw=fixed.g(ec+r,q)
                y=rgb01(raw).cpu()
                symbols_hash.append(tensor_hash(s))
            frames.append(y)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-t)
    assert before==dpb_hash(pnet)
    fixed.assert_frozen()
    save_pt(args.output,{"frames":frames,"symbols_hash":symbols_hash,"zero_exact_frames":zero_checks,
                         "source_access_guard":True,"base_state_unchanged":True,"entropy_decode_seconds":entropy_seconds,
                         "base_decode_wall_seconds":base_seconds,"synthesis_seconds":times,
                         "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                         "timing_note":"model loading excluded; synthesis includes stage-D ZERO and original G; zero-path audit extra G call included"})


if __name__=="__main__":
    main()
