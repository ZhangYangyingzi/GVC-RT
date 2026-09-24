"""Source-free V1.3 receiver: unchanged ORC2, V1.2 fixed D_enh, V1 stage-D ZERO baseline."""
import argparse
from bridge import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--stream")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    setup()
    source_root=Path(json.loads((V1/"config.json").read_text())["dataset_root"])
    def guard(event,arguments):
        if event=="open" and isinstance(arguments[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(arguments[0])).resolve()
            if path.is_relative_to(source_root) or path.is_relative_to(V1/"data") or path.is_relative_to(V11/"results_v1/oracle") or any(k in path.parts for k in ["teachers_train6","sender_oracles","optimizations","input_cache"]) or (path.parent.name=="base" and path.suffix==".pt"):
                raise RuntimeError(f"Receiver cannot access sender data: {path}")
    sys.addaudithook(guard)
    bundle=Bundle(perceptual=False)
    i,pnet,_=load_models(json.loads((V1/"config.json").read_text()))
    torch.cuda.synchronize()
    t=time.perf_counter()
    rows=decode_base(i,pnet,args.base)
    torch.cuda.synchronize()
    base_seconds=time.perf_counter()-t
    t=time.perf_counter()
    decoded=ORC.decode(args.stream,args.base,MODEL,bundle.net.entropy) if args.stream else None
    entropy_seconds=time.perf_counter()-t
    if decoded:
        assert decoded["level"]==3 and decoded["delta"]==DELTA and decoded["shape"]==(8,17,30)
    before=dpb_hash(pnet)
    frames=[]
    symbol_hashes=[]
    zero_exact=[]
    times=[]
    with torch.no_grad():
        for row in rows:
            frame=row["frame"]
            torch.cuda.synchronize()
            t=time.perf_counter()
            if row["type"]=="I":
                if decoded:
                    assert decoded["frames"][frame] is None
                y=rgb01(row["x_base"])
                symbol_hashes.append(None)
            else:
                ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
                ec=bundle.fixed.ell_c(ell)
                s=decoded["frames"][frame] if decoded else torch.zeros(CONFIG["code_shape"],dtype=torch.int16)
                r=bundle.net.synthesize(s.cuda().float()*DELTA,ec)
                y=rgb01(bundle.fixed.g(ec+r,q))
                if torch.count_nonzero(s)==0:
                    assert torch.count_nonzero(r)==0 and torch.equal(y,rgb01(bundle.fixed.g(ec,q)))
                    zero_exact.append(frame)
                symbol_hashes.append(tensor_hash(s))
            frames.append(y.cpu())
            torch.cuda.synchronize()
            times.append(time.perf_counter()-t)
    assert dpb_hash(pnet)==before
    bundle.check()
    save_pt(args.output,{"frames":frames,"symbols_hash":symbol_hashes,"zero_exact_frames":zero_exact,
                         "base_state_unchanged":True,"source_access_guard":True,
                         "base_decode_seconds":base_seconds,"entropy_decode_seconds":entropy_seconds,"synthesis_seconds":times})


if __name__=="__main__":
    main()
