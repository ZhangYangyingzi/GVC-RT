"""Inspect transmitted symbol utilization from actual files; CPU-only and no sources."""
import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-name",default="run_v1")
    parser.add_argument("--output",default="symbol_utilization.json")
    args=parser.parse_args()
    import torch
    from enhancement import FactorizedEntropy
    from evaluation import read_container
    torch.set_num_threads(4)
    cfg=json.loads((HERE/"config.json").read_text())
    run=HERE/args.run_name
    results=[]
    for stage in ["c","d"]:
        path=run/f"stage_{stage}/checkpoint.pt"
        if not path.exists():
            continue
        ck=torch.load(path,map_location="cpu",weights_only=False)
        entropy=FactorizedEntropy(cfg["architecture"]["enhancement_channels"])
        entropy.load_state_dict(ck["entropy"],strict=True)
        for stream in sorted(run.glob(f"quantized_{stage}_*/*.gle")):
            base=run/"base"/f'{stream.stem}_q{ck["base_qp"]}.bin'
            hw,packets=read_container(stream,base)
            symbols=[entropy.decode(p)[0] for p in packets if p]
            stacked=torch.cat(symbols,0)
            results.append({"method":stream.parent.name,"video":stream.stem,"actual_bits":stream.stat().st_size*8,
                            "symbol_count":stacked.numel(),"symbol_min":int(stacked.min()),"symbol_max":int(stacked.max()),
                            "nonzero_fraction":float((stacked!=0).float().mean()),
                            "distinct_symbol_values":torch.unique(stacked).tolist(),
                            "consecutive_frames_symbol_change_fraction":float((stacked[1:]!=stacked[:-1]).float().mean()),
                            "identical_all_zero_frames":int((stacked==0).flatten(1).all(1).sum()),
                            "channel_scales":(torch.nn.functional.softplus(entropy.log_scale.detach()).flatten()+1e-4).tolist()})
    with (run/args.output).open("x") as f:
        f.write(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
