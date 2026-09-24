"""Read-only estimated-vs-actual coding diagnosis using the exact checkpoint CDFs."""
import argparse
import struct
from support import *
import bitstream


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stream",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    ck=torch.load(args.model,map_location="cpu",weights_only=False)
    channels=18 if ck["method"]=="direct" else 8
    entropy=FactorizedEntropy(channels)
    sd=ck["entropy"] if ck["method"]=="direct" else {"log_scale":ck["model"]["entropy.log_scale"]}
    entropy.load_state_dict(sd,strict=True)
    video=Path(args.stream).stem
    base=V1/"run_v1/base"/f"{video}_q0.bin"
    decoded=bitstream.decode(args.stream,base,args.model,entropy)
    delta=decoded["delta"]
    cdfs,_=entropy.coder(delta)
    cdf=torch.tensor(cdfs)
    probs=(cdf[:,1:]-cdf[:,:-1]).double()/65536
    raw=Path(args.stream).read_bytes()
    pos=bitstream.HEADER.size
    rows=[]
    for frame,s in enumerate(decoded["frames"]):
        flag=raw[pos]
        pos+=1
        if flag!=1:
            rows.append({"frame":frame,"flag":flag,"packet_bits":8,"payload_bits":0})
            continue
        length,crc=struct.unpack_from("<II",raw,pos)
        pos+=8+length
        hist=torch.stack([torch.bincount((a.flatten()+128).long(),minlength=256) for a in s[0]])
        ideal=-(hist*probs.log2()).sum().item()
        estimated=entropy.bits(s.float(),delta).item()
        # Stable float64 logistic differences expose float32 cancellation separately.
        sdscale=F.softplus(entropy.log_scale.double())+1e-4
        lo=torch.sigmoid((s.double()-.5)*delta/sdscale)
        hi=torch.sigmoid((s.double()+.5)*delta/sdscale)
        model64=-(hi-lo).clamp_min(1e-9).log2().sum().item()
        rows.append({"frame":frame,"flag":flag,"packet_bits":(length+9)*8,"payload_bits":length*8,
                     "float32_estimated_bits":estimated,"float64_model_bits":model64,"actual_CDF_ideal_bits":ideal,
                     "arithmetic_minus_CDF_ideal":length*8-ideal,"symbol_min":s.min().item(),"symbol_max":s.max().item()})
    assert pos==len(raw)
    result={"stream":args.stream,"model":args.model,"delta":delta,"CDF_totals":cdf[:,-1].tolist(),"frames":rows}
    save_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
