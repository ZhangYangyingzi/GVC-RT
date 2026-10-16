"""Synthetic native-coder validation only; never a scientific RD measurement."""
from channel_rate import *
import sys
from types import SimpleNamespace

if __name__=='__main__':
    import torch
    sys.path.insert(0,str(REPO))
    from src.models.entropy_models import EntropyCoder
    cdf=np.array([[0,65535,65536]],dtype=np.int32)
    lengths=np.array([3],dtype=np.int32);offsets=np.array([0],dtype=np.int32)
    model=SimpleNamespace(get_cdf_info=lambda:(cdf,lengths,offsets))
    coder=EntropyCoder();coder.set_use_two_entropy_coders(False)
    group=coder.add_cdf(cdf,lengths,offsets);rows=[];N=8192
    for symbol in (0,32,-32,127,-128):
        # Native output allocation is one byte per symbol. Use sparse tails
        # so this auxiliary check stays within that existing coder constraint.
        values=torch.zeros(N,dtype=torch.int16);values[::16]=symbol
        coder.reset();packed=values<<8
        coder.encode_y(packed,group);coder.flush();payload=coder.get_encoded_stream()
        coder.set_stream(payload);decoded=coder.decode_and_get_y(torch.zeros(N,dtype=torch.uint8),group,'cpu',torch.float32)
        assert torch.equal(decoded,values.float())
        bits,summary=cdf_bits(values.numpy(),np.zeros(N),model)
        assert float(bits.sum())+256<N*8
        delta=len(payload)*8-float(bits.sum())
        assert abs(delta)<=256,('Native escape rule inconsistent',symbol,delta)
        p=ROOT/'bitstreams'/f'native_escape_validation_{symbol}.bin';p.write_bytes(payload)
        rows.append(dict(symbol=symbol,actual_native_stream_bytes=len(payload),estimated_CDF_plus_bypass_bits=float(bits.sum()),stream_bits_minus_estimate=delta,decode_exact=True,payload_path=str(p),payload_sha256=sha(p),**summary))
    dump('sanity/native_escape_bypass.json',dict(status='PASS',scope='synthetic validation of installed RANS escape rule only; excluded from all scientific RD and utility tables',bypass_precision=2,per_channel_actual_rate_available=False,cases=rows))
    print('NATIVE ESCAPE VALIDATION PASS',[(r['symbol'],r['stream_bits_minus_estimate']) for r in rows])
