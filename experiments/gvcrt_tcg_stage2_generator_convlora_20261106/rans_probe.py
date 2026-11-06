"""Real stream tests, including a disposable nonzero LoRA model; no quality scan."""
from common_s2 import *
import argparse

def main():
    args=argparse.ArgumentParser();args.add_argument('--case',choices=['baseline','zero','changed'],required=True);args.add_argument('--rank',type=int,default=0);a=args.parse_args()
    from evaluate import backend,videos,point,historical_input
    env();backend(False,a.rank)
    v=next(v for v in videos('final') if v['sequence']=='Kimono1')
    checkpoint=ROOT/'large_artifacts/engineering/disposable_lora.pt' if a.case=='changed' else None
    p=point(v,5,checkpoint,'engineering_rans_'+a.case,a.rank,scoring=False,candidate='baseline' if a.case=='baseline' else 'lora',reuse_input=False)
    canonical,stream,trace=historical_input(v,5)
    assert p['result']['real_bytes']==stream.stat().st_size==canonical['result']['real_bytes']
    assert p['result']['bitstream_sha256']==sha(stream)==canonical['result']['bitstream_sha256']
    dump(ROOT/f'audits/RANS_{a.case}.json',dict(status='PASS',case=a.case,real_bytes=stream.stat().st_size,bitstream_sha256=sha(stream),
        canonical_stream_equal=True,independent_decode=True,codec_encoder_decoder_features_equal=True,
        reconstruction_sha256=p['reconstruction_sha256'],formal_updates=0,quality_scan=False))
    if all((ROOT/f'audits/RANS_{n}.json').exists() for n in ['baseline','zero','changed']):
        d={n:load(ROOT/f'audits/RANS_{n}.json') for n in ['baseline','zero','changed']}
        assert d['baseline']['reconstruction_sha256']==d['zero']['reconstruction_sha256']
        assert d['changed']['reconstruction_sha256']!=d['zero']['reconstruction_sha256']
        dump(ROOT/'audits/RANS_identity.json',dict(status='PASS',baseline_zero_strict_reconstruction_hash=True,
            nonzero_LoRA_actually_changes_reconstruction=True,baseline_zero_changed_same_bitstream=True,real_bytes=d['baseline']['real_bytes'],
            bitstream_sha256=d['baseline']['bitstream_sha256'],formal_updates=0))
    print('RANS_CASE_PASS',a.case)

if __name__=='__main__':main()
