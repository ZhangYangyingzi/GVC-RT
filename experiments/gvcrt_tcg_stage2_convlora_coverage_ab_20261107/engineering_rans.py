"""Representative fresh RANS control, zero A/B and disposable nonzero A/B."""
from common_s2 import *
import argparse

def main():
    args=argparse.ArgumentParser();args.add_argument('--case',choices=['baseline','zero','changed'],required=True);args.add_argument('--group',choices=GROUPS,default='A_control');args.add_argument('--rank',type=int,default=0);a=args.parse_args()
    from evaluate import backend,videos,point,historical_input,read_csv
    env();backend(False,a.rank)
    v=next(v for v in videos('final') if v['sequence']=='Kimono1')
    checkpoint=ROOT/f'large_artifacts/engineering/{a.group}/disposable_lora.pt' if a.case=='changed' else None
    label='baseline' if a.case=='baseline' else a.group+'_'+a.case
    batch='engineering/rans_'+label
    p=point(v,5,checkpoint,batch,a.rank,scoring=False,candidate='baseline' if a.case=='baseline' else 'lora',reuse_input=False,group=a.group)
    canonical,stream,trace=historical_input(v,5)
    assert p['result']['bitstream_sha256']==sha(stream) and p['result']['real_bytes']==stream.stat().st_size
    refpoint=load(REFERENCE/'evaluation/engineering_rans_baseline/points/hevc_b_Kimono1_qp5.json')
    if a.case=='baseline':assert p['reconstruction_sha256']==refpoint['reconstruction_sha256'],'REFERENCE_BASELINE_REPRODUCTION_FAILED'
    dump(ROOT/f'audits/RANS_{label}.json',dict(status='PASS',label=label,bitstream_sha256=sha(stream),real_bytes=stream.stat().st_size,
        independent_decode=True,codec_features_equal=True,reconstruction_sha256=p['reconstruction_sha256'],formal_updates=0,quality_scoring=False))
    labels=['baseline']+[g+'_'+c for g in GROUPS for c in ['zero','changed']]
    if all((ROOT/f'audits/RANS_{s}.json').exists() for s in labels):
        d={s:load(ROOT/f'audits/RANS_{s}.json') for s in labels}
        for group in GROUPS:
            assert d[group+'_zero']['reconstruction_sha256']==d['baseline']['reconstruction_sha256']
            assert d[group+'_changed']['reconstruction_sha256']!=d[group+'_zero']['reconstruction_sha256']
            base_trace=read_csv(ROOT/'engineering/rans_baseline/codec_trace/hevc_b_Kimono1_qp5.csv')
            for case in ['zero','changed']:
                assert read_csv(ROOT/f'engineering/rans_{group}_{case}/codec_trace/hevc_b_Kimono1_qp5.csv')==base_trace
        dump(ROOT/'audits/RANS_coverage_identity.json',dict(status='PASS',sequence='Kimono1',QP=5,frames=64,
            baseline_A0_B0_strict_reconstruction_hash=True,nonzero_A_B_change_reconstruction=True,all_stream_SHA_bytes_codec_features_equal=True,
            independent_decode=True,reference_baseline_reproduced=True,formal_updates=0))
    print('RANS_CASE_PASS',label)

if __name__=='__main__':main()
