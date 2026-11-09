"""Reuse parent baseline metrics; supplement per-frame GT detail without retraining."""
from common_s2 import *
import argparse

def reuse():
    checks=[]
    for group,pg,step in [('Original','Original',0),('A2000','A2000',0),('B300','B',300)]:
        src=PARENT_EXPERIMENT/f'evaluation/{pg}/step_{step:04d}';dest=ROOT/f'evaluation/{group}/step_0000'
        for p in sorted((src/'points').glob('*.json')):
            a=load(p);assert a['status']=='PASS' and a['codec_features_match_original'] and a['codec_references_forced_to_original']
            for n,h in a['record_SHA256'].items():assert sha(PARENT_EXPERIMENT/n)==h
            if group=='B300':assert a['identity']['checkpoint_SHA256']==B300_SHA
            # Retain parent records verbatim, under a provenance wrapper; do not
            # pretend a supplemental metric was present in the old evaluation.
            a['result']['model']=group;a['result']['step']=0
            dump(dest/'points'/p.name,dict(a,reused_from=str(p),reused_SHA256=sha(p)))
            checks.append(dict(group=group,point=p.name,source=str(p),SHA256=sha(p)))
    assert len(checks)==108
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',points=108,records=checks,new_B300_detail_pending=True))

def detail(rank):
    import torch
    from experiment_model import construct,reconstruct
    from model_context import assert_reset
    from cache_codec import cache_paths
    from pipeline import wait_rows
    from loss import detail_error
    env();reuse();jobs=[(p,q) for p in load(ROOT/'manifests/quick_development.json') for q in [0,5,9]]
    for ix,(profile,q) in enumerate(jobs):
        if ix%2!=rank:continue
        dest=ROOT/'baseline_detail'/f'{profile["sequence"]}_qp{q}.json'
        if dest.exists():assert load(dest)['status']=='PASS';continue
        row=profile['row'];wait_rows([row],q,False);key,p,rp=cache_paths(row,q,False);proof=load(rp);assert sha(p)==proof['cache_sha256'];data=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        c=construct('B300',False,rank);assert_reset(c);values=[];h,w=proof['native_shape']
        parentframes=list(csv.DictReader((PARENT_EXPERIMENT/f'evaluation/B/step_0300/frames/{profile["sequence"]}_qp{q}.csv').open()))
        with torch.no_grad():
            for i in range(64):
                raw=data['original'][i].to(rank) if i==0 else reconstruct(c,data['features'][i].to(rank),proof['actual_qps'][i])
                out=c['core'].unit(raw,h,w);assert tensor_sha(out)==parentframes[i]['RGB_tensor_SHA256'],('DETAIL_B300_PATH_DIFF',ix,i)
                target=(data['GT'][i].permute(2,0,1)[None].float()/255).to(rank)
                values.append(dict(frame=i,GT_multiscale_detail_error=float(detail_error(out,target))))
        dump(dest,dict(status='PASS',model='B300',sequence=profile['sequence'],QP=q,frames=values,GT_multiscale_detail_error=sum(v['GT_multiscale_detail_error'] for v in values)/64,stream_SHA256=proof['stream_SHA256'],cache_identity=key,exact_parent_RGB_hashes=True));del c,data;torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int);a=p.parse_args();env();reuse() if a.rank is None else detail(a.rank)
