"""Training-only8-window gradient calibration at unchanged B300."""
from common_s2 import *

def main():
    import torch,statistics
    from experiment_model import construct
    from cache_codec import cache_paths
    from pipeline import wait_rows
    from loss import backward_window,norm
    env()
    if (ROOT/'audits/calibration.json').exists():assert load(ROOT/'audits/calibration.json')['status']=='PASS';return
    rows=plan_rows()[:8];wait_rows(rows)
    seed(PLAN_SEED);c=construct('F',True,0);initial=tensor_hash(c['named']);results=[]
    for i,r in enumerate(rows):
        _,p,rp=cache_paths(r,r['external_qp'],True);assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        values={}
        for mode in ['spatial','detail']:
            c['optimizer'].zero_grad(set_to_none=True);stats=backward_window(c,d,1,mode=mode)
            values[mode]=norm([p.grad for p in c['params']]);assert math.isfinite(values[mode]) and values[mode]>0,(mode,values)
        results.append(dict(sample_key=r['sample_key'],spatial_gradient_L2=values['spatial'],detail_gradient_L2=values['detail'],ratio=values['spatial']/values['detail']))
        dump(ROOT/'audits/calibration_progress.json',dict(completed=i+1,windows=results,time=time.time()))
        del d
    coefficient=.1*statistics.median(x['ratio'] for x in results);assert math.isfinite(coefficient) and coefficient>0
    assert initial==tensor_hash(c['named']) and len(c['optimizer'].state)==0
    dump(ROOT/'audits/calibration.json',dict(status='PASS',lambda_detail=coefficient,lambda_temp=LAMBDA_TEMP,windows=results,trainable_parameters=sum(p.numel() for p in c['params']),unupdated_B300=True,trainable_SHA256=initial,development_or_test_used=False,time=time.time()))
    print('CALIBRATION_PASS',coefficient,flush=True)
if __name__=='__main__':main()
