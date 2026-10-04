"""Read-only old training log and checkpoint movement; no optimizer or model forward."""
import math,statistics
from v67b_io import *
def main():
    import torch
    torch.set_num_threads(2);cfg=frozen();rows=read(V66/'training_logs'/f'{OLD_B}.csv');assert [int(r['step']) for r in rows]==list(range(1,1001))
    keys=['LPIPS','DISTS','rate_bpp','proxy_L1','proxy_MS_SSIM','weighted_structure_loss','total_loss','wrapper_gradient_norm','bridge_gradient_norm','generator_gradient_norm','wrapper_parameter_norm','bridge_parameter_norm','generator_parameter_norm']
    curves=[dict(step=int(r['step']),**{k:float(r[k]) for k in keys}) for r in rows];assert all(math.isfinite(r[k]) for r in curves for k in keys)
    write(ROOT/'results/B_training_curve_raw.csv',curves);summary=[]
    for name,lo,hi in [('1-250',1,250),('251-500',251,500),('501-750',501,750),('751-1000',751,1000),('last_100',901,1000),('last_250',751,1000)]:
        own=[r for r in curves if lo<=r['step']<=hi]
        for k in keys:summary.append(dict(window=name,start_step=lo,end_step=hi,N=len(own),metric=k,mean=statistics.mean(r[k] for r in own),median=statistics.median(r[k] for r in own)))
    write(ROOT/'results/B_training_curve_summary.csv',summary)
    states=[]
    for m in B_METHODS:
        cp=cfg['checkpoints'][m];assert sha(cp['path'])==cp['sha256'];s=torch.load(cp['path'],map_location='cpu',weights_only=True);states.append({k:s[k] for k in ('wrapper','bridge','generator')});del s
    movement=[]
    for component in ('wrapper','bridge','generator'):
        ss=[s[component] for s in states];assert list(ss[0])==list(ss[1])==list(ss[2])
        dot_delta=n1=n2=0.
        for k in ss[0]:
            delta1=ss[1][k].double()-ss[0][k].double();delta2=ss[2][k].double()-ss[1][k].double();dot_delta+=float((delta1*delta2).sum());n1+=float(delta1.square().sum());n2+=float(delta2.square().sum())
        assert n1>0 and n2>0;delta_cos=dot_delta/math.sqrt(n1*n2)
        for i,(start,end) in enumerate(((250,500),(500,1000))):
            base=target=dot=delta=0.
            for k in ss[i]:
                a,b=ss[i][k].double(),ss[i+1][k].double();base+=float(a.square().sum());target+=float(b.square().sum());dot+=float((a*b).sum());delta+=float((b-a).square().sum())
            assert base>0 and target>0
            movement.append(dict(module=component,start_step=start,end_step=end,relative_L2_parameter_change=math.sqrt(delta/base),L2_parameter_change=math.sqrt(delta),cosine_parameter_vectors=dot/math.sqrt(base*target),cosine_parameter_delta=delta_cos,cosine_parameter_delta_definition='cosine(delta_250_to_500, delta_500_to_1000), same value for both interval rows',compression_core_excluded=True))
    assert all(math.isfinite(v) for r in movement for v in r.values() if isinstance(v,float));write(ROOT/'results/checkpoint_parameter_movement.csv',movement)
    dump(ROOT/'audits/training_log_audit.json',dict(status='PASS',source=str(V66/'training_logs'/f'{OLD_B}.csv'),sha256=sha(V66/'training_logs'/f'{OLD_B}.csv'),updates=1000,no_training=True));print('TRAINING LOG AUDIT PASS',flush=True)
if __name__=='__main__':main()
