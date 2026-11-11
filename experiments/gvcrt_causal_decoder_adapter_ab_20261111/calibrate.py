"""One fixed training-only gradient calibration at common step0, zero updates."""
from common_s2 import *
import statistics


def main():
    env();import torch
    from adapter import CausalAdapter
    from cache_h import paths,ready
    from loss import load_training_context,window,norm
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in (0,1,2,3)
    rows=load(ROOT/'manifests/calibration_windows.json')
    while not all(ready(r) for r in rows):time.sleep(2)
    p=ROOT/'checkpoints/common_adapter_step0.pt'
    if not p.exists():
        seed(TRAIN_SEED);a=CausalAdapter();atomic_torch(p,dict(adapter=a.state_dict(),seed=TRAIN_SEED,updates=0,parameters=21696))
    if (ROOT/'gradient_calibration.json').exists():return
    c=load_training_context(torch.device('cuda',0));targets=list(c['adapter'].B.parameters());records=[]
    before=tensor_hash(c['adapter'].state_dict().items())
    for row in rows:
        p,rp=paths(row);assert sha(p)==load(rp)['SHA256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        s,gs=window(c,d,mode='spatial',gradient_targets=targets)
        t,gt=window(c,d,mode='temporal',gradient_targets=targets)
        ns,nt=norm(gs),norm(gt);valid=math.isfinite(ns) and math.isfinite(nt) and nt>0
        cosine=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs,gt))/(ns*nt) if ns>0 and nt>0 else None
        angle=math.degrees(math.acos(max(-1,min(1,cosine)))) if cosine is not None else None
        records.append(dict(sample_key=row['sample_key'],plan_slot=row['plan_slot'],QP=row['external_qp'],
            L_spatial=s['objective'],L_temporal=t['objective'],gradient_spatial_norm=ns,gradient_temporal_norm=nt,
            ratio=ns/nt if valid else None,cosine=cosine,angle_degrees=angle,valid=valid))
        event('calibration_window',slot=row['plan_slot'],spatial_norm=ns,temporal_norm=nt)
    ratios=[r['ratio'] for r in records if r['valid']];assert ratios
    coeff=.1*statistics.median(ratios);assert math.isfinite(coeff)
    assert tensor_hash(c['adapter'].state_dict().items())==before
    dump(ROOT/'gradient_calibration.json',dict(status='PASS',lambda_temp=coeff,records=records,
        valid_records=len(ratios),rule='0.1*median(grad_B spatial norm / grad_B temporal norm)',
        targets=['B.weight','B.bias'],optimizer_updates=0,training_only=True,common_step0_SHA256=sha(ROOT/'checkpoints/common_adapter_step0.pt'),
        head_SHA256=sha(ROOT/'checkpoints/frozen_FP32_head.pt'),no_model_update=True,time=time.time()))
    write_csv(ROOT/'calibration_gradient_norms.csv',records)
    event('calibration_complete',lambda_temp=coeff,valid_records=len(ratios))

if __name__=='__main__':main()
