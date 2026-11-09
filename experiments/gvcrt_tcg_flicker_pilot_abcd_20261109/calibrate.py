"""One shared lambda from8 frozen training windows, never developer metrics."""
from common_s2 import *
from experiment_model import construct
from model_context import frozen_items
from loss import window_gradients,gradient_norm
from cache_codec import cache_paths

def main():
    env()
    with lock('locks/calibration.lock'):
        if (ROOT/'audits/calibration.json').exists():assert load(ROOT/'audits/calibration.json')['status']=='PASS';return
        import torch,statistics
        c=construct('A',True,0);before=tensor_hash(c['named']);guard=FrozenGuard(frozen_items(c));records=[];repeat=None
        for i,row in enumerate(plan_rows()[:8]):
            key,path,receipt=cache_paths(row,row['external_qp'],True);proof=load(receipt);assert sha(path)==proof['cache_sha256']
            data=torch.load(path,map_location='cpu',weights_only=True)
            seed(PLAN_SEED);grad,stats=window_gradients(c,data,1.,calibration=True)
            assert stats['temporal_gradient_L2']>0 and math.isfinite(stats['temporal_gradient_L2'])
            records.append(dict(sample_key=row['sample_key'],source_group=row['source_group'],spatial_gradient_norm=stats['spatial_gradient_L2'],temporal_gradient_norm=stats['temporal_gradient_L2'],
                ratio=stats['spatial_gradient_L2']/stats['temporal_gradient_L2'],statistics=stats))
            if i==0:
                seed(PLAN_SEED);again,other=window_gradients(c,data,1.,calibration=True)
                absolute=gradient_norm([a-b for a,b in zip(grad,again)]);relative=absolute/max(gradient_norm(grad),1e-20)
                assert relative<1e-3,('OWN_REPEAT_GRADIENT_RELATIVE_L2_TOO_LARGE',relative)
                repeat=dict(absolute_L2=absolute,relative_L2=relative,tolerance=1e-3,standard_CUDA_grid_sample_atomic_backward=True,bitwise_determinism_claimed=False)
                del again
            del data,grad
            guard.check();print('CALIBRATION_WINDOW_COMPLETE',i+1,flush=True)
        value=.25*statistics.median([r['ratio'] for r in records]);assert value>0 and math.isfinite(value)
        assert tensor_hash(c['named'])==before;guard.full();assert len(c['optimizer'].state)==0
        dump(ROOT/'audits/calibration.json',dict(status='PASS',lambda_temp=value,rule='.25*median(spatial_gradient_norm/temporal_gradient_norm)',valid_windows=8,records=records,
            parent_A2000_SHA256=INIT_SHA,optimizer_updates=0,common_for_all_branches=True,development_quality_used=False,own_repeat=repeat,trainable_hash_unchanged=before,GT_alignment_not_enabled=True,time=time.time()))
        print('LAMBDA_TEMP_FROZEN',value,flush=True)

if __name__=='__main__':main()
