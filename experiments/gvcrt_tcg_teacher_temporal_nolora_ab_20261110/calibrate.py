"""Eight frozen train windows, unchanged common state, unclipped gradient ratios."""
from common_s2 import *
import statistics

def main():
    import torch
    from experiment_model import construct
    from model_context import frozen_items
    from cache_codec import cache_paths
    from cache_teacher import paths
    from pipeline import wait_rows
    from loss import backward_window,norm
    env()
    if (ROOT/'gradient_calibration.json').exists():assert load(ROOT/'gradient_calibration.json')['status']=='PASS';return
    rows=plan_rows()[:8];wait_rows(rows);seed(PLAN_SEED);c=construct(GROUPS[1],True,0);seed(PLAN_SEED);original=tensor_hash(c['pm'].state_dict().items());frozen=tensor_hash(frozen_items(c));records=[]
    for ix,row in enumerate(rows):
        _,p,rp=cache_paths(row,row['external_qp'],True);assert sha(p)==load(rp)['cache_sha256'];data=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        _,tp,tr=paths(row,row['external_qp']);assert sha(tp)==load(tr)['SHA256'];target=torch.load(tp,map_location='cpu',weights_only=True,mmap=True)
        terms={};trace=None
        for mode in ['content','output_temporal','latent_temporal','cosine','LPIPS_VGG','DISTS','RAFT_motion']:
            c['optimizer'].zero_grad(set_to_none=True);stats=backward_window(c,data,target,dict(output=1,latent=1),dict(lambda_out=0.,lambda_latent=0.),mode=mode)
            gn={k:norm([p.grad for p in ps]) for k,ps in c['parameter_sets'].items()};gn['Bridge_TCG']=norm([p.grad for p in c['params']])
            assert all(math.isfinite(v) for v in gn.values()),(mode,gn)
            if mode=='latent_temporal':
                assert stats['latent_valid_frames']>0 and gn['Bridge']>0 and gn['TCG']==0 and all(p.grad is None for p in c['parameter_sets']['TCG'])
            else:assert gn['Bridge_TCG']>0,(mode,gn)
            terms[mode]=gn
            if mode=='content':trace=stats['trace']
        assert terms['output_temporal']['Bridge_TCG']>0
        record=dict(sample_key=row['sample_key'],QP=row['external_qp'],raw_gradient_norms=terms,ratio_out=terms['content']['Bridge_TCG']/terms['output_temporal']['Bridge_TCG'],ratio_latent=terms['content']['Bridge']/terms['latent_temporal']['Bridge'],latent_direct_TCG_gradient=False,frames=trace,GT_mean_flow_magnitude=sum(float(x.square().sum(1).sqrt().mean()) for x in data['GT_forward'][49:64])/15)
        records.append(record);dump(ROOT/'audits/calibration_progress.json',dict(completed=ix+1,windows=records,time=time.time()))
        assert original==tensor_hash(c['pm'].state_dict().items()) and frozen==tensor_hash(frozen_items(c)) and len(c['optimizer'].state)==0
        del data,target
    coeff=dict(lambda_out=.1*statistics.median(r['ratio_out'] for r in records),lambda_latent=.1*statistics.median(r['ratio_latent'] for r in records))
    assert all(math.isfinite(v) and v>0 for v in coeff.values())
    dump(ROOT/'gradient_calibration.json',dict(status='PASS',**coeff,windows=records,unclipped=True,updated=False,trainables=dict(Bridge=801124,TCG=137760),ratio=.1,development_or_test_used=False,time=time.time()))
    protocol=load(ROOT/'protocol.json');protocol['losses'].update(coeff);protocol['status']='FROZEN';dump(ROOT/'protocol.json',protocol)
    print('CALIBRATION_PASS',coeff,flush=True)
if __name__=='__main__':main()
