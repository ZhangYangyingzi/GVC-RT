"""Preserve training records; independently audit pre/post-clip alias diagnostics."""
from io32 import *
import argparse,math

def aggregate():
 rows=[]
 for branch in BRANCHES[2:]:
  for step in range(1,1001):
   p=ROOT/'branches'/branch/'receipts'/f'{step:04d}.json';r=load(p)
   n=r['C1_full_gradient_norm'];e=r['additional_gradient_norm'];pre=r['global_preclip_norm'];post=r['global_postclip_norm'];c=r['clip_coefficient']
   composed=math.sqrt(n*n+e*e+2*n*e*r['additional_C1_cosine'])
   checks=dict(preclip_norm_composition_relative_error=abs(composed-pre)/pre,clip_scaling_relative_error=abs(pre*c-post)/post,legacy_alias_formula_residual=abs(r['gradient_formula_error']-abs(1-c)/c)/max(1,r['gradient_formula_error']))
   assert all(v<=1e-6 for v in checks.values()),(branch,step,checks)
   rows.append(dict(branch=branch,step=step,receipt=str(p),receipt_sha256=sha(p),legacy_gradient_formula_error=r['gradient_formula_error'],**checks))
 probes=[load(ROOT/'audits'/f'gradient_clip_replay_{b}.json') for b in BRANCHES[2:]]
 assert all(x['status']=='PASS' and x['optimizer_updates']==0 and x['EMA_commits']==0 for x in probes)
 write(ROOT/'audits/gradient_recording_corrected_checks.csv',rows)
 dump(ROOT/'audits/gradient_recording_integrity.json',dict(status='PASS',training_receipts_unchanged=True,train_source_sha256=sha(ROOT/'train.py'),steps_checked=2000,tolerance=1e-6,maximum_errors={k:max(r[k] for r in rows) for k in checks},diagnosis='gr tensors were assigned to p.grad, then clip_grad_norm_ modified them in place. Legacy diagnostic compares clipped gr to unscaled base+extra; it equals abs(1-clip)/clip, not preclip vector merge error.',actual_training_merge='gr=[base+extra]; assign p.grad; one global clip; one AdamW step; unchanged',raw_field_is_not_preclip_merge_error=True,probes=probes))

def probe(branch,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from training32 import setup,official_model,restore_rng,buffer_state,restore_buffers,crop_c1,full_pass,norm,groups,check_frozen,rng_state
 from frames32 import PlanFrames
 from calibration32b import get_stratum,feature_derivatives
 from distribution32 import to
 ctx=setup();models=ctx[3];params=ctx[6];official=official_model()
 cp=ROOT/'branches'/branch/'checkpoints/adaptation_step_0000.pt';s=torch.load(cp,map_location='cpu',weights_only=True)
 assert s['adaptation_step']==0 and s['signature']==signature('train')
 for k,m in models.items():m.load_state_dict(s[k])
 restore_buffers(models,s['buffers']);restore_rng(s['crop_rng']);before={k:tensor_hash(m.state_dict()) for k,m in models.items()};oh=tensor_hash(official.state_dict())
 row=load(ROOT/'plans/shared.json')['updates'][0];buf=buffer_state(models)
 f=PlanFrames(row,'cuda:0');base,_,trace=crop_c1(ctx,f,row['external_qp']);f.close();after_rng=rng_state();after_buf=buffer_state(models)
 key=f'{row["domain"]}:{row["external_qp"]}';_,ref=get_stratum(key,'cuda:0');old=to(s['EMA'][key],'cuda:0')
 restore_rng(s['global_rng']);restore_buffers(models,buf);f=PlanFrames(row,'cuda:0',full=True);z,_,_,_=full_pass(ctx,f,row['external_qp'],official);f.close()
 objectives=('FD',) if branch==BRANCHES[2] else ('mean','CORAL');cal=load(ROOT/'calibration/final.json');weights=dict(FD=cal['lambda_FD'],mean=cal['alpha_mu'],CORAL=cal['alpha_cov']);fg,_,_=feature_derivatives(z,old,ref,objectives)
 restore_rng(s['global_rng']);restore_buffers(models,buf);f=PlanFrames(row,'cuda:0',full=True);_,_,unit,replay=full_pass(ctx,f,row['external_qp'],official,fg,expected=z);f.close()
 extra=[sum(weights[k]*unit[k][j] for k in objectives) for j in range(len(params))]
 gr=[a+b for a,b in zip(base,extra)];saved=[g.clone() for g in gr];pre=norm(saved);merge_error=norm([g-a-e for g,a,e in zip(saved,base,extra)])/pre
 for p,g in zip(params,gr):p.grad=g
 reported=float(torch.nn.utils.clip_grad_norm_(params,1.));clip=min(1.,1./(reported+1e-6));post=norm([p.grad for p in params]);clip_error=norm([p.grad-g*clip for p,g in zip(params,saved)])/post
 legacy=norm([g-a-e for g,a,e in zip(gr,base,extra)])/norm(gr)
 original=load(ROOT/'branches'/branch/'receipts/0001.json');repeat={k:abs(v-original[k])/max(abs(original[k]),1e-30) for k,v in dict(C1_full_gradient_norm=norm(base),additional_gradient_norm=norm(extra),global_preclip_norm=pre).items()}
 assert merge_error<=1e-6 and clip_error<=1e-6 and all(v<=1e-4 for v in repeat.values()),(merge_error,clip_error,repeat)
 for p in params:p.grad=None
 restore_buffers(models,s['buffers']);restore_rng(s['crop_rng']);check_frozen(ctx)
 assert before=={k:tensor_hash(m.state_dict()) for k,m in models.items()} and oh==tensor_hash(official.state_dict())
 dump(ROOT/'audits'/f'gradient_clip_replay_{branch}.json',dict(status='PASS',branch=branch,checkpoint=str(cp),checkpoint_sha256=sha(cp),physical_GPU=gpu,optimizer_updates=0,EMA_commits=0,parameters_and_buffers_unchanged=True,preclip_vector_merge_relative_error=merge_error,postclip_vector_scaling_relative_error=clip_error,first_step_norm_repeat_relative_errors=repeat,legacy_alias_diagnostic_reproduced=legacy,legacy_recorded_value=original['gradient_formula_error'],global_preclip_norm=pre,global_postclip_norm=post,clip_coefficient=clip,replay=replay,additional_modules=groups(models,extra)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=BRANCHES[2:]);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),default=4);ap.add_argument('--aggregate',action='store_true');a=ap.parse_args()
 aggregate() if a.aggregate else probe(a.branch,a.gpu)
