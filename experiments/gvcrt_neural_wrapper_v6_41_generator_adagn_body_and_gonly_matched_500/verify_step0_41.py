"""No-update source-step0 G gradient and full-FD path comparison."""
from io32 import *
import traceback

def main():
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 import torch
 from training32 import setup,official_model,restore_rng,restore_buffers,buffer_state,crop_c1,full_pass,norm
 from calibration32b import get_stratum,feature_derivatives
 from distribution32 import to
 from frames32 import PlanFrames
 source=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt'
 s=torch.load(source,map_location='cpu',weights_only=True)
 ctx=setup(BRANCHES[0]);v4,im,pm,models,quality,common,_,_=ctx
 official=official_model();allmods=dict(models,im=im,pm=pm,official=official,lp=quality[0],dists=quality[1])
 for k,m in models.items():m.load_state_dict(s[k],strict=True)
 restore_buffers(allmods,s['buffers']);restore_rng(s['crop_rng'])
 row=source_row(load(ROOT/'plans/shared.json')['updates'][0]);q=row['external_qp']
 def one(bg):
  models['bridge'].requires_grad_(bg)
  params=[p for k in ('bridge','generator') for p in models[k].parameters() if p.requires_grad]
  c=(v4,im,pm,models,quality,common,params,ctx[-1]);restore_buffers(allmods,s['buffers']);restore_rng(s['crop_rng'])
  first_buffer=buffer_state(allmods)
  f=PlanFrames(row,'cuda:0');g0,values,trace=crop_c1(c,f,q);f.close()
  restore_rng(s['global_rng']);restore_buffers(allmods,first_buffer)
  f=PlanFrames(row,'cuda:0',full=True);z,gt,_,_=full_pass(c,f,q,official);f.close()
  key=f'{row["domain"]}:{q}';old=to(s['EMA'][key],'cuda:0');_,ref=get_stratum(key,'cuda:0')
  fg,ls,_=feature_derivatives(z,old,ref,('FD',))
  restore_rng(s['global_rng']);restore_buffers(allmods,first_buffer)
  f=PlanFrames(row,'cuda:0',full=True);_,_,fd,replay=full_pass(c,f,q,official,fg['FD'],expected=z);f.close()
  gen_count=len(list(models['generator'].parameters()))
  return dict(g0=[x.detach().cpu() for x in g0[-gen_count:]],fd=[x.detach().cpu() for x in fd[-gen_count:]],
   values=values,FD=ls['FD'],z=z,gt=gt,replay=replay,bridge_grad_norm=norm(g0[:-gen_count]),
   generator_grad_norm=norm(g0[-gen_count:]),FD_generator_grad_norm=norm(fd[-gen_count:]))
 bg=one(True);go=one(False)
 def compare(a,b):
  diff=[x.double()-y.double() for x,y in zip(a,b)]
  return dict(relative_L2=norm(diff)/max(norm(a),1e-12),maxabs=max(float(x.abs().max()) for x in diff),
   reference_norm=norm(a),candidate_norm=norm(b))
 crop=compare(bg['g0'],go['g0']);fd=compare(bg['fd'],go['fd']);lam=load(ROOT/'effective_config.json')['lambda_FD']
 combined=compare([a+lam*b for a,b in zip(bg['g0'],bg['fd'])],[a+lam*b for a,b in zip(go['g0'],go['fd'])])
 assert crop['relative_L2']<=1e-4 and fd['relative_L2']<=1e-4 and combined['relative_L2']<=1e-4
 assert bg['z'].shape==go['z'].shape and torch.allclose(bg['z'],go['z'],atol=1e-4,rtol=1e-4)
 value_errors={k:abs(bg['values'][k]-go['values'][k]) for k in bg['values'] if isinstance(bg['values'][k],(float,int))}
 print('STEP0_VALUE_ERRORS',value_errors,'CROP',crop,'FD',fd,'COMBINED',combined,flush=True)
 assert all(v<=1e-5 for k,v in value_errors.items() if k not in ('base_gradient_norm','reconstruction_gradient_norm','stat_gradient_norm','C1_gradient_norm','stat_base_cosine'))
 assert abs(bg['FD']-go['FD'])<=1e-5
 assert bg['generator_grad_norm']>0 and go['generator_grad_norm']>0
 assert bg['FD_generator_grad_norm']>0 and go['FD_generator_grad_norm']>0
 dump(ROOT/'audits/generator_gradient_equivalence.json',dict(status='PASS',source_checkpoint=str(source),source_sha256=sha(source),
  plan_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  crop_generator=crop,FD_generator=fd,combined_generator=combined,loss_values_bg=bg['values'],
  loss_values_gonly=go['values'],FD_bg=bg['FD'],FD_gonly=go['FD'],source_bridge_C1_norm=bg['bridge_grad_norm'],
  frozen_bridge_C1_norm=0,full_replay_checks=[len(bg['replay']),len(go['replay'])],optimizer_updates=0))
 print('STEP0_GRAD_PASS',crop['relative_L2'],fd['relative_L2'],combined['relative_L2'],flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  error(dict(stage='step0_gradient_verification',traceback=traceback.format_exc(),time=time.time()))
  raise
