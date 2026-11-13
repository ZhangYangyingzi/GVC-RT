"""One bounded G11 trainer; every formal update is atomically resumable."""
from util import *
import torch,statistics,gc,traceback
from model import load_cref,allow_receiver,native_state,load_native,frozen_hashes,G11
from data import Data,cache_dir
from objective import *
from evaluate import Evaluator
MILESTONES=[0,50,100,200,300]

def calibrate_means(reference,quality):
 p=ROOT/'audits/budget_calibration.json'
 if p.exists():return load(p)['means']
 es=[]
 for clip in load(ROOT/'sample_schedule.json')['calibration']:
  wait_file(cache_dir(clip['video'])/'completion.json','WAITING_FOR_CALIBRATION_CACHE');es.append(reference_errors(quality,reference,clip))
 means={k:statistics.mean(e[k] for e in es) for k in METRICS};assert all(math.isfinite(v) and v>1e-8 for v in means.values());dump(p,dict(status='PASS',training_only=True,calibration=load(ROOT/'sample_schedule.json')['calibration'],errors=es,means=means,schedule_sha256=sha(ROOT/'sample_schedule.json')));return means

def calibrate_lambda(branch,data,quality,natives):
 state=rng_state();before=thash(branch.state_dict().items())+thash(native_state(data.c).items());rows=[];params=[p for n,p in natives]
 for clip in load(ROOT/'sample_schedule.json')['calibration']:
  heartbeat('CALIBRATING_LAMBDA',committed_updates=50,sample_id=clip['sample_id']);ys,aux,states,resets=rollout(branch,data,clip,True);es,sp,yn,gt=errors(quality,ys,data,clip);temp,counts=temporal_loss(yn,gt,data,clip,resets);a=torch.autograd.grad(sp,params,retain_graph=True);b=torch.autograd.grad(temp,params);na,nb=grad_norm(a),grad_norm(b);assert math.isfinite(na+nb) and na>1e-12 and nb>1e-12,('LAMBDA_CALIBRATION_INVALID_GRADIENT',na,nb);ratio=na/(nb+1e-12);cos=sum(float(x.detach().double().mul(y.detach().double()).sum()) for x,y in zip(a,b))/(na*nb);rows.append(dict(sample_id=clip['sample_id'],video=clip['video'],grad_sp=na,grad_temp=nb,ratio=ratio,cosine=cos,valid_counts=counts));del ys,aux,states,es,sp,temp,yn,gt,a,b;gc.collect()
 value=.2*statistics.median(r['ratio'] for r in rows);assert math.isfinite(value) and value>0;assert before==thash(branch.state_dict().items())+thash(native_state(data.c).items());restore_rng(state);result=dict(status='PASS',step=50,lambda_star=value,coefficient=.2,rows=rows,no_parameter_update=True,native_receiver_parameters_only=True,training_only=True);dump(ROOT/'audits/lambda_calibration.json',result);return result

def stop_decision(step,rows):
 cref={r['sequence']:r for r in rows['C_ref']};cand=rows['G11'];evidence={}
 for key in METRICS:
  a=statistics.mean(r[key] for r in cand);b=statistics.mean(r[key] for r in cref.values());relative=a/max(b,1e-12)-1;per=[dict(sequence=r['sequence'],candidate=r[key],C_ref=cref[r['sequence']][key],relative=r[key]/max(cref[r['sequence']][key],1e-12)-1) for r in cand];count=sum(x['relative']>.01 for x in per);evidence[key]=dict(candidate_video_equal_mean=a,C_ref_video_equal_mean=b,mean_relative_worsening=relative,sources_worse_1pct=count,per_source=per,violation=(relative>.03 and count>=4))
 previous=None
 if step in [200,300]:
  pp=ROOT/'evaluation'/f'decision_{100 if step==200 else 200:04d}.json';previous=load(pp)
 stopped=any(evidence[k]['violation'] and previous['evidence'][k]['violation'] for k in METRICS) if previous else False
 result=dict(status='PASS',step=step,stop=stopped,stop_reason='SPATIAL_BUDGET_TWO_CHECKPOINTS' if stopped else 'CONTINUE' if step<300 else 'BUDGET_300',evidence=evidence,eligible_from=100,validation_sources=6,protocol_sha256=protocol_sha());dump(ROOT/'evaluation'/f'decision_{step:04d}.json',result);return result

def run():
 setup_torch();assert load(ROOT/'audits/preflight.json')['status']=='PASS';candidate=load_cref();reference=load_cref();natives=allow_receiver(candidate);seed(20261112);branch=G11().cuda().float();initial=read(ROOT/'checkpoints/initial.pt');branch.load_state_dict(initial['branch']);load_native(candidate,initial['native']);opt=torch.optim.AdamW([dict(params=branch.parameters(),lr=5e-5,group_name='new'),dict(params=[p for n,p in natives],lr=5e-6,group_name='native_receiver')],betas=(.9,.999),eps=1e-8,weight_decay=1e-4);opt.load_state_dict(initial['optimizer']);restore_rng(initial['RNG']);quality=quality_models();data=Data(candidate);refdata=Data(reference);means=calibrate_means(refdata,quality);mu=torch.full((3,),.1,device='cuda');lam=None;step=0;seconds=0.;guard=frozen_hashes(candidate);refhash=thash(reference['pm'].state_dict().items())+thash(reference['S'].state_dict().items());versions={n:p._version for n,p in candidate['pm'].named_parameters() if not p.requires_grad};implementation=source_fingerprint();latest=ROOT/'checkpoints/latest.pt';evaluator=None;baseline={};stop=dict(stop=False)
 def frozen_check(full=False):
  for n,p in candidate['pm'].named_parameters():
   if not p.requires_grad:assert p.grad is None and p._version==versions[n],('FROZEN_MUTATION',n)
  assert all(p.grad is None for p in candidate['S'].parameters());assert not candidate['pm'].dpb
  if full:assert frozen_hashes(candidate)==guard and thash(reference['pm'].state_dict().items())+thash(reference['S'].state_dict().items())==refhash
 def checkpoint(milestone=False):
  frozen_check(full=milestone);rec=dict(branch=branch.state_dict(),native_receiver=native_state(candidate),optimizer=opt.state_dict(),step=step,committed_updates=step,LR=[g['lr'] for g in opt.param_groups],RNG=rng_state(),mu=mu.cpu(),lambda_calibration=lam,budget_means=means,schedule_position=step,schedule_sha256=sha(ROOT/'sample_schedule.json'),protocol_sha256=protocol_sha(),implementation=implementation,initialization_sha256=sha(ROOT/'checkpoints/initial.pt'),frozen_hashes=guard,training_seconds=seconds)
  save(latest,rec)
  if milestone:save(ROOT/'checkpoints'/f'step_{step:04d}.pt',rec)
  heartbeat('TRAINING',committed_updates=step,training_seconds=seconds,mu=mu.tolist(),lambda_star=lam['lambda_star'] if lam else None)
 def evaluation():
  nonlocal evaluator,baseline
  evaluation_rng=rng_state()
  if evaluator is None:evaluator=Evaluator(quality)
  if not baseline:
   baseline['C_ref']=evaluator.validation('C_ref',0,refdata,checkpoint_sha=load(ROOT/'weight_manifest.json')['D12_S_900']['sha256']);baseline['B300']=evaluator.validation('B300',0,refdata,checkpoint_sha=load(ROOT/'weight_manifest.json')['B300']['sha256']);baseline['Original']=evaluator.validation('Original',0,refdata,checkpoint_sha=digest({k:load(ROOT/'weight_manifest.json')[k]['sha256'] for k in ['I','P']}))
  rows=evaluator.validation('G11',step,data,branch,sha(ROOT/'checkpoints'/f'step_{step:04d}.pt'));frozen_check(True)
  if step==0:
   ref={r['sequence']:r for r in baseline['C_ref']}
   for r in rows:
    for key in ['LPIPS','DISTS','RGB_MSE','highpass_signed_error_MSE']:assert abs(r[key]-ref[r['sequence']][key])<=1e-7,('STEP0_METRIC_MISMATCH',r['sequence'],key)
  decision=stop_decision(step,dict(baseline,G11=rows));restore_rng(evaluation_rng);return decision
 if latest.with_suffix('.pt.json').exists():
  cp=read(latest);assert cp['protocol_sha256']==protocol_sha() and cp['schedule_sha256']==sha(ROOT/'sample_schedule.json') and cp['implementation']==implementation;branch.load_state_dict(cp['branch']);load_native(candidate,cp['native_receiver']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['RNG']);mu=cp['mu'].cuda();lam=cp['lambda_calibration'];step=cp['step'];seconds=cp['training_seconds'];versions={n:p._version for n,p in candidate['pm'].named_parameters() if not p.requires_grad}
 else:checkpoint(True)
 if step==50 and lam is None:lam=calibrate_lambda(branch,data,quality,natives);checkpoint(True)
 if step in MILESTONES:stop=evaluation()
 while step<300 and not stop['stop']:
  k=step+1;clip=load(ROOT/'sample_schedule.json')['updates'][step];wait_file(cache_dir(clip['video'])/'completion.json','WAITING_FOR_CACHE');heartbeat('TRAINING',committed_updates=step,next_sample_id=clip['sample_id']);torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
  factor=min(k/20,1.);opt.param_groups[0]['lr']=5e-5*factor;opt.param_groups[1]['lr']=5e-6*factor;weight=0. if k<=50 else lam['lambda_star']*min((k-50)/50,1.);loss=clip_loss(branch,data,refdata,quality,clip,mu,means,weight);loss['loss'].backward();grad=grad_groups(branch,natives);clip_norm=float(torch.nn.utils.clip_grad_norm_(list(branch.parameters())+[p for n,p in natives],1.,error_if_nonfinite=True));opt.step();assert all(torch.isfinite(p).all() for p in list(branch.parameters())+[p for n,p in natives]),'NONFINITE_PARAMETERS';mu=(mu+.01*loss['g'].detach()).clamp(0,1);step=k;torch.cuda.synchronize();elapsed=time.perf_counter()-t;seconds+=elapsed;frozen_check();checkpoint(step in MILESTONES)
  aux=loss['stats'][-1];act=dict(offset_RMS=float(aux['offset'].detach().square().mean().sqrt()),offset_max_abs=float(aux['offset'].detach().abs().max()),mask_mean=float(aux['modulation'].detach().mean()),valid_fraction=float(aux['v'].mean()),correction_RMS=float(aux['correction'].detach().square().mean().sqrt()));event(ROOT/'logs/train.jsonl',stage='UPDATE_COMMITTED',step=step,sample_id=clip['sample_id'],video=clip['video'],start=clip['start'],ROI=[clip['x'],clip['y'],256,256],loss=float(loss['loss'].detach()),L_sp=float(loss['L_sp'].detach()),L_temp=float(loss['L_temp'].detach()),L_budget=float(loss['L_budget'].detach()),errors={k:float(v.detach()) for k,v in loss['errors'].items()},reference=loss['reference'],g=loss['g'].detach().tolist(),mu=mu.tolist(),lambda_temp=weight,LR=[g['lr'] for g in opt.param_groups],gradient_groups=grad,gradient_before_clip=clip_norm,valid_counts=loss['valid_counts'],activation=act,seconds=elapsed,training_seconds=seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30);del loss;gc.collect();torch.cuda.empty_cache()
  if step==20:dump(ROOT/'audits/ETA_after20.json',dict(committed_updates=20,training_seconds=seconds,mean_compute_seconds=seconds/20,remaining_compute_seconds_estimate=(300-step)*seconds/20,cache_eval_wait_excluded=True))
  if step==50:lam=calibrate_lambda(branch,data,quality,natives);checkpoint(True)
  if step in MILESTONES:stop=evaluation()
 frozen_check(True);reason=stop['stop_reason'];dump(ROOT/'completion.json',dict(status='COMPLETE',committed_updates=step,stop_reason=reason,decision=stop,training_seconds=seconds,frozen_hashes=guard,frozen_unchanged=True,codec_invariant=True,device=device_info(),new_parameters=sum(p.numel() for p in branch.parameters()),native_parameters=sum(p.numel() for n,p in natives),mu=mu.tolist(),lambda_calibration=lam));heartbeat('TRAINING_AND_VALIDATION_COMPLETE',committed_updates=step,stop_reason=reason)
if __name__=='__main__':
 with lock('trainer.lock'):
  try:run()
  except BaseException:
   dump(ROOT/'receipts/train.error.json',dict(status='STOPPED_WITH_ERROR',error=traceback.format_exc(),pid=os.getpid(),time=stamp()));raise
