"""Native-size independent causal evaluation; no test135 data and no model selection."""
from util import *
import argparse,math,statistics,gc

def jobs():
 out=[]
 for r in manifest()['dev']+load(ROOT/'additional_development_manifest.json')['dev']:
  for qp in [5,0,9]:
   for label in ['Original','B300']:out.append(dict(id=r['id'],qp=qp,label=label,step=0))
   for group in ['S_current','U_unaligned','T_aligned']:
    for step in ([0,50,100,200,300] if qp==5 and r['id'] in [v['id'] for v in manifest()['dev']] else [300]):out.append(dict(id=r['id'],qp=qp,label=group,step=step))
 return out

def highpass(x):
 import torch.nn.functional as F
 k=x.new_tensor([1,4,6,4,1])/16;k=(k[:,None]*k[None,:])[None,None].expand(3,1,5,5)
 return x-F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),k,groups=3)
def tag(j):return f"{j['label']}_{j['step']:04d}_{j['id']}_q{j['qp']}"

def ready_jobs(shard,n,stream=False):
 pending=jobs()[shard::n]
 while pending:
  progressed=False
  for j in list(pending):
   if (ROOT/'metrics'/f'{tag(j)}.complete.json').exists():pending.remove(j);progressed=True;continue
   cp=ROOT/'checkpoints'/j['label']/f"step_{j['step']:04d}.pt.json"
   if j['label'] not in ['Original','B300'] and not cp.exists():continue
   yield j
   if (ROOT/'metrics'/f'{tag(j)}.complete.json').exists():pending.remove(j);progressed=True
  if not stream:break
  if not progressed:heartbeat('async_evaluation','WAITING_FOR_CHECKPOINT_OR_OTHER_EVALUATOR',remaining=len(pending));time.sleep(3)

def worker(shard,n,stream=False):
 import torch
 import torch.nn.functional as F
 from model import AlignedLatent,bridge,decode
 from train import frozen,versionguard,assertguard
 from restore import restore_b300
 from objective import quality_models
 from temporal import temporal_errors,sobel
 setup();seed();c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);guard=versionguard(c);initial=frozen(c);stats=readtensor(ROOT/'cache/statistics.pt');quality=quality_models()
 sys.path.insert(0,str(ROOT/'vendor/flo'));import flolpips_compat
 pwc,flo=flolpips_compat.load_models(torch.device('cuda:0'));metricsha={k:thash(v.state_dict().items()) for k,v in [('LPIPS',quality[0]),('DISTS',quality[1]),('PWC',pwc),('FloLPIPS',flo)]}
 for j in ready_jobs(shard,n,stream):
  name=tag(j);proof=ROOT/'metrics'/f'{name}.complete.json'
  if proof.exists():continue
  claim=(ROOT/'locks'/f'evaluation_{name}.lock').open('a')
  try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:claim.close();continue
  heartbeat('evaluate'+str(shard),'WAITING_FOR_CACHE',job=j);d=waitcache(j['id'],'evaluate'+str(shard),j['qp']);row=d['row'];h,w=d['native_shape'];g=c['models']['generator'];g.reset_temporal_buffer();m=None;cp=None
  if j['label'] not in ['Original','B300']:
   cp=ROOT/'checkpoints'/j['label']/f"step_{j['step']:04d}.pt";ck=readtensor(cp);assert ck['protocol_sha256']==check_protocol();m=AlignedLatent(stats,j['label']).cuda();m.load_state_dict(ck['model']);m.eval();m.reset()
  t=time.time();frames=[];media=[];heat=[];prev=None;prevgt=None;reconsec=0.
  with torch.no_grad():
   for i in range(len(d['features'])):
    heartbeat('evaluate'+str(shard),'EVALUATING',job=j,frame=i);gt=d['GT'][i].permute(2,0,1)[None].cuda().float()/255;a=None;torch.cuda.synchronize();tr=time.time()
    if i==0 or j['label']=='Original':y=d['original'][i].cuda()
    else:
     f=d['features'][i].cuda().float();l=bridge(c,f)
     if m is None:lp=l
     else:lp,a=m(l,f,d['actual_qps'][i],d['id'],i)
     y=decode(c,lp,d['actual_qps'][i])[...,:h,:w]
    torch.cuda.synchronize();reconsec+=time.time()-tr;err=y-gt;mse=float(err.square().mean());hp,hg=highpass(y),highpass(gt)
    r=dict(**j,frame=i,source_frame=row['source_frame_indices'][i],actual_QP=d['actual_qps'][i],bpp=d['bpp'],frame_bits=d['bits'][i]+(len(Path(d['stream_path']).read_bytes())*8-sum(d['bits']) if i==0 else 0),PSNR=-10*math.log10(max(mse,1e-15)),RGB_MSE=mse,RGB_L1=float(err.abs().mean()),LPIPS=float(quality[0](y,gt,normalize=True).mean()),DISTS=float(quality[1](y,gt,require_grad=False).mean()),edge_strength=float(sobel(y).abs().mean()),GT_edge_strength=float(sobel(gt).abs().mean()),high_frequency_error_MSE=float((hp-hg).square().mean()),high_frequency_energy=float(hp.square().mean()),GT_high_frequency_energy=float(hg.square().mean()))
    if prev is not None:
     back=d['flow'][i].cuda();valid=d['valid'][i].cuda();mc,ms,v=temporal_errors(y,prev,gt,prevgt,back,valid);flowdiff=pwc(prevgt,gt)-pwc(prev,y);weight=float(flowdiff.square().sum(1,keepdim=True).sqrt().sum());assert math.isfinite(weight) and weight>0,'UNDEFINED_FLOLPIPS'
     r.update(FloLPIPS=float(flo(prevgt,prev,flowdiff,normalize=True).mean()),MC_RGB=float(mc) if mc is not None else None,MC_Sobel=float(ms) if ms is not None else None,valid_supervision_fraction=v,invalid_RGB_L1=float(err.abs().mul(~valid).sum()/((~valid).sum()*3)) if (~valid).any() else None)
    if a is not None:
     q=a['q'];delta=a['delta'];l=a['original_l'];lp=a['corrected_l'];r.update(q_mean=float(q.mean()),q_p05=float(q.quantile(.05)),q_p25=float(q.quantile(.25)),q_p50=float(q.quantile(.5)),q_p75=float(q.quantile(.75)),q_p95=float(q.quantile(.95)),q_below01=float((q<.01).float().mean()),q_above99=float((q>.99).float().mean()),displacement_mean=float(a['u'].square().sum(1).sqrt().mean()),geometric_history_valid_fraction=float(a['v'].mean()),delta_l_RMS=float(delta.square().mean().sqrt()),latent_drift_RMS=float((lp-l).square().mean().sqrt()))
     for channel in range(18):
      r.update({f'latent_mean_{channel}':float(l[:,channel].mean()),f'corrected_mean_{channel}':float(lp[:,channel].mean()),f'latent_std_{channel}':float(l[:,channel].std(unbiased=False)),f'corrected_std_{channel}':float(lp[:,channel].std(unbiased=False))})
     if j['qp']==5 and j['step'] in [100,300]:heat.append(dict(frame=i,q=q.cpu(),u=a['u'].cpu(),correction=(q*delta).square().mean(1,keepdim=True).sqrt().cpu()))
    assert all(not isinstance(v,float) or math.isfinite(v) for v in r.values()),'NONFINITE_METRIC';frames.append(r);prev=y;prevgt=gt
    if j['qp']==5 and (j['label'] in ['Original','B300'] or j['step']==300 or (j['label']=='T_aligned' and j['step']==100)) and j['id'] in [manifest()['dev'][k]['id'] for k in [0,2]]:media.append(y.mul(255).round().clamp(0,255).byte().cpu())
   assertguard(c,guard)
  csvwrite(ROOT/'metrics'/f'{name}_frames.csv',frames)
  summary={**j,'bpp':d['bpp'],'frames':len(frames),'transitions':len(frames)-1,'dataset':row.get('domain','ulong')}
  for k in frames[0].keys()|set().union(*(r.keys() for r in frames)):
   if k not in summary and k not in ['frame','source_frame','actual_QP']:
    vals=[r[k] for r in frames if k in r and isinstance(r[k],(int,float))];summary[k]=sum(vals)/len(vals) if vals else None
  summary['PSNR']=-10*math.log10(summary['RGB_MSE'])
  if media:save(ROOT/'cache/media'/f'{name}.pt',dict(frames=torch.cat(media),job=j,fps=row['source_fps'],ROI=row['ROI'],source_frames=row['source_frame_indices']))
  if heat:save(ROOT/'cache/heatmaps'/f'{name}.pt',dict(frames=heat,job=j))
  dump(proof,dict(status='PASS',job=j,summary=summary,checkpoint_sha256=None if cp is None else sha(cp),protocol_sha256=check_protocol(),cache_sha256=load(cachepath(j['id'],j['qp']).with_suffix('.pt.json'))['sha256'],stream_sha256=d['stream_sha256'],metric_model_sha256=metricsha,reconstruction_seconds=reconsec,evaluation_seconds=time.time()-t,codec_seconds=d['decode_seconds'],input_hashes=d['codec_feature_hashes'],independent_candidate_TCG_roll=True,precision='matched FP32 receiver; original FP16 codec',Original_independent=True));event('EVALUATION_COMMITTED',job=j,seconds=time.time()-t)
  claim.close();del d,prev,prevgt,media,heat,m;gc.collect();torch.cuda.empty_cache()
 assert frozen(c)==initial;heartbeat('async_evaluation' if stream else 'evaluate'+str(shard),'COMPLETE')

def orchestrate():
 with lock('evaluation.lock'):
  # GPU3 may still be writing native development caches. Three free training GPUs are reused.
  ps=[]
  for shard,gpu in enumerate([0,1,2]):
   h=(ROOT/'logs'/f'evaluate{shard}.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));p=subprocess.Popen([PYTHON,'-B','evaluate.py','--shard',str(shard),'--workers','3'],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT);ps.append((p,h));event('EVALUATOR_STARTED',child_pid=p.pid,physical_gpu=gpu,shard=shard)
  while any(p.poll() is None for p,h in ps):heartbeat('evaluation','EVALUATING',pids=[p.pid for p,h in ps],complete=len(list((ROOT/'metrics').glob('*.complete.json'))));time.sleep(5)
  for p,h in ps:h.close();assert p.returncode==0,('EVALUATION_WORKER_FAILED',p.pid,p.returncode)
  while not all((ROOT/'metrics'/f'{tag(j)}.complete.json').exists() for j in jobs()):heartbeat('evaluation','WAITING_FOR_ASYNC_POINTS');time.sleep(3)
  summaries=[load(ROOT/'metrics'/f'{tag(j)}.complete.json')['summary'] for j in jobs()];csvwrite(ROOT/'metrics/per_video_metrics.csv',summaries);dump(ROOT/'receipts/evaluation.complete.json',dict(points=len(summaries),jobs=jobs(),protocol_sha256=check_protocol(),time=time.time()))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--stream',action='store_true');a.add_argument('--shard',type=int);a.add_argument('--workers',type=int,default=3);p=a.parse_args()
 if p.shard is None:orchestrate()
 else:worker(p.shard,p.workers,p.stream)
