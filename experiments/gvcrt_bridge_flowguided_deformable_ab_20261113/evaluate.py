"""64-frame native causal evaluations and fixed checkpoint/beta queues."""
from util import *
import argparse,math,statistics,gc
GROUPS=['A_zero_base','B_flow_guided'];STEPS=[0,50,100,200,300]
def tag(j):return f"{j['label']}_s{j['step']:04d}_{j['id']}_q{j['qp']}_b{j.get('beta',1):g}"+(f"_blur{j['sigma']}" if j.get('sigma') else '')
def proofpath(j):return ROOT/'evaluation/points'/f'{tag(j)}.json'
def available_jobs():
 jobs=[]
 for row in devrows():
  for qp in [5,0,9]:
   for label in ['Original','B300']:jobs.append(dict(label=label,step=0,id=row['id'],qp=qp,beta=1.))
 for group in GROUPS:
  for step in STEPS:
   if (ROOT/'checkpoints'/group/f'step_{step:04d}.pt.json').exists():
    for row in devrows()[:4]:jobs.append(dict(label=group,step=step,id=row['id'],qp=5,beta=1.))
 done=all((ROOT/'checkpoints'/g/'completion.json').exists() for g in GROUPS)
 if done:
  common=sorted(set.intersection(*[{s for s in STEPS if (ROOT/'checkpoints'/g/f'step_{s:04d}.pt.json').exists()} for g in GROUPS]));fixed=sorted(set(([100] if 100 in common else [])+[common[-1]]))
  for step in fixed:
   for group in GROUPS:
    for row in devrows():
     for qp in [0,5,9]:jobs.append(dict(label=group,step=step,id=row['id'],qp=qp,beta=1.))
    for row in [devrows()[k] for k in [0,2]]:
     for beta in [0.,.25,.5,1.]:jobs.append(dict(label=group,step=step,id=row['id'],qp=5,beta=beta))
  for row in devrows()[:4]:
   for sigma in [.25,.5,.75]:jobs.append(dict(label='B300_blur',step=0,id=row['id'],qp=5,beta=1.,sigma=sigma))
  dump(ROOT/'evaluation/fixed_points.json',dict(last_common_checkpoint=common[-1],fixed_points=fixed,group_completion={g:load(ROOT/'checkpoints'/g/'completion.json') for g in GROUPS}))
 return list({tag(j):j for j in jobs}.values()),done

def highpass(x):
 import torch.nn.functional as F
 k=x.new_tensor([1,4,6,4,1])/16;k=(k[:,None]*k[None,:])[None,None].expand(3,1,5,5);return x-F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),k,groups=3)
def blur(x,sigma):
 import torch,torch.nn.functional as F
 radius=math.ceil(3*sigma);a=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype);k=torch.exp(-a.square()/(2*sigma*sigma));k=k/k.sum();kernel=(k[:,None]*k[None,:])[None,None].expand(3,1,-1,-1);return F.conv2d(F.pad(x,(radius,radius,radius,radius),mode='replicate'),kernel,groups=3)
def snapshot(g):return {n:None if getattr(g,n)._previous is None else getattr(g,n)._previous.detach().clone() for n in ['tcg_low','tcg_mid','tcg_out']}
def restore(g,s):
 for n,v in s.items():getattr(g,n)._previous=v
class VideoSink:
 def __init__(self,path,h,w,fps):
  self.path=path;path.parent.mkdir(parents=True,exist_ok=True);self.tmp=path.with_suffix('.part.mkv');self.hash=hashlib.sha256();self.frames=0;self.p=subprocess.Popen(['ffmpeg','-y','-v','error','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{w}x{h}','-framerate',fps,'-i','pipe:0','-c:v','ffv1','-level','3','-g','1','-pix_fmt','bgr0','-color_range','pc','-colorspace','rgb','-threads','2',str(self.tmp)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
 def write(self,x):
  raw=x[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy().tobytes();self.p.stdin.write(raw);self.hash.update(raw);self.frames+=1
 def close(self):
  self.p.stdin.close();error=self.p.stderr.read();code=self.p.wait();assert code==0,error.decode();self.tmp.replace(self.path);return dict(path=str(self.path),sha256=sha(self.path),RGB8_sha256=self.hash.hexdigest(),frames=self.frames)

class Evaluator:
 def __init__(self):
  import torch
  from restore import restore_b300
  from reference import coexist_original,load_raft
  from objective import quality_models
  setup();self.c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);self.rt,self.im,self.pm=coexist_original();self.raft,self.transform=load_raft();self.q=quality_models();sys.path.insert(0,str(ROOT/'vendor/flo'));import flolpips_compat
  self.pwc,self.flo=flolpips_compat.load_models(torch.device('cuda:0'));self.torch=torch;self.metric_hashes={k:thash(v.state_dict().items()) for k,v in [('LPIPS',self.q[0]),('DISTS',self.q[1]),('PWC',self.pwc),('FloLPIPS',self.flo)]};self.frozen=thash(self.c['pm'].state_dict().items());self.stats=stats()
 def point(self,j):
  torch=self.torch;import torch.nn.functional as F
  from model import DeformableHistory,bridge,decode
  from stream import raw_frames,stream_file,decode_frames,ref_sidecar,gt_supervision
  from temporal import temporal_errors,sobel
  name=tag(j);path=proofpath(j)
  if path.exists():return
  row=next(r for r in devrows() if r['id']==j['id']);arrays=raw_frames(row);h,w=arrays.shape[1:3];stream,meta=stream_file(self.rt,self.im,self.pm,row,j['qp'],arrays);stream_sha=sha(stream);g=self.c['models']['generator'];g.reset_temporal_buffer();m=None;cp=None;candidate_state=snapshot(g);base_state=snapshot(g);identity=(j['label'] in GROUPS and (j['step']==0 or j['beta']==0));rows=[];feature_hashes=[];prev_ref=None;prev_y=None;prev_gt=None;sink=None;video=None
  if j['label'] in GROUPS:
   cp=ROOT/'checkpoints'/j['label']/f"step_{j['step']:04d}.pt";ck=readtensor(cp);assert ck['protocol_sha256']==check_protocol();m=DeformableHistory(self.stats,j['label']).cuda();m.load_state_dict(ck['model']);m.eval();m.reset()
  selected=[devrows()[k]['id'] for k in [0,2]];export=j['qp']==5 and j['id'] in selected and (j['label'] in ['Original','B300'] or (j['label'] in GROUPS and j['step']>=100 and j['beta']>0 and all((ROOT/'checkpoints'/q/'completion.json').exists() for q in GROUPS)))
  if export:sink=VideoSink(ROOT/'videos'/j['id']/f'{name}.mkv',h,w,row['source_fps'])
  start=time.perf_counter();reconstruction_seconds=0.;decoded_ref_seconds=0.
  with torch.no_grad():
   for i,qp,f,ref,sps,codec_seconds in decode_frames(self.rt,self.im,self.pm,stream,meta):
    heartbeat('eval_gpu'+os.environ['CUDA_VISIBLE_DEVICES'],'EVALUATING',job=j,frame=i);gt=torch.from_numpy(arrays[i]).permute(2,0,1)[None].cuda().float()/255;aux=None;t=time.perf_counter()
    if i==0 or j['label']=='Original':y=ref[...,:h,:w]
    else:
     l=bridge(self.c,f.float())
     if m is not None:
      if prev_ref is None:b=torch.zeros(1,2,*ref.shape[-2:],device='cuda');e=torch.zeros_like(b)
      else:
       tr=time.perf_counter();b,e=ref_sidecar(row,j['qp'],i,ref,prev_ref,self.raft,self.transform);decoded_ref_seconds+=time.perf_counter()-tr
      lp,aux=m(l,b,e,qp,j['id'],i,j['beta'])
      if identity:
       restore(g,base_state);baseline=decode(self.c,l,qp)[...,:h,:w];base_state=snapshot(g);restore(g,candidate_state)
      y=decode(self.c,lp,qp)[...,:h,:w]
      if identity:assert torch.equal(y,baseline),('FULL64_IDENTITY_FAILURE',j,i);candidate_state=snapshot(g)
     else:y=decode(self.c,l,qp)[...,:h,:w]
    if j.get('sigma'):y=blur(y,j['sigma'])
    torch.cuda.synchronize();reconstruction_seconds+=time.perf_counter()-t;err=y-gt;mse=float(err.square().mean());hp,hg=highpass(y),highpass(gt)
    r=dict(**j,frame=i,source_frame=row['source_frame_indices'][i],actual_QP=qp,bpp=meta['bpp'],PSNR=-10*math.log10(max(mse,1e-15)),RGB_MSE=mse,RGB_L1=float(err.abs().mean()),LPIPS=float(self.q[0](y,gt,normalize=True).mean()),DISTS=float(self.q[1](y,gt,require_grad=False).mean()),edge_strength=float(sobel(y).abs().mean()),GT_edge_strength=float(sobel(gt).abs().mean()),HF_energy=float(hp.square().mean()),GT_HF_energy=float(hg.square().mean()),HF_GT_error=float((hp-hg).square().mean()),RWE=None,RWE_status='RWE_NOT_VERIFIED')
    if i>0:
     flow,valid=gt_supervision(row,i,arrays,self.raft,self.transform);a,b,_=temporal_errors(y,prev_y,gt,prev_gt,flow,valid);fd=self.pwc(prev_gt,gt)-self.pwc(prev_y,y);assert float(fd.square().sum(1).sqrt().sum())>0,'FLOLPIPS_ZERO_FLOW_DIFFERENCE';r.update(FloLPIPS=float(self.flo(prev_gt,prev_y,fd,normalize=True).mean()),MC_RGB=float(a) if a is not None else None,MC_Sobel=float(b) if b is not None else None)
    if aux is not None:
     o=aux['offset'];q=aux['q'];mod=aux['modulation'];r.update(offset_RMS=float(o.square().mean().sqrt()),offset_max_abs=float(o.abs().max()),offset_p95=float(o.abs().quantile(.95)),mask_mean=float(mod.mean()),q_mean=float(q.mean()),q_p05=float(q.quantile(.05)),q_p50=float(q.quantile(.5)),q_p95=float(q.quantile(.95)),point_valid=float(aux['point_validity'].float().mean()),center_valid=float(aux['v'].mean()),delta_l_RMS=float(aux['delta'].square().mean().sqrt()),correction_RMS=float((aux['corrected_l']-l).square().mean().sqrt()))
    assert all(not isinstance(v,float) or math.isfinite(v) for v in r.values()),'NONFINITE_METRIC';rows.append(r);feature_hashes.append(meta['features'][i]);prev_ref=ref.detach();prev_y=y.detach();prev_gt=gt.detach()
    if sink:sink.write(y)
   assert not self.c['pm'].dpb,'CANDIDATE_DPB_WRITE';assert thash(self.c['pm'].state_dict().items())==self.frozen
  if sink:video=sink.close()
  csvwrite(ROOT/'metrics/frames'/f'{name}.csv',rows);summary=dict(**j,dataset=row.get('domain','ulong'),frames=64,transitions=63,bpp=meta['bpp'])
  for k in set().union(*(r.keys() for r in rows)):
   if k not in summary and k not in ['frame','source_frame','actual_QP']:
    vals=[r[k] for r in rows if isinstance(r.get(k),(int,float))];summary[k]=statistics.mean(vals) if vals else None
  summary['PSNR']=-10*math.log10(summary['RGB_MSE']);dump(path,dict(status='PASS',job=j,summary=summary,stream_path=str(stream),stream_sha256=stream_sha,codec_feature_hashes=feature_hashes,checkpoint_sha256=sha(cp) if cp else None,protocol_sha256=check_protocol(),metric_hashes=self.metric_hashes,identity64_exact=identity,independent_candidate_state=True,Original_reference_independent=True,reference_flow_GT_free=True,video=video,evaluation_seconds=time.perf_counter()-start,reconstruction_seconds=reconstruction_seconds,decoded_reference_sidecar_seconds=decoded_ref_seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30));event('EVALUATION_COMMITTED',job=j,seconds=time.perf_counter()-start);del m,arrays;gc.collect();torch.cuda.empty_cache()

def worker(gpu):
 import torch
 setup();ev=None
 while not (ROOT/'receipts/evaluation_queue.complete.json').exists():
  jobs,done=available_jobs();eligible=[]
  for j in jobs:
   if proofpath(j).exists():continue
   if gpu in [0,1]:
    group=GROUPS[gpu];p=ROOT/'receipts'/f'{group}.status.json';st=load(p) if p.exists() else {}
    if not done and (j['label']!=group or st.get('stage')!='WAITING_FOR_EVALUATION' or j['step']!=st.get('committed_updates')):continue
   if gpu==3:pass
   eligible.append(j)
  if not eligible:heartbeat('eval_gpu'+str(gpu),'WAITING_FOR_JOB');time.sleep(3);continue
  j=eligible[0]
  try:
   with lock('evaluation_'+tag(j)+'.lock'):
    if proofpath(j).exists():continue
    if ev is None:ev=Evaluator()
    ev.point(j)
  except BlockingIOError:time.sleep(1)
  except torch.cuda.OutOfMemoryError:
   heartbeat('eval_gpu'+str(gpu),'WAITING_FOR_MEMORY',failed_job=j);ev=None;gc.collect();torch.cuda.empty_cache();time.sleep(15)
  except Exception as error:
   import traceback
   dump(ROOT/'evaluation/errors'/f'{tag(j)}.json',dict(status='ERROR',job=j,error=repr(error),traceback=traceback.format_exc()));raise
 heartbeat('eval_gpu'+str(gpu),'COMPLETE')

def dominates(earlier,later):
 keys=load(ROOT/'PROTOCOL.json')['stop_rule']['metrics'];le=[];strict=[]
 for k in keys:
  a,b=earlier[k],later[k];tol=1e-6*max(abs(a),abs(b),1e-12)
  if k=='PSNR':le.append(a>=b-tol);strict.append(a>b+tol)
  else:le.append(a<=b+tol);strict.append(a<b-tol)
 return all(le) and any(strict)
def milestones():
 for group in GROUPS:
  completed=[]
  for step in STEPS:
   js=[dict(label=group,step=step,id=r['id'],qp=5,beta=1.) for r in devrows()[:4]]
   if not all(proofpath(j).exists() for j in js):continue
   ss=[load(proofpath(j))['summary'] for j in js];metrics={k:statistics.mean(s[k] for s in ss) for k in load(ROOT/'PROTOCOL.json')['stop_rule']['metrics']};completed.append((step,metrics))
   dest=ROOT/'evaluation/milestones'/group/f'{step:04d}.json'
   if dest.exists():continue
   stop=False;evidence=None
   if len(completed)>=3:
    previous=completed[-2]
    for early in completed[:-2]:
     if dominates(early[1],previous[1]) and dominates(early[1],metrics):stop=True;evidence=dict(early_step=early[0],early=early[1],previous_step=previous[0],previous=previous[1],current_step=step,current=metrics);break
   dump(dest,dict(status='PASS',group=group,step=step,metrics=metrics,stop=stop,stop_evidence=evidence,RWE_status='RWE_NOT_VERIFIED',sources=[r['id'] for r in devrows()[:4]],beta=1,QP=5,frames=64));event('MECHANICAL_EVALUATION_DECISION',group=group,step=step,stop=stop)

def queue():
 with lock('evaluation_queue.lock'):
  ps=[]
  for gpu in [0,1,3]:
   h=(ROOT/'logs'/f'eval_gpu{gpu}.log').open('a');p=subprocess.Popen([PYTHON,'-B','evaluate.py','--worker',str(gpu)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=h,stderr=subprocess.STDOUT);ps.append((p,h,gpu));event('EVALUATOR_STARTED',child_pid=p.pid,physical_gpu=gpu)
  while True:
   for p,h,gpu in ps:
    if p.poll() is not None:assert p.returncode==0,('EVALUATOR_FAILED',gpu,p.pid,p.returncode)
   milestones();js,done=available_jobs();remaining=[j for j in js if not proofpath(j).exists()];heartbeat('evaluation_queue','RUNNING',planned_points=len(js),completed_points=len(js)-len(remaining),children=[dict(pid=p.pid,gpu=gpu) for p,h,gpu in ps])
   if done and not remaining:
    summaries=[load(proofpath(j))['summary'] for j in js];csvwrite(ROOT/'metrics/per_video_metrics.csv',summaries);dump(ROOT/'receipts/evaluation_queue.complete.json',dict(status='COMPLETE',points=len(js),per_frame_rows=64*len(js),jobs=js));break
   time.sleep(3)
  for p,h,gpu in ps:p.wait();h.close();assert p.returncode==0
  heartbeat('evaluation_queue','COMPLETE')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--queue',action='store_true');a.add_argument('--worker',type=int);p=a.parse_args()
 if p.queue:queue()
 else:worker(p.worker)
