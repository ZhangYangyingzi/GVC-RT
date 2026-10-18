"""Validated baseline aliases or old-stream feature extraction; new real encoding."""
from io32 import *
import argparse,traceback,resource,fcntl,math
def task_id(t):return f'{t["stage"]}_{t["dataset"]}_{t["video"]}_{t["method"]}_{t["qp"]}'
def receipt(t):return ROOT/'tasks'/(task_id(t)+'.json')
_EVALUATION_SIGNATURES={}
def eval_signature(t):
 key=json.dumps(t,sort_keys=True)
 if key in _EVALUATION_SIGNATURES:return _EVALUATION_SIGNATURES[key]
 code=code_signature(['io32.py','evaluate32.py','adapter32.py','frames32.py','training32.py'])
 x=dict(task=t,config=sha(ROOT/'config.json'),evaluation_config=sha(ROOT/'evaluation_config.json'),qp_mapping=sha(ROOT/'qp_semantics_audit.json'),protocol=sha(ROOT/'protocol.json'),manifest=sha(ROOT/'eval_manifest.json'),sources=sha(ROOT/'source_hashes.json'),code=code)
 if t['stage']!='baseline':
  p=ROOT/'branches'/t['branch']/'checkpoint_index.json'
  if not p.exists() or str(t['step']) not in load(p):return None
  x['checkpoint']=load(p)[str(t['step'])]['inference']
 value=hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest();_EVALUATION_SIGNATURES[key]=value;return value
def valid(t):
 try:
  r=load(receipt(t));return r['status']=='PASS' and r['signature']==eval_signature(t) and all(Path(p).is_file() and Path(p).stat().st_size==v['bytes'] and Path(p).stat().st_mtime_ns==v['mtime_ns'] for p,v in r['artifact_metadata'].items())
 except (OSError,KeyError,ValueError):return False
def oldpoint(t):return V29/'parts/ip_-1'/t['dataset']/t['method']/f'video_{t["video"]:02d}_qp{t["qp"]}.json'
def source(t):return next(v for v in load(ROOT/'eval_manifest.json')[t['dataset']] if v['video_index']==t['video'])
def artifacts(paths):
 return {str(p):dict(sha256=sha(p),bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in paths}
def audit_old(t):
 v=source(t);p=oldpoint(t);r=load(p);ec=load(ROOT/'evaluation_config.json');m=t['method']
 assert r['IP']==-1 and r['frames']==64 and r['num_transitions']==63 and r['independent_decode_pass'] and r['state_sync_pass'] and r['real_RANS']
 assert r['source_frame_indices']==v['source_frame_indices'] and r['source_rgb_sha256']==v['rgb_sha256']
 assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(t['qp'])][:64]
 assert r['compression_hash_before']==r['compression_hash_after']==ec['compression_hash']
 assert r['loaded_receiver_module_hashes']==(ec['original_receiver_hashes'] if m=='original' else ec['deployment_receiver_hashes'][m])
 if m!='original':assert r['checkpoint_sha256']==ec['checkpoints'][m]['sha256'] or r['runtime_audit']['receiver_hashes']==ec['deployment_receiver_hashes'][m]
 assert r['bytes_consumed']==r['real_bytes']==Path(r['bitstream_path']).stat().st_size
 assert abs(r['bpp']-r['real_bytes']*8/(64*1920*1080))<1e-15
 for k in ('bitstream','feature','frame_metrics','transitions'):
  assert sha(r[k+'_path'])==r[k+'_sha256'],('historical artifact hash mismatch',k,str(p))
 assert len(read(r['frame_metrics_path']))==64 and len(read(r['transitions_path']))==63
 return p,r
def prepare_baselines():
 """CPU-only cached aliases, only when prior official independent decode is verified."""
 index=[]
 for t in load(ROOT/'task_manifest.json')['tasks']:
  if t['stage']!='baseline':continue
  if valid(t):continue
  p,r=audit_old(t);c=V30/'tasks'/f'closure_{t["dataset"]}_{t["video"]}_{t["method"]}_{t["qp"]}.json'
  if not c.exists():continue
  cr=load(c);assert cr['status']=='PASS' and cr['reconstruction_hash_match']
  assert cr['input_record_sha256']==sha(p) and cr['bitstream_sha256']==r['bitstream_sha256'] and cr['actual_qps']==r['actual_qps'] and cr['bytes_consumed']==r['real_bytes']
  assert all(x['PASS'] for x in cr['records']);f=Path(next(k for k in cr['artifacts'] if k.endswith('.npz')))
  assert sha(f)==cr['artifacts'][str(f)]
  rr=dict(r,method=t['method'],dataset=t['dataset'],video=t['video'],QP=t['qp'],feature_path=str(f),feature_sha256=sha(f),official_feature_source='verified V630 independent decode',baseline_logical_alias=True,baseline_reextracted_this_run=False)
  arts=artifacts([p,c,Path(r['bitstream_path']),Path(r['frame_metrics_path']),Path(r['transitions_path']),f])
  dump(receipt(t),dict(status='PASS',signature=eval_signature(t),task=t,result=rr,artifact_metadata=arts,reused=True,reuse_from=str(p),reuse_sha256=sha(p),official_closure_source=str(c),official_closure_sha256=sha(c),fresh_encode=False,feature_reextraction=False))
  index.append(dict(task=t,source=str(p),source_sha256=sha(p),closure=str(c),closure_sha256=sha(c)))
 dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',verified_this_launch=index,baseline_expected=920))
def main(t,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from adapter32 import runtime,frames_for,decode,backend
 settings=backend();start=time.time();torch.cuda.reset_peak_memory_stats();lf=receipt(t).with_suffix('.lock').open('a');fcntl.flock(lf,fcntl.LOCK_EX)
 if valid(t):return
 v=source(t);frames=frames_for(v);eng,rt=runtime(t);official=rt.metric_models[-1];oh=tensor_hash(official.state_dict())
 color_ns=dict(globals());color_ns.update(ROOT=ROOT);exec(compile((V31/'color31.py').read_text().replace('from audit31 import *',''),str(V31/'color31.py'),'exec'),color_ns);color=color_ns['color']
 f=ROOT/'features/evaluation'/(task_id(t)+'.npz');frpath=ROOT/'raw'/f'{task_id(t)}.frames.csv';cp=ROOT/'raw'/f'{task_id(t)}.color.csv';colors=[]
 if False:  # Unreusable historical points are freshly encoded, not old-stream substitutes.
  p,r=audit_old(t);real=[];rec=[];b=Path(r['bitstream_path'])
  for i,x,info in decode(eng,rt,b,r):
   gt=frames[i].cuda()
   with torch.no_grad():real.append(official(gt).cpu().numpy());rec.append(official(x).cpu().numpy());colors.append(dict(frame=i,**color(x,gt)))
  save_npz(f,real=np.concatenate(real),reconstruction=np.concatenate(rec));frpath=Path(r['frame_metrics_path']);trans=Path(r['transitions_path'])
  r=dict(r,feature_path=str(f),feature_sha256=sha(f),baseline_reextracted_this_run=True,official_feature_source='v632 independent old-stream decode',source_record=str(p),source_record_sha256=sha(p))
  paths=[p,b,f,frpath,trans]
 else:
  b=ROOT/'bitstreams'/t['dataset']/t['method']/f'video_{t["video"]:02d}_qp{t["qp"]}.bin'
  r,fr=rt.run(frames,t['qp'],b,f);write(frpath,fr);trans=f.with_suffix('.transitions.csv')
  r.update(actual_qps=[x['actual_qp'] for x in fr]);assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(t['qp'])][:64]
  for i,x,info in decode(eng,rt,b,r):
   with torch.no_grad():colors.append(dict(frame=i,**color(x,frames[i].cuda())))
  paths=[b,f,frpath,trans]
 assert r['frames']==64 and r['num_transitions']==63 and r['state_sync_pass'] and r['independent_decode_pass'] and r['bytes_consumed']==b.stat().st_size
 assert oh==tensor_hash(official.state_dict());assert len(read(trans))==63
 write(cp,colors);paths.append(cp)
 r.update(method=t['method'],dataset=t['dataset'],video=t['video'],sequence=v['name'],QP=t['qp'],source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],native_fps=v['rate_accounting_fps'],rate_accounting_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,kbps_20fps=r['bits_per_frame']*20/1000,frame_metrics_path=str(frpath),frame_metrics_sha256=sha(frpath),transitions_path=str(trans),transitions_sha256=sha(trans),color_path=str(cp),official_hash=oh,model_train_or_eval='FP16 deployment codec/receiver, FP32 P and official Inception',adaptation_step=t.get('step',0))
 assert abs(r['bpp']-r['real_bytes']*8/(64*1920*1080))<1e-15
 dump(receipt(t),dict(status='PASS',signature=eval_signature(t),task=t,result=r,artifact_metadata=artifacts(paths),reused=False,feature_reextraction=False,fresh_encode=True,decode=rt.decode_audit,backend=settings,physical_GPU=gpu,seconds=time.time()-start,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
 print('EVALUATION PASS',task_id(t),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--gpu',type=int,required=True,choices=(4,5,6,7));a=ap.parse_args()
 try:main(json.loads(a.task),a.gpu)
 except Exception:error(dict(stage='evaluate',task=json.loads(a.task),error=traceback.format_exc()));raise
