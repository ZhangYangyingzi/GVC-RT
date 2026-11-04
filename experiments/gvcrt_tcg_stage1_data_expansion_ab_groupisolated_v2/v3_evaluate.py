"""Versioned fixed development/final batches; fresh real RANS and independent decoding."""
from v3_common import *
from gtflow import setup,load_module,build_point,numerical_checks,METRICS
import argparse,traceback
def videos(split):
 from evaluation_runner import eval_videos
 return eval_videos(split)
def run_worker(batch_path,rank):
 verify_sources();batch=load(batch_path);assert batch['protocol_sha256']==sha(ROOT/'protocol_v3.json')
 if batch['kind']=='final':assert batch['selection_sha256']==sha(ROOT/'selected_models_v3.json')
 base=load_module('v3_fixed_real_rans',REPO/'experiments/gvcrt_tcg_stage1_evaluation/evaluate.py');torch,engine,quality,flow,perceptual=setup(rank)
 base.ROOT=ROOT/'evaluation_v3'/batch['id'];base.LARGE=ROOT/'large_artifacts/evaluation_v3'/batch['id'];(base.ROOT/'logs').mkdir(parents=True,exist_ok=True);(base.LARGE/'bitstreams').mkdir(parents=True,exist_ok=True)
 dump(base.ROOT/f'sampler_rank{rank}.json',numerical_checks(torch));point_function=build_point(base)
 from adapter35 import frames_for
 from ab_data_reader import PlanFrames
 import frames35
 historical=videos('historical_reference');frames35._SOURCES['ulong']={v['source_row']['video']:dict(path=v['source_row']['source_path'],sha256=v['source_row']['source_sha256'],width=v['width'],height=v['height']) for v in historical if v['dataset']=='ulong'}
 allvideos=[(split,v) for split in batch['splits'] for v in videos(split)]
 originals=torch.load(ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)['models']
 completed=0
 for index,(split,v) in enumerate(allvideos):
  if index%4!=rank:continue
  frames=frames_for(v) if split=='historical_reference' else PlanFrames(v['source_row'],torch.device('cpu'),full=True);cache={}
  for model in batch['states']:
   path=Path(model['checkpoint_path']);assert sha(path)==model['checkpoint_sha256'];state=torch.load(path,map_location='cpu',weights_only=True)
   assert state.get('absolute_optimizer_step',state['committed_global_updates'])==model['absolute_step']
   payload={k:dict(t) for k,t in originals.items()};payload['generator'].update(state['tcg_model_state'])
   class Runtime(engine.Runtime):
    def __init__(self):self.device=torch.device('cuda:0');self.wrapper=None;self.config=dict(force_zero_thres=.12);self.payload=payload
   for q in (0,5,9):
    key=f"{model['id']}_{v['dataset']}_{v['name']}_qp{q}";destination=base.ROOT/f'points/{key}.json'
    if destination.exists():
     point=load(destination);result=point['result'];assert point['status']=='PASS' and point['fresh_encode'] and point['independent_decode'] and result['checkpoint_sha256']==model['checkpoint_sha256']
     assert sha(base.LARGE/'bitstreams'/result['bitstream_file'])==result['bitstream_sha256']
    else:
     preview=None
     if batch['kind']=='final' and q==5 and (v['name'] in ('Kimono1','BasketballDrive','HoneyBee') or (split=='new_holdout_test' and v['name']==videos('new_holdout_test')[0]['name'])):preview=ROOT/f'large_artifacts/viewing_previews_v3/{key}'
     cp=dict(filename=path.name,sha256=model['checkpoint_sha256']);result=point_function(engine,Runtime(),quality,flow,perceptual,frames,v,q,model['absolute_step'],cp,key,cache,sample_dir=preview)
     point=load(destination)
    result.update(split=split,model_state=model['id'],additional_step=model['additional_step'],source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],source_group=v['source_group'],fresh_encode=True,input_stream_reused=False,historical_parent_group_exposure=v['historical_parent_group_exposure'],primary_new_unexposed_test=split=='new_holdout_test',evaluation_kind=batch['kind'])
    point['result']=result;point.update(protocol_v3_sha256=batch['protocol_sha256'],batch_manifest_sha256=sha(batch_path),snapshot_model_id=model['id'],selection_sha256=batch['selection_sha256']);point.setdefault('evaluated_unix',time.time());dump(destination,point)
    assert all(math.isfinite(float(result[k])) for k in METRICS);completed+=1
    event('V3_EVALUATION_POINT_COMPLETE',batch=batch['id'],model=model['id'],sequence=v['name'],QP=q,physical_GPU=4+rank,completed=completed)
    disk();assert sum(p.stat().st_size for p in (ROOT/'large_artifacts').rglob('*') if p.is_file())<=20*2**30,'LARGE_ARTIFACT_BUDGET_20GiB_EXCEEDED'
  if hasattr(frames,'close'):frames.close()
  del cache,frames
 dump(base.ROOT/f'worker{rank}_status.json',dict(status='PASS',completed_points=completed,physical_GPU=4+rank))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--worker',type=int);args=ap.parse_args();path=Path(args.batch)
 if args.worker is not None:return run_worker(path,args.worker)
 with lock('evaluation_v3.lock'):
  batch=load(path);out=ROOT/'evaluation_v3'/batch['id'];(out/'logs').mkdir(parents=True,exist_ok=True)
  if (out/'run_status.json').exists() and load(out/'run_status.json')['status']=='PASS':return
  # Training segments and every RAFT helper have exited before this entry.
  for arm in ARMS:
   p=ROOT/arm/'v3_training.lock';p.parent.mkdir(exist_ok=True)
   with p.open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  begun=time.time();processes=[]
  with (ROOT/'evaluation_v3_workers.lock').open('a') as lease:
   fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for rank in range(4):
    handle=(ROOT/f'logs/v3_evaluation_{batch["id"]}_{rank}.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(4+rank),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'v3_evaluate.py'),'--batch',str(path),'--worker',str(rank)],cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),));processes.append((p,handle))
   codes=[p.wait() for p,h in processes]
   for p,h in processes:h.close()
  if any(codes):dump(out/'run_status.json',dict(status='FAILED',exit_codes=codes));raise RuntimeError('INCOMPLETE_EVALUATION_NOT_A_PATIENCE_EVENT')
  points=[load(p) for p in sorted((out/'points').glob('*.json'))];count=sum(len(videos(s)) for s in batch['splits'])*3*len(batch['states'])
  assert len(points)==count and all(p['status']=='PASS' for p in points)
  records=[p['result'] for p in points];keys={(r['split'],r['sequence'],int(r['QP']),r['model_state']) for r in records}
  expected={(s,v['name'],q,m['id']) for s in batch['splits'] for v in videos(s) for q in (0,5,9) for m in batch['states']};assert keys==expected
  audit=audit_batch(batch,out);dump(out/'integrity.json',audit)
  write_csv(out/'metrics.csv',records)
  dump(out/'run_status.json',dict(status='PASS',expected_points=count,completed_points=count,elapsed_seconds=time.time()-begun,allocated_GPU_seconds=4*(time.time()-begun),metrics_sha256=sha(out/'metrics.csv'),batch_manifest_sha256=sha(path),fresh_encode=True,independent_decode=True))
def audit_batch(batch,out):
 points=[load(p) for p in sorted((out/'points').glob('*.json'))];models={m['id']:m for m in batch['states']};flow_hashes=collections.defaultdict(set);observed=[]
 from gtflow import METRICS
 qpmap=load(ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/qp_semantics_audit.json')['actual_qps']
 for point in points:
  r=point['result'];v=next(v for v in videos(r['split']) if v['name']==r['sequence']);model=models[r['model_state']]
  assert point['status']=='PASS' and point['real_RANS'] and point['fresh_encode'] and not point['input_stream_reused'] and point['independent_decode'] and point['model_hashes_unchanged'] and all(x['cleared'] for x in point['TCG_reset_audits'])
  assert r['checkpoint_sha256']==model['checkpoint_sha256'] and sha(model['checkpoint_path'])==model['checkpoint_sha256']
  assert point['actual_qps']==qpmap[str(r['QP'])][:64] and point['source_rgb_sha256']==v['rgb_sha256']
  stream=ROOT/'large_artifacts/evaluation_v3'/batch['id']/'bitstreams'/r['bitstream_file'];assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==point['bytes_consumed']==r['real_bytes']
  key=Path(r['bitstream_file']).stem
  with (out/f'frames/{key}.csv').open() as f:frames=list(csv.DictReader(f))
  with (out/f'transitions/{key}.csv').open() as f:transitions=list(csv.DictReader(f))
  assert len(frames)==64 and len(transitions)==63 and [int(x['frame']) for x in frames]==list(range(64)) and [int(x['to_frame']) for x in transitions]==list(range(1,64))
  assert sum(int(x['real_bits']) for x in frames)==8*r['real_bytes']
  assert abs(r['bpp']-8*r['real_bytes']/(64*v['height']*v['width']))<1e-12 and abs(r['kbps']-8*r['real_bytes']*v['rate_accounting_fps']/64/1000)<1e-10
  mse=sum(float(x['pixel_SSE']) for x in frames)/(64*v['height']*v['width']*3);assert abs(r['PSNR']+10*math.log10(max(mse,1e-15)))<1e-10
  assert all(x['source_frame_RGB_sha256']==v['frame_rgb_sha256'][i] and x['checkpoint_sha256']==model['checkpoint_sha256'] for i,x in enumerate(frames))
  assert hashlib.sha256(''.join(x['reconstruction_tensor_sha256'] for x in frames).encode()).hexdigest()==point['reconstruction_sha256']
  for metric in ('MS_SSIM','LPIPS','DISTS'):assert abs(sum(float(x[metric]) for x in frames)/64-float(r[metric]))<1e-10
  for metric in ('FloLPIPS','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC'):assert abs(sum(float(x[metric]) for x in transitions)/63-float(r[metric]))<1e-10
  assert all(math.isfinite(float(r[m])) for m in METRICS) and r['WarpingError']==r['WarpingError_ReconFlow_PWC']
  for t in transitions:flow_hashes[(r['split'],r['sequence'],int(t['to_frame']))].add(t['GT_flow_sha256'])
  observed.append((r['split'],r['sequence'],int(r['QP']),r['model_state']))
 expected={(split,v['name'],q,m['id']) for split in batch['splits'] for v in videos(split) for q in (0,5,9) for m in batch['states']}
 assert len(observed)==len(expected) and set(observed)==expected and all(len(h)==1 for h in flow_hashes.values())
 return dict(status='PASS',points=len(points),complete_Cartesian_product=True,streams_and_actual_rates=True,independent_decoder_and_reset=True,frame_and_transition_aggregation=True,nonfinite_metrics=False,matching_GT_flow_hashes=True)
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();sys.exit(1)
