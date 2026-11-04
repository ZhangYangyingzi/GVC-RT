"""Complete fixed development trajectory and final Cartesian products after A/B."""
from ab_common import *
import csv,functools,traceback
from gtflow import load_module,setup,build_point,write,METRICS,numerical_checks
from reuse import parent_rows
from source_graph import identity
@functools.lru_cache(maxsize=3)
def eval_videos(split):
 if split=='historical_reference':
  mapping=load(ROOT/'graph/final_identity_to_group.json');exposure={mapping[identity(r['video'])] for r in parent_rows()};videos=[]
  for vs in load(P/'evaluation/manifest.local.json').values():
   for v in vs:videos.append(dict(v,source_group=mapping[identity(v['name'])],historical_parent_group_exposure=mapping[identity(v['name'])] in exposure,primary_new_unexposed_test=False))
  return videos
 rows=[r for r in load(ROOT/'manifests/evaluation_rows.json') if r['split']==split];videos=[]
 for i,r in enumerate(rows):
  fps=r['source_fps'];a,b=fps.split('/');fps=float(a)/float(b)
  videos.append(dict(dataset='ulong_new',video_index=i,name=Path(r['video']).stem,width=r['width'],height=r['height'],frames=64,rate_accounting_fps=fps,source_frame_indices=r['source_frame_indices'],source_sha256=r['source_sha256'],rgb_sha256=r['rgb_sha256'],frame_rgb_sha256=r['full_RGB_sha256'],source_row=r,source_group=r['source_group'],historical_parent_group_exposure=False,primary_new_unexposed_test=split=='new_holdout_test'))
 return videos
def states(kind):
 p=load(ROOT/'protocol.json');S=p['additional_global_updates']
 if kind=='development':
  return [('parent5000',P/'checkpoints/step_5000.pt',5000,0)]+[(f'{arm}_add{s}',ROOT/arm/f'checkpoints/step_{s}.pt',5000+s,s) for arm in ('A_clean_old_pool','B_expanded_ulong') for s in p['development_steps'] if s]
 return [('parent0',P/'checkpoints/step_0.pt',0,0),('parent5000',P/'checkpoints/step_5000.pt',5000,0),('A_final',ROOT/f'A_clean_old_pool/checkpoints/step_{S}.pt',5000+S,S),('B_final',ROOT/f'B_expanded_ulong/checkpoints/step_{S}.pt',5000+S,S)]
def worker(kind,rank):
 base=load_module('tcg_fresh_eval',REPO/'experiments/gvcrt_tcg_stage1_evaluation/evaluate.py')
 torch,engine,quality,flow,perceptual=setup(rank)
 sampler=numerical_checks(torch);sampler.update(strict_one_pixel_translation_claim=False,width11_dx1_sample_displacement=1.1);dump(ROOT/f'audits/PWC_sampler_rank{rank}.json',sampler)
 base.ROOT=ROOT/f'evaluation/{kind}';base.LARGE=ROOT/f'large_artifacts/{kind}';run_point=build_point(base)
 from adapter35 import frames_for
 from ab_data_reader import PlanFrames
 import frames35
 groups=[('development',eval_videos('development'))] if kind=='development' else [('historical_reference',eval_videos('historical_reference')),('new_holdout_test',eval_videos('new_holdout_test'))]
 historical=eval_videos('historical_reference')
 frames35._SOURCES['ulong']={v['source_row']['video']:dict(path=v['source_row']['source_path'],sha256=v['source_row']['source_sha256'],width=v['width'],height=v['height']) for v in historical if v['dataset']=='ulong'}
 allvideos=[(split,v) for split,vs in groups for v in vs];originals=torch.load(ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)['models']
 results=[]
 for i,(split,v) in enumerate(allvideos):
  if i%4!=rank:continue
  frames=PlanFrames(v['source_row'],torch.device('cpu'),full=True) if split!='historical_reference' else frames_for(v);cache={}
  for label,path,step,additional in states(kind):
   st=torch.load(path,map_location='cpu',weights_only=True);cp=dict(filename=path.name,sha256=sha(path));payload={k:dict(s) for k,s in originals.items()};payload['generator'].update(st['tcg_model_state'])
   manifest=load(P/'checkpoints_manifest.json')['checkpoints'] if path.is_relative_to(P) else load(path.parents[1]/'checkpoints_manifest.json')['checkpoints']
   assert cp['sha256']==next(x['sha256'] for x in manifest if x['filename']==path.name)
   assert st.get('absolute_optimizer_step',st['committed_global_updates'])==step
   class Runtime(engine.Runtime):
    def __init__(self):self.device=torch.device('cuda:0');self.wrapper=None;self.config=dict(force_zero_thres=.12);self.payload=payload
   for q in (0,5,9):
    key=f"{label}_{v['dataset']}_{v['name']}_qp{q}";dest=base.ROOT/f'points/{key}.json'
    if dest.exists():
     point=load(dest);assert point['status']=='PASS' and point['fresh_encode'] and point['result']['checkpoint_sha256']==cp['sha256']
     result=point['result']
     if 'model_state' not in result:
      result.update(split=split,model_state=label,additional_step=additional,source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],source_group=v['source_group'],historical_parent_group_exposure=v['historical_parent_group_exposure'],primary_new_unexposed_test=v['primary_new_unexposed_test']);point['result']=result;dump(dest,point)
     stream=base.LARGE/'bitstreams'/result['bitstream_file'];assert sha(stream)==result['bitstream_sha256'] and stream.stat().st_size==result['real_bytes']
     results.append(result);continue
    sample=None
    if kind=='final' and q==5 and (v['name'] in ('Kimono1','BasketballDrive','HoneyBee') or (split=='new_holdout_test' and v['name']==eval_videos('new_holdout_test')[0]['name'])):sample=ROOT/f'large_artifacts/viewing_previews/{key}'
    event('FRESH_EVALUATION_POINT_STARTED',kind=kind,point=key)
    result=run_point(engine,Runtime(),quality,flow,perceptual,frames,v,q,step,cp,key,cache,sample_dir=sample)
    result.update(split=split,model_state=label,additional_step=additional,source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],source_group=v['source_group'],historical_parent_group_exposure=v['historical_parent_group_exposure'],primary_new_unexposed_test=v['primary_new_unexposed_test'])
    point=load(dest);point['result']=result;point.update(split=split,model_state=label);dump(dest,point)
    results.append(result);write(base.ROOT/f'worker_{rank}_metrics.csv',results);event('FRESH_EVALUATION_POINT_PASS',kind=kind,point=key,completed=len(results))
    assert sum(p.stat().st_size for p in (ROOT/'large_artifacts').rglob('*') if p.is_file())<=20*2**30,'LARGE_ARTIFACT_BUDGET_20GiB_EXCEEDED'
    disk()
  if hasattr(frames,'close'):frames.close()
  del cache,frames
 dump(ROOT/f'evaluation/{kind}/worker_{rank}_status.json',dict(status='PASS',completed_points=len(results)))
def main():
 stage=sys.argv[sys.argv.index('--stage')+1]
 if stage=='parent_GTFlow':
  p=ROOT/'evaluation/parent_gtflow/run_status.json'
  if p.exists() and load(p)['status']=='PASS':return
  # A previous run may still own the exclusive GT-flow lock; wait for it.
  with (ROOT/'gtflow.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX)
  if not p.exists() or load(p)['status']!='PASS':assert subprocess.run([PYTHON,'-u',str(ROOT/'gtflow.py')],cwd=REPO).returncode==0
  return
 for arm in ('A_clean_old_pool','B_expanded_ulong'):assert load(ROOT/f'{arm}/training_summary.json')['status']=='PASS'
 protocol=load(ROOT/'protocol.json');assert sha(ROOT/'manifests/evaluation_rows.json')==protocol['evaluation_rows_sha256'] and sha(ROOT/'graph/final_identity_to_group.json')==protocol['source_group_map_sha256']
 manifest=load(ROOT/'source_manifest.json')
 for name,h in manifest['code'].items():assert sha(ROOT/name)==h,('EVALUATION_SOURCE_CHANGED_AFTER_FREEZE',name)
 for name,h in manifest['read_only_runtime'].items():assert sha(REPO/name)==h,('FROZEN_EVALUATION_RUNTIME_CHANGED',name)
 for name,h in manifest['original_helpers'].items():assert sha(ORIGINAL/name)==h,('FIXED_PARENT_EVALUATION_HELPER_CHANGED',name)
 dump(ROOT/'evaluation/historical_reference_exposure.json',dict(records=[{k:v[k] for k in ('dataset','name','source_group','historical_parent_group_exposure','primary_new_unexposed_test')} for v in eval_videos('historical_reference')],main_summary_split='new_holdout_test',historical_values_retained=True))
 dump(ROOT/'evaluation/protocol.json',dict(training_protocol_sha256=sha(ROOT/'protocol.json'),development_states=[dict(model_state=s,absolute_step=a,additional_step=b,checkpoint_filename=p.name,checkpoint_sha256=sha(p)) for s,p,a,b in states('development')],final_states=[dict(model_state=s,absolute_step=a,additional_step=b,checkpoint_filename=p.name,checkpoint_sha256=sha(p)) for s,p,a,b in states('final')],QP=[0,5,9],frames=64,metric_precision='FP32',deployment_precision='Parent real-RANS FP16 including bridge/generator; training precision is intentionally not imposed',fresh_encode=True,independent_decode=True,metrics=list(METRICS),primary_split='new_holdout_test',historical_reference_retained_separately=True,PWC_definition=load(ROOT/'protocol.json')['PWC_GTFlow'],PWC_sampler=load(ROOT/'protocol.json')['PWC_sampler'],sampler_interpretation='Compatibility with fixed existing implementation; width11 dx1 gives1.1pixel sampling displacement; no strict one-pixel claim',no_S2VC_exact_reproduction_claim=True,no_checkpoint_selection=True))
 for kind,count in [('development',32*3*len(states('development'))),('final',45*3*4)]:
  final=ROOT/f'evaluation/{kind}/run_status.json'
  if final.exists() and load(final)['status']=='PASS':continue
  # Workers inherit this lifetime lease, so a resumed coordinator waits for
  # orphaned same-task workers before it checks or resumes point receipts.
  with (ROOT/f'evaluation_{kind}_workers.lock').open('a') as lease:
   while True:
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:event('TRACK_EXISTING_EVALUATION_WORKERS',kind=kind);time.sleep(30)
   ps=[]
   for rank in range(4):
    f=(ROOT/f'logs/evaluation_{kind}_{rank}.log').open('a');ps.append((subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'evaluation_runner.py'),'--worker',kind,str(rank)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),)),f))
   codes=[p.wait() for p,f in ps]
   for p,f in ps:f.close()
  if any(codes):dump(final,dict(status='FAILED',exit_codes=codes));raise RuntimeError(f'{kind}_worker_failed')
  points=[load(p) for p in sorted((ROOT/f'evaluation/{kind}/points').glob('*.json'))];assert len(points)==count and all(p['status']=='PASS' for p in points)
  write(ROOT/f'evaluation/{kind}/metrics.csv',[p['result'] for p in points]);dump(final,dict(status='PASS',completed_points=count,expected_points=count,fresh_encode=True,independent_decode=True))
 assert subprocess.run([PYTHON,'-u',str(ROOT/'summarize.py')],cwd=REPO).returncode==0
if __name__=='__main__':
 try:
  if '--worker' in sys.argv:worker(sys.argv[-2],int(sys.argv[-1]))
  else:
   with lock('evaluation.lock'):main()
 except Exception:traceback.print_exc();sys.exit(1)
