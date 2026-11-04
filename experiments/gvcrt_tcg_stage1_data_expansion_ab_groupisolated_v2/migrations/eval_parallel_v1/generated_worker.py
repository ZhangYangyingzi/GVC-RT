def compute_worker(batch_path,rank):
 verify_sources();batch=load(batch_path);assert batch['protocol_sha256']==sha(ROOT/'protocol_v3.json')
 if batch['kind']=='final':assert batch['selection_sha256']==sha(ROOT/'selected_models_v3.json')
 base=load_module('v3_fixed_real_rans',REPO/'experiments/gvcrt_tcg_stage1_evaluation/evaluate.py');torch,engine,quality,flow,perceptual=worker_setup(rank)
 base.ROOT=output_root(batch);base.LARGE=stream_root(batch);(base.ROOT/'logs').mkdir(parents=True,exist_ok=True);(base.LARGE/'bitstreams').mkdir(parents=True,exist_ok=True)
 dump(base.ROOT/f'sampler_rank{rank}.json',numerical_checks(torch));point_function=build_point(base)
 from adapter35 import frames_for
 from ab_data_reader import PlanFrames
 import frames35
 historical=videos('historical_reference');frames35._SOURCES['ulong']={v['source_row']['video']:dict(path=v['source_row']['source_path'],sha256=v['source_row']['source_sha256'],width=v['width'],height=v['height']) for v in historical if v['dataset']=='ulong'}
 allvideos=[(split,v) for split in batch['splits'] for v in videos(split)]
 originals=torch.load(ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)['models']
 completed=0
 for index,(split,v) in enumerate(allvideos):
  if index%WORKER_COUNT!=rank:continue
  frames=bounded_frames(v,split,frames_for,PlanFrames,torch);cache={}
  for model in batch['states']:
   path=Path(model['checkpoint_path']);assert sha(path)==model['checkpoint_sha256'];state=torch.load(path,map_location='cpu',weights_only=True)
   assert state.get('absolute_optimizer_step',state['committed_global_updates'])==model['absolute_step']
   payload={k:dict(t) for k,t in originals.items()};payload['generator'].update(state['tcg_model_state'])
   class Runtime(engine.Runtime):
    def __init__(self):self.device=torch.device('cuda:0');self.wrapper=None;self.config=dict(force_zero_thres=.12);self.payload=payload
   for q in (0,5,9):
    key=f"{model['id']}_{v['dataset']}_{v['name']}_qp{q}";destination=base.ROOT/f'points/{key}.json'
    created_new=not destination.exists()
    if destination.exists():
     point=load(destination);result=point['result'];assert point['status']=='PASS' and point['fresh_encode'] and point['independent_decode'] and result['checkpoint_sha256']==model['checkpoint_sha256']
     assert sha(base.LARGE/'bitstreams'/result['bitstream_file'])==result['bitstream_sha256']
    else:
     preview=None
     if batch['kind']=='final' and q==5 and (v['name'] in ('Kimono1','BasketballDrive','HoneyBee') or (split=='new_holdout_test' and v['name']==videos('new_holdout_test')[0]['name'])):preview=ROOT/f'large_artifacts/viewing_previews_v3/{key}'
     cp=dict(filename=path.name,sha256=model['checkpoint_sha256']);result=point_function(engine,Runtime(),quality,flow,perceptual,frames,v,q,model['absolute_step'],cp,key,cache,sample_dir=preview)
     point=load(destination)
    result.update(split=split,model_state=model['id'],additional_step=model['additional_step'],source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],source_group=v['source_group'],fresh_encode=True,input_stream_reused=False,historical_parent_group_exposure=v['historical_parent_group_exposure'],primary_new_unexposed_test=split=='new_holdout_test',evaluation_kind=batch['kind'])
    if created_new:point['parallel_execution']=dict(addendum=ADDENDUM,addendum_sha256=ADDENDUM_SHA,worker=rank,workers_per_gpu=WORKERS_PER_GPU,physical_GPU=4+(rank//WORKERS_PER_GPU))
    point['result']=result;point.update(protocol_v3_sha256=batch['protocol_sha256'],batch_manifest_sha256=sha(batch_path),snapshot_model_id=model['id'],selection_sha256=batch['selection_sha256']);point.setdefault('evaluated_unix',time.time());dump(destination,point)
    assert all(math.isfinite(float(result[k])) for k in METRICS);completed+=1
    event('V3_EVALUATION_POINT_COMPLETE',batch=batch['id'],model=model['id'],sequence=v['name'],QP=q,physical_GPU=4+rank//WORKERS_PER_GPU,completed=completed)
    disk();assert sum(p.stat().st_size for p in (ROOT/'large_artifacts').rglob('*') if p.is_file())<=20*2**30,'LARGE_ARTIFACT_BUDGET_20GiB_EXCEEDED'
  if hasattr(frames,'close'):frames.close()
  del cache,frames
 dump(base.ROOT/f'worker{rank}_status.json',dict(status='PASS',completed_points=completed,physical_GPU=4+rank//WORKERS_PER_GPU))
