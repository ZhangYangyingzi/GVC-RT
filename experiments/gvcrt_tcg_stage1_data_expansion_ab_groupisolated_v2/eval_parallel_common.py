"""Versioned evaluator scheduling only; immutable v3 training and quality protocol."""
from v3_common import *
ADDENDUM='evaluation-parallelism-v1'
EP=ROOT/'migrations/eval_parallel_v1'

def verify_parallel_sources():
    verify_sources()
    manifest=load(EP/'source_manifest.json')
    assert sha(ROOT/'protocol_v3.json')==manifest['protocol_v3_sha256']
    assert sha(ROOT/'source_manifest_v3.json')==manifest['source_manifest_v3_sha256']
    assert sha(EP/'protocol.json')==manifest['addendum_sha256']
    for name,h in manifest['code'].items():
        assert sha(ROOT/name)==h,('PARALLEL_SOURCE_CHANGED',name)
    return manifest

def output_root(batch):
    return ROOT/('evaluation_parallel_probe' if batch['kind']=='engineering' else 'evaluation_v3')/batch['id']

def stream_root(batch):
    return ROOT/'large_artifacts'/('evaluation_parallel_probe' if batch['kind']=='engineering' else 'evaluation_v3')/batch['id']

def input_lease():
    """At most four simultaneous native input decodes across all evaluator workers."""
    @contextlib.contextmanager
    def lease():
        while True:
            for i in range(4):
                path=EP/f'input_{i}.lock';handle=path.open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                try:yield
                finally:handle.close()
                return
            time.sleep(.02)
    return lease()

def bounded_frames(video,split,frames_for,PlanFrames,torch):
    with input_lease():
        inner=frames_for(video) if split=='historical_reference' else PlanFrames(video['source_row'],torch.device('cpu'),full=True)
    class Frames:
        def __len__(self):return len(inner)
        def __getitem__(self,index):
            if split!='historical_reference':return inner[index]
            with input_lease():return inner[index]
        def close(self):
            if hasattr(inner,'close'):inner.close()
    return Frames()

def selected_videos(batch,split):
    from v3_evaluate import videos
    result=videos(split)
    if batch['kind']=='engineering':result=[v for v in result if v['name'] in batch['engineering_sequences']]
    return result

def replace_exact(source,old,new,count=1):
    assert source.count(old)==count,('FROZEN_EVALUATOR_TEMPLATE_MISMATCH',old,source.count(old),count)
    return source.replace(old,new)

def worker_source():
    import inspect,v3_evaluate
    s=inspect.getsource(v3_evaluate.run_worker)
    s=replace_exact(s,'def run_worker(batch_path,rank):','def compute_worker(batch_path,rank):')
    s=replace_exact(s,'=setup(rank)','=worker_setup(rank)')
    s=replace_exact(s,"base.ROOT=ROOT/'evaluation_v3'/batch['id'];base.LARGE=ROOT/'large_artifacts/evaluation_v3'/batch['id']","base.ROOT=output_root(batch);base.LARGE=stream_root(batch)")
    s=replace_exact(s,"if index%4!=rank:continue","if index%WORKER_COUNT!=rank:continue")
    s=replace_exact(s,"frames=frames_for(v) if split=='historical_reference' else PlanFrames(v['source_row'],torch.device('cpu'),full=True);cache={}","frames=bounded_frames(v,split,frames_for,PlanFrames,torch);cache={}")
    s=replace_exact(s,"if destination.exists():","created_new=not destination.exists()\n    if destination.exists():")
    s=replace_exact(s,'physical_GPU=4+rank','physical_GPU=4+rank//WORKERS_PER_GPU',2)
    s=replace_exact(s,"point['result']=result;","if created_new:point['parallel_execution']=dict(addendum=ADDENDUM,addendum_sha256=ADDENDUM_SHA,worker=rank,workers_per_gpu=WORKERS_PER_GPU,physical_GPU=4+(rank//WORKERS_PER_GPU))\n    point['result']=result;")
    return s

def audit_parallel_batch(batch,directory):
    import inspect,v3_evaluate
    s=inspect.getsource(v3_evaluate.audit_batch)
    s=replace_exact(s,"ROOT/'large_artifacts/evaluation_v3'/batch['id']","stream_root(batch)")
    namespace=dict(v3_evaluate.__dict__,stream_root=stream_root,videos=lambda split:selected_videos(batch,split))
    exec(compile(s,str(EP/'generated_audit.py'),'exec'),namespace)
    result=namespace['audit_batch'](batch,directory)
    return dict(result,original_audit_source_sha256=sha(ROOT/'v3_evaluate.py'),addendum_sha256=sha(EP/'protocol.json'))
