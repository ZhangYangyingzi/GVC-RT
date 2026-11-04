"""One or two independent evaluator processes per physical GPU, same numerical body."""
from eval_parallel_common import *
import argparse,traceback

def run_worker(batch_path,rank,workers_per_gpu):
    verify_parallel_sources();batch=load(batch_path);card=4+rank//workers_per_gpu
    assert workers_per_gpu in (1,2) and 0<=rank<4*workers_per_gpu and card in (4,5,6,7)
    worker_dir=EP/'worker_audits'/batch['id']/f'worker{rank}'
    (worker_dir/'audits').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'audits/GTFlow_sources.json',worker_dir/'audits/GTFlow_sources.json')
    import v3_evaluate
    begun=time.time();mapping={}
    def worker_setup(worker):
        isolated=v3_evaluate.load_module(f'parallel_gtflow_setup_{worker}',ROOT/'gtflow.py')
        isolated.ROOT=worker_dir
        result=isolated.setup(worker//workers_per_gpu)
        mapping.update(gpu_mapping(0,[card]));result[0].cuda.reset_peak_memory_stats(0)
        return result
    source=worker_source();assert hashlib.sha256(source.encode()).hexdigest()==load(EP/'source_manifest.json')['generated_worker_sha256']
    namespace=dict(v3_evaluate.__dict__,verify_sources=verify_parallel_sources,WORKER_COUNT=4*workers_per_gpu,WORKERS_PER_GPU=workers_per_gpu,worker_setup=worker_setup,output_root=output_root,stream_root=stream_root,bounded_frames=bounded_frames,videos=lambda split:selected_videos(batch,split),ADDENDUM=ADDENDUM,ADDENDUM_SHA=sha(EP/'protocol.json'))
    exec(compile(source,str(EP/'generated_worker.py'),'exec'),namespace)
    namespace['compute_worker'](batch_path,rank)
    import torch
    torch.cuda.synchronize()
    dump(worker_dir/'runtime.json',dict(status='PASS',worker=rank,workers_per_gpu=workers_per_gpu,physical_GPU=card,GPU_mapping=mapping,elapsed_seconds=time.time()-begun,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0],input_decode_concurrency_cap=4,original_numerical_body_preserved=True,addendum_sha256=sha(EP/'protocol.json')))

def run_batch(path,workers_per_gpu):
    verify_parallel_sources();batch=load(path);out=output_root(batch);(out/'logs').mkdir(parents=True,exist_ok=True)
    done=out/'run_status.json'
    if done.exists() and load(done)['status']=='PASS':return
    for arm in ARMS:
        for name in ('v3_training.lock','v3_rank0_active.lock'):
            with (ROOT/arm/name).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with lock('evaluation_v3.lock'):
        processes=[];begun=time.time()
        for rank in range(4*workers_per_gpu):
            card=4+rank//workers_per_gpu;log=ROOT/f'logs/parallel_evaluation_{batch["id"]}_{rank}.log';handle=log.open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            command=[PYTHON,'-B','-u',str(ROOT/'eval_parallel.py'),'--batch',str(path),'--worker',str(rank),'--workers-per-gpu',str(workers_per_gpu)]
            # Child workers retain the lifecycle lease if this parent exits.
            # The separate descriptor below is held by all eight children.
            processes.append((command,env,handle,card,rank))
        with (ROOT/'evaluation_v3_workers.lock').open('a') as lifetime:
            fcntl.flock(lifetime,fcntl.LOCK_EX|fcntl.LOCK_NB)
            running=[]
            for command,env,handle,card,rank in processes:
                process=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lifetime.fileno(),))
                running.append((process,handle,card,rank))
            dump(out/'active_workers.json',dict(workers=[dict(PID=p.pid,physical_GPU=card,worker=rank) for p,h,card,rank in running],workers_per_gpu=workers_per_gpu,started_unix=begun,addendum_sha256=sha(EP/'protocol.json')))
            codes=[p.wait() for p,h,card,rank in running]
            for p,h,card,rank in running:h.close()
        if any(codes):
            dump(done,dict(status='FAILED',exit_codes=codes,workers_per_gpu=workers_per_gpu));raise RuntimeError('PARALLEL_EVALUATION_FAILED; not a patience event')
        points=[load(p) for p in sorted((out/'points').glob('*.json'))];count=sum(len(selected_videos(batch,s)) for s in batch['splits'])*3*len(batch['states'])
        assert len(points)==count and all(p['status']=='PASS' for p in points)
        audit=audit_parallel_batch(batch,out);dump(out/'integrity.json',audit);write_csv(out/'metrics.csv',[p['result'] for p in points])
        elapsed=time.time()-begun
        dump(done,dict(status='PASS',expected_points=count,completed_points=count,elapsed_seconds=elapsed,allocated_GPU_seconds=4*elapsed,metrics_sha256=sha(out/'metrics.csv'),batch_manifest_sha256=sha(path),fresh_encode=True,independent_decode=True,workers_per_gpu=workers_per_gpu,evaluator_processes=4*workers_per_gpu,addendum_sha256=sha(EP/'protocol.json')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--worker',type=int);parser.add_argument('--workers-per-gpu',type=int,choices=(1,2));args=parser.parse_args()
    verify_parallel_sources()
    w=args.workers_per_gpu
    if w is None:
        decision=load(EP/'benchmark.json');assert decision['correctness_status']=='PASS';w=decision['selected_workers_per_gpu']
    if args.worker is not None:return run_worker(Path(args.batch),args.worker,w)
    return run_batch(Path(args.batch),w)

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
