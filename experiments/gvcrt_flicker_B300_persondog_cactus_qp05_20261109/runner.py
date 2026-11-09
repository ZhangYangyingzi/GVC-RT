"""Persistent inference-only controller; four deterministic video/QP shards."""
from common_s2 import *
from task import tasks,verify,completed_points
import threading,traceback
STOP=threading.Event();STAGE='preflight';CHILDREN={}

def status(state='RUNNING',**extra):
    dump(ROOT/'status.json',dict(status=state,complete=state=='COMPLETE',current_stage=STAGE,PID=os.getpid(),heartbeat=time.time(),
        completed_points=completed_points(),expected_points=8,training_updates=0,checkpoint_SHA256=B_FINAL_SHA,children=dict(CHILDREN),
        experiment_directory=str(ROOT),resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py',**extra))

def heartbeat():
    while not STOP.is_set():status();STOP.wait(10)

def phase(name):
    global STAGE
    STAGE=name;event('runner_stage',stage=name)

def workers(mode):
    pending={}
    for t in tasks():
        name=f'{mode}_{t["index"]}';log=ROOT/'logs'/f'{name}_{int(time.time())}.log';environment=dict(os.environ)
        if mode=='cache':environment['MODEL_KIND']='original'
        else:environment.pop('MODEL_KIND',None)
        with log.open('ab') as f:process=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'worker.py'),'--phase',mode,'--index',str(t['index'])],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        pending[name]=process;CHILDREN[name]=dict(pid=process.pid,physical_GPU=physical_gpu(t['rank']),log=str(log))
    codes={}
    while pending:
        for name,p in list(pending.items()):
            code=p.poll()
            if code is None:continue
            codes[name]=code;pending.pop(name);CHILDREN.pop(name)
        if pending:time.sleep(3)
    dump(ROOT/f'audits/{mode}_worker_exits.json',dict(status='PASS' if not any(codes.values()) else 'FAILED',exit_codes=codes,time=time.time()))
    assert not any(codes.values()),('WORKER_FAILED',codes)

def main():
    env();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4,5,6,7'
    with lock('runner.lock'):
        if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
            assert verify()
            from publish import main as publish
            if not (ROOT/'publication_status.json').exists() or load(ROOT/'publication_status.json').get('status')!='PUBLISHED':publish()
            status('COMPLETE',publication_status=load(ROOT/'publication_status.json')['status']);return
        assert verify();thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            phase('fresh_original_RANS_and_independent_decoder_cache');workers('cache')
            phase('matched_original_and_B300_metrics_with_native_video_exports');workers('evaluate')
            phase('GT_original_B300_fixed_crops_and_2x_detail_videos')
            from visualize import main as visualize
            visualize()
            phase('paired_metrics_and_final_integrity')
            from finalize import main as finalize
            finalize()
            STOP.set();thread.join(timeout=12);phase('GitHub_publication');status('COMPLETE',publication_status='PENDING')
            from publish import main as publish
            published=publish();phase('finished');status('COMPLETE',publication_status='PUBLISHED' if published else 'FAILED')
        except Exception as error:
            STOP.set();thread.join(timeout=12);status('FAILED',exception=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
