"""Resume verified points; memory-aware fresh evaluations and numeric audit."""
import fcntl,traceback
from io18 import *
def main():
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(REPO);os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');frozen()
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':print('ALREADY PASS');return
    command([PYTHON,'-B','-u',str(Path(__file__).resolve())]);schedule=load(ROOT/'gpu_schedule.json') if (ROOT/'gpu_schedule.json').exists() else dict(status='RUNNING',allowed_GPUs=[4,5,6,7],startup_snapshot=gpu_snapshot(),workers=[],OOM_retries=[],other_processes_terminated=False)
    def persist():dump(ROOT/'gpu_schedule.json',schedule)
    def status(phase):dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),completed_points=len(list((ROOT/'evaluation/points').glob('*/*/video_*_qp*.json'))),expected_points=240,updated_unix=time.time()))
    def run(script,args=(),gpu=None):
        argv=[PYTHON,'-B','-u',str(ROOT/script),*args];command(argv);log=ROOT/'logs'/(Path(script).stem+'.log')
        with log.open('a') as f:rc=subprocess.run(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT).returncode
        return rc,log
    if not (ROOT/'audits/live_codec_modules.json').exists() or load(ROOT/'audits/live_codec_modules.json')['status']!='PASS':
        status('live_codec_audit');attempts=[]
        while len(attempts)<3:
            snapshot=gpu_snapshot();available=[g for g in snapshot if g['free_MiB']>=14000 and g['index'] not in attempts]
            if not available:
                if attempts:available=[g for g in snapshot if g['free_MiB']>=18000]
                if not available:persist();time.sleep(15);continue
            gpu=max(available,key=lambda x:x['free_MiB'])['index'];attempts.append(gpu);t=time.time();rc,log=run('audit_gpu_codec.py',('--gpu',str(gpu)))
            schedule['workers'].append(dict(kind='live_codec_audit',gpu=gpu,start_unix=t,end_unix=time.time(),returncode=rc,live_snapshot=snapshot,log=str(log)))
            if rc==0:break
            isoom='out of memory' in log.read_text().lower();schedule['OOM_retries'].append(dict(kind='live_codec_audit',gpu=gpu,OOM=isoom,returncode=rc));persist()
            assert isoom,'live codec audit failed: '+str(log)
        else:raise RuntimeError('Live codec audit could not finish after retries')
    persist();status('strict_reuse')
    if not (ROOT/'evaluation/reuse_manifest.json').exists() or load(ROOT/'evaluation/reuse_manifest.json')['status']!='PASS':
        rc,log=run('reuse.py');assert rc==0,log
    from adapter import validate
    pending=load(ROOT/'evaluation/reuse_manifest.json')['fresh_pending'];jobs={}
    for r in pending:
        key=(r['dataset'],r['method'],r['video']);jobs.setdefault(key,[]).append(r['external_QP'])
    active={};attempts={};failed_gpu={};cap=2
    def done(key,qs):return all(point(*key,q).exists() for q in qs)
    while any(not done(k,q) for k,q in jobs.items()) or active:
        for key,w in list(active.items()):
            rc=w['process'].poll()
            if rc is None:continue
            w['stream'].close();record=w['record'];record.update(end_unix=time.time(),returncode=rc);del active[key]
            if rc:
                isoom='out of memory' in Path(record['log']).read_text().lower();schedule['OOM_retries'].append(dict(job=list(key),gpu=record['gpu'],OOM=isoom,attempt=attempts[key],returncode=rc));failed_gpu[key]=record['gpu'];cap=1
                assert isoom and attempts[key]<4,record
            elif done(key,jobs[key]):
                d,m,i=key;v=next(v for v in sources(d) if v['video_index']==i)
                for q in jobs[key]:validate(load(point(d,m,i,q)),v,q,m)
            persist()
        snapshot=gpu_snapshot();free={g['index']:g['free_MiB'] for g in snapshot};count={g:sum(w['record']['gpu']==g for w in active.values()) for g in free}
        for key,qs in jobs.items():
            if key in active or done(key,qs):continue
            choices=[g for g in free if free[g]>=15000 and count[g]<cap]
            if failed_gpu.get(key) in choices and len(choices)>1:choices.remove(failed_gpu[key])
            if not choices:continue
            g=max(choices,key=lambda x:free[x]);d,m,i=key;argv=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--dataset',d,'--method',m,'--video',str(i),'--qps',','.join(map(str,qs)),'--gpu',str(g)];command(argv)
            log=ROOT/'logs'/f'eval_{d}_{m}_{i}.log';stream=log.open('a');process=subprocess.Popen(argv,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);attempts[key]=attempts.get(key,0)+1
            rec=dict(kind='fresh_real_RANS',job=list(key),external_QPs=qs,gpu=g,PID=process.pid,start_unix=time.time(),start_snapshot=snapshot,attempt=attempts[key],log=str(log));schedule['workers'].append(rec);active[key]=dict(process=process,stream=stream,record=rec);free[g]-=15000;count[g]+=1;persist()
        status('fresh_evaluation');time.sleep(10)
    schedule.update(status='PASS',fresh_jobs=len(jobs),reused_points=load(ROOT/'evaluation/reuse_manifest.json')['reused_points'],final_success=True,finished_unix=time.time());persist()
    pre=load(ROOT/'preflight_audit.json');pre.update(GPU_audit_pending=False,GPU_codec_audit='PASS');dump(ROOT/'preflight_audit.json',pre)
    for script,phase in [('report.py','numeric_tables'),('plot.py','plots'),('finalize.py','integrity')]:
        status(phase)
        if script=='plot.py':
            argv=[PLOT_PYTHON,'-B',str(ROOT/script)];command(argv)
            with (ROOT/'logs/plot.log').open('a') as f:rc=subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT).returncode
        else:rc,log=run(script)
        assert rc==0,script
    assert load(ROOT/'final_integrity.json')['status']=='PASS';dump(ROOT/'pipeline_status.json',dict(status='PASS',phase='complete',expected_points=240,completed_points=240,finished_unix=time.time()));print('EXPERIMENT COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();dump(ROOT/'pipeline_status.json',dict(status='FAIL',error=error,updated_unix=time.time()));raise
