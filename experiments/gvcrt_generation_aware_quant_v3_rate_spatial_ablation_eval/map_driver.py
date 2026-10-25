"""Generate fixed receiver-side maps and crops after all evaluations pass."""
import csv
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
import task
import driver

P=Path(__file__).resolve().parent
PYTHON=Path('/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python')


def main():
    driver.validate_gate()
    work=queue.Queue();errors=[]
    with (P/'evaluation.csv').open() as f:records=list(csv.DictReader(f))
    for row in records:
        work.put((row['sequence'],int(row['external_qp']),row['mode'],int(row['alpha_index'])))
    def worker(gpu):
        while not errors:
            try:source,qp,mode,alpha=work.get_nowait()
            except queue.Empty:return
            try:
                key=f'{mode}_s300_{source}_q{qp}_a{alpha}'
                audit=P/'server_artifacts/receiver_audits'/f'{key}.json'
                cached=json.loads(audit.read_text()) if audit.exists() else {}
                record=json.loads((P/'server_artifacts/evaluation'/f'{key}.json').read_text())
                valid=(cached.get('status')=='PASS' and cached.get('maps_code_sha256')==task.sha(P/'maps.py')
                       and cached.get('stream_sha256')==record['stream_sha256']
                       and cached.get('task_code_sha256')==task.sha(P/'task.py')
                       and cached.get('ablation_code_sha256')==task.sha(P/'ablation.py')
                       and cached.get('checkpoint_sha256')==record['checkpoint_sha256']
                       and task.sha(Path(record['stream_path']))==record['stream_sha256'])
                if not valid or not all((P/'figures/quant_maps'/f'{key}_f{frame}.png').exists() and
                           (P/'previews'/f'{key}_f{frame}.png').exists() for frame in (15,47)):
                    while True:
                        try:
                            status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                '--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=10)
                            used={int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in status.splitlines())}
                            if used[gpu]<1024:break
                        except (OSError,subprocess.SubprocessError,KeyError,ValueError):pass
                        print('WAIT_GPU_MAP',gpu,key,flush=True)
                        time.sleep(20)
                    args=[str(PYTHON),str(P/'maps.py'),'--source',source,'--qp',str(qp),
                          '--mode',mode,'--alpha-index',str(alpha),'--gpu',str(gpu)]
                    log=P/'server_artifacts/logs'/f'{key}_maps.log';log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('w') as f:done=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=P.parents[1])
                    if done.returncode:raise RuntimeError(str(log))
                print('MAP_PASS',key,flush=True)
            except Exception as exc:errors.append(str(exc));print('MAP_FAIL',str(exc),flush=True)
            finally:work.task_done()
    threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in (4,5,6,7)]
    for t in threads:t.start()
    for t in threads:t.join()
    if errors:raise RuntimeError(errors)
    print('MAPS_AND_RECEIVER_AUDITS_COMPLETE',len(records))


if __name__=='__main__':main()
