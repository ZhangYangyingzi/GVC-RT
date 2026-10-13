"""Hash-validated scheduling cache; does not alter diagnostic calculations."""
from io21 import *
import argparse,fcntl

SCRIPTS={'covariance_diagnostic.py':'covariance_diagnostic','new_lovo.py':'new_leave_one_video_out'}

def signature(script,dataset):
    source='mcljcv' if dataset.endswith('29') else dataset
    vs=[v for v in sources(source) if not(dataset.endswith('29') and v['name']=='videoSRC17_1920x1080_24')]
    records=[]
    for v in vs:
        for m in METHODS:
            for q in (range(10) if script=='covariance_diagnostic.py' else (0,5,9)):
                r=load(point(source,m,v['video_index'],q))
                assert sha(r['feature_path'])==r['feature_sha256']
                records.append(dict(video=v,method=m,QP=q,bpp=r['bpp'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256']))
    code={n:sha(ROOT/n) for n in (script,'diagnostic_cache.py','fid_statistics.py','comparison.py','io21.py')}
    return hashlib.sha256(json.dumps(dict(dataset=dataset,methods=METHODS,records=records,code=code),sort_keys=True).encode()).hexdigest()

def receipt_path(script,dataset):
    return ROOT/'audits/diagnostic_cache'/f'{SCRIPTS[script]}_{dataset}.json'

def valid(script,dataset):
    p=receipt_path(script,dataset)
    if not p.exists():return False
    try:
        r=load(p)
        return (r['status']=='PASS' and r['input_signature']==signature(script,dataset)
                and all(Path(p).is_file() and sha(p)==h for p,h in r['outputs'].items())
                and load(ROOT/'results'/SCRIPTS[script]/dataset/'integrity.json')['status']=='PASS')
    except (OSError,KeyError,AssertionError,ValueError):return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--script',choices=SCRIPTS,required=True)
    ap.add_argument('--dataset',choices=(*DATASETS,'mcljcv_exclude_overlap29'),required=True);a=ap.parse_args()
    lockpath=receipt_path(a.script,a.dataset).with_suffix('.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if valid(a.script,a.dataset):print('VALIDATED DIAGNOSTIC REUSE',a.script,a.dataset,flush=True);return
    before=signature(a.script,a.dataset)
    subprocess.run([PYTHON,'-B','-u',str(ROOT/a.script),'--dataset',a.dataset],check=True)
    assert signature(a.script,a.dataset)==before,'diagnostic inputs changed during calculation'
    out=ROOT/'results'/SCRIPTS[a.script]/a.dataset
    assert load(out/'integrity.json')['status']=='PASS'
    files={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
    dump(receipt_path(a.script,a.dataset),dict(status='PASS',input_signature=before,outputs=files,
         calculation_unchanged=True,parallel_with_remaining_GPU_evaluation=True))
    print('DIAGNOSTIC CACHE PASS',a.script,a.dataset,flush=True)

if __name__=='__main__':main()
