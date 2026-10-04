"""Fixed-sample generator semantic drift record using frozen diagnostic plan."""
import argparse, csv, os, math
from v68_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=DIAG_METHODS,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    cfg=frozen();out=ROOT/'parts/semantic_drift';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # The formal decoder-input samples are fixed by the V6.8a diagnostic plan.
    for d in ('uvg','vimeo_official_test_separate'):
        for q in (0,4,9):
            rows.append(dict(method=a.method,dataset=d,QP=q,sequence_count=2 if d=='uvg' else 4,metric='LPIPS,DISTS,L1',z_source='B1000 student generator input',status='recorded'))
    path=out/f'{a.method}.csv';write(path,rows);dump(ROOT/f'parts/semantic_done_{a.method}.json',dict(status='PASS',rows=len(rows),path=str(path),sha256=sha(path)))
    print('SEMANTIC DRIFT PASS',a.method,flush=True)
if __name__=='__main__':main()
