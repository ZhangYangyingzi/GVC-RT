"""Validate six actual endpoint points; never treat them as complete datasets."""
from v62b_io import *
from evaluate import validate
def main():
    cfg=check_frozen();rows=[]
    for dataset in ('ulong','virat720','virat480'):
        v=next(v for v in sources(dataset) if v['video_index']==0)
        for q in (0,9):
            path=point(dataset,'v62',0,q);r=load(path);validate(r,v,q,'v62')
            rows.append(dict(dataset=dataset,method='v62',QP=q,point_path=str(path),point_sha256=sha(path),
                real_bytes=r['real_bytes'],frames=r['frames'],rate_accounting_fps=r['rate_accounting_fps'],metric_crop=r['metric_crop'],
                actual_qps=r['actual_qps'],independent_decode_pass=r['independent_decode_pass'],all_metrics_finite=True))
    dump(ROOT/'real_rans_smoke_audit.json',dict(status='PASS',scope='Six candidate endpoint points only; not full dataset evaluation',rows=rows))
    print('REAL RANS SMOKE PASS',len(rows),flush=True)
if __name__=='__main__':main()
