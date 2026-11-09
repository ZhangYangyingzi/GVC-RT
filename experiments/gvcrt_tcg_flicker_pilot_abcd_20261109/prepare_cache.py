"""Four original-only cache workers. Produces only this task's frozen samples."""
from common_s2 import *
import argparse

def jobs(kind):
    if kind in ('training','engineering'):
        rows=plan_rows();return [(r,r['external_qp'],True) for r in rows[:80 if kind=='engineering' else 1200]]
    if kind=='development':return [(p['row'],q,False) for p in load(ROOT/'manifests/quick_development.json') for q in [0,5,9]]
    result=[]
    for p in load(ROOT/'manifests/long_videos.json'):
        r=dict(p['row']);r.update(long_video=True,source_frame_indices=list(range(128)),canonical_indices=list(range(128)));result.append((r,5,False))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['training','engineering','development','long'],required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args()
    env();os.environ['MODEL_KIND']='original'
    from cache_codec import ensure
    all_jobs=jobs(a.kind)
    with lock(f'locks/prepare_{a.kind}_{a.rank}.lock'):
      for i,(r,q,tr) in enumerate(all_jobs):
        if i%4!=a.rank:continue
        ensure(r,q,tr,a.rank)
        dump(ROOT/f'audits/cache_progress_{a.kind}_{a.rank}.json',dict(status='RUNNING',last_job=i,total_jobs=len(all_jobs),time=time.time()))
    dump(ROOT/f'audits/cache_progress_{a.kind}_{a.rank}.json',dict(status='PASS',total_jobs=len(all_jobs),time=time.time()))

if __name__=='__main__':main()
