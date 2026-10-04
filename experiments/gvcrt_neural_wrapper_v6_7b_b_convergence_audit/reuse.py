"""Validate and reference 930 existing Original/A/B1000 points, without copying streams."""
from v67b_io import *
def main():
    frozen();sys.path.insert(0,str(V67A));old=module('v67b_anchor_io',V67A/'v67_io.py');ev=old.eval_adapter();count=0
    for d in DATASETS:
        for v in sources(d):
            for src,dst in [('original',O),('A_baseline',A),('B_proxy_msssim','B1000')]:
                for q in range(10):
                    path=old.point(d,src,v['video_index'],q);r=load(path);ev.validate(r,v,q,src)
                    r.update(method=dst,reused=True,reused_from=str(path),reused_point_sha256=sha(path));validate_point(r,v,q,dst);dump(point(d,dst,v['video_index'],q),r);count+=1
    assert count==930;dump(ROOT/'audits/reused_points.json',dict(status='PASS',points=count));print('REUSED',count,flush=True)
if __name__=='__main__':main()
