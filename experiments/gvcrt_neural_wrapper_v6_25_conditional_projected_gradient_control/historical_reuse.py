from io21 import *
def main():
 from adapter import validate,cache_key
 oldio=module('historical625_io24',V24/'io21.py');ns=dict(oldio.__dict__)
 exec(compile((V24/'adapter.py').read_text().replace('from io21 import *',''),str(V24/'adapter.py'),'exec'),ns)
 good=[]
 for d in DATASETS:
  for v in sources(d):
   assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
   for b in ('A_continue','B_projected'):
    m=f'historical_{b}_1000';oldm=b+'_1000'
    for q in range(10):
     p=oldio.point(d,oldm,v['video_index'],q);r=load(p);ns['validate'](r,v,q,oldm,-1)
     r.update(method=m,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,cache_key=cache_key(v,q,m,-1),source_method=oldm,source_protocol_sha256=sha(V24/'protocol.json'))
     validate(r,v,q,m,-1);dump(point(d,m,v['video_index'],q),r);good.append(dict(method=m,dataset=d,video=v['video_index'],QP=q,source=str(p),sha256=sha(p)))
 dump(ROOT/'audits/historical_reuse.json',dict(status='PASS',reused=len(good),records=good))
if __name__=='__main__':main()
