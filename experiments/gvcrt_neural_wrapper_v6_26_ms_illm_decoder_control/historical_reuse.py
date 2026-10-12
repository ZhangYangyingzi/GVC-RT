from io21 import *
def main():
 from adapter import validate,cache_key
 oldio=module('hist626_io25',V25/'io21.py');ns=dict(oldio.__dict__);exec(compile((V25/'adapter.py').read_text().replace('from io21 import *',''),str(V25/'adapter.py'),'exec'),ns);good=[]
 for d in DATASETS:
  for v in sources(d):
   assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
   for oldm in ('A_continue_1000','B_capped_projected_1000','C_capped_conditional_projected_1000'):
    m='v625_'+oldm
    for q in range(10):
     p=oldio.point(d,oldm,v['video_index'],q);r=load(p);ns['validate'](r,v,q,oldm,-1);r.update(method=m,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,cache_key=cache_key(v,q,m,-1),source_method=oldm,source_protocol_sha256=sha(V25/'protocol.json'));validate(r,v,q,m,-1);dump(point(d,m,v['video_index'],q),r);good.append(dict(method=m,dataset=d,video=v['video_index'],QP=q,source=str(p),sha256=sha(p)))
 dump(ROOT/'audits/historical_reuse.json',dict(status='PASS',reused=len(good),records=good))
if __name__=='__main__':main()
