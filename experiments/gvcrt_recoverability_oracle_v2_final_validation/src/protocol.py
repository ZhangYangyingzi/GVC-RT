"""Rate-only constraints and immutable encoding identities shared by all strategies."""
from base import *
import math
V1=REPO/'experiments/gvcrt_recoverability_oracle_v1_feasibility'

def cfg():return load(ROOT/'configs/experiment.json')
def point(dataset,q):return dataset+'_q'+str(q)
def signature(phase,dataset,q,plan):
 return hashlib.sha256(json.dumps([phase,dataset,q,plan],sort_keys=True).encode()).hexdigest()[:20]
def levels(plan):
 return tuple(next((p['factor'] for p in plan if all(p[k]==b[k] for k in ('frame','y','x','h','w'))),1.) for b in cfg()['blocks'])
def plan_for(ls):return [dict(b,factor=f) for b,f in zip(cfg()['blocks'],ls) if f!=1.]
def matched(bits,target):return 0<=target-bits<=cfg()['rate_tolerance_fraction']*target
def rate_key(bits,target,q,base_q):return (abs(bits-target),abs(q-base_q),q)
def read_receipt(ref):return load(ref['receipt_path'])
def usable(r):
 return r['status']=='PASS' and all(math.isfinite(r['result'][m]) for m in ('LPIPS','DISTS','FloLPIPS'))
def refs():return load(ROOT/'manifests/encoding_refs.json')
def register(phase,dataset,q,plan,kind='candidate',**extra):
 ident=signature(phase,dataset,q,plan)
 job=dict(id=ident,phase=phase,dataset=dataset,qp=q,plan=plan,kind=kind,**extra)
 p=ROOT/'manifests/jobs'/(ident+'.json')
 if p.exists():assert load(p)==job
 else:dump(p,job)
 return job
def lookup(phase,dataset,q,plan):return refs().get(signature(phase,dataset,q,plan))
def baseline(phase,dataset,q):
 ref=lookup(phase,dataset,q,[])
 return read_receipt(ref) if ref else None
def put_ref(job,path,source):
 mapping=refs();mapping[signature(job['phase'],job['dataset'],job['qp'],job['plan'])]=dict(receipt_path=str(path),source=source)
 dump('manifests/encoding_refs.json',mapping)
