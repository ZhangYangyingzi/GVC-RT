"""Validate historical points and transparently alias v6.17 final model."""
import types
from io18 import *
def main():
    frozen();oldio=module('long18_old_io',V17/'io17.py');old=types.ModuleType('long18_old_adapter');old.__dict__.update({k:v for k,v in oldio.__dict__.items() if not k.startswith('__')})
    exec(compile((V17/'adapter.py').read_text().replace('from io17 import *',''),str(V17/'adapter.py'),'exec'),old.__dict__);oldio.frozen();good=[];bad=[];records=[]
    for d in DATASETS:
        for v in sources(d):
            assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
            for m,oldm in [('original','original'),('v617_initial','dists_w05_1000')]:
                for q in range(10):
                    p=oldio.point(d,oldm,v['video_index'],q)
                    try:
                        r=load(p);old.validate(r,v,q,oldm);cfg=evalcfg();assert r['checkpoint_sha256']==('' if m=='original' else cfg['checkpoints'][m]['sha256']);assert r['loaded_receiver_module_hashes']==(cfg['original_receiver_hashes'] if m=='original' else cfg['deployment_receiver_hashes'][m])
                        if 'runtime_audit' in r:assert r['runtime_audit']['metric_module_hashes']==cfg['metric_module_hashes']
                        else:r['metric_dependency_verification']['dependencies_sha256']=sha(ROOT/'audits/dependencies.json')
                        r.update(method=m,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,source_method=oldm,method_alias_only=True);dest=point(d,m,v['video_index'],q);records.append((dest,r));good.append(dict(source=str(p),sha256=sha(p),target=str(dest),method=m,source_method=oldm,checkpoint_sha256=r['checkpoint_sha256'],input_RGB_sha256=r['source_rgb_sha256'],assets={k:dict(path=r[k+'_path'],sha256=r[k+'_sha256']) for k in ('bitstream','feature','frame_metrics','transitions')}))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(source=str(p),dataset=d,method=m,video_index=v['video_index'],QP=q,reason=repr(e)))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',points=len(good),expected=320,verified=good,rerun_required=bad,protocol_and_metric_dependencies_verified=True));digest=sha(ROOT/'audits/baseline_reuse.json')
    for p,r in records:r['reuse_audit_sha256']=digest;dump(p,r)
    print('BASELINE REUSE',len(good),'/320; RERUN',len(bad),flush=True)
if __name__=='__main__':main()
