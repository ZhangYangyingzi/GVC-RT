"""Read and validate existing V6.15 baseline points without changing them."""
from io16 import *
def main():
    from adapter import evaluator
    frozen();good=[];bad=[];records=[]
    oldio=module('dists16_oldio',V15/'mixed_io.py')
    old=module('dists16_oldadapter',V15/'adapter.py') if 'retention_io' in sys.modules else None
    if old is None:
        # Pin the old adapter's imported globals rather than using new-directory IO.
        src=(V15/'adapter.py').read_text().replace('from retention_io import *','')
        import types
        old=types.ModuleType('dists16_oldadapter');old.__dict__.update({k:v for k,v in oldio.__dict__.items() if not k.startswith('__')});exec(compile(src,str(V15/'adapter.py'),'exec'),old.__dict__)
    oldio.frozen()
    for d in DATASETS:
        for v in sources(d):
            assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
            for m in METHODS[:3]:
                for q in range(10):
                    src=oldio.point(d,m,v['video_index'],q)
                    try:
                        r=load(src);old.validate(r,v,q,m)
                        assert r['checkpoint_sha256']==('' if m=='original' else evalcfg()['checkpoints'][m]['sha256'])
                        assert r['loaded_receiver_module_hashes']==(evalcfg()['original_receiver_hashes'] if m=='original' else evalcfg()['deployment_receiver_hashes'][m])
                        if 'runtime_audit' in r:assert r['runtime_audit']['metric_module_hashes']==evalcfg()['metric_module_hashes']
                        else:
                            # Early native baselines have no per-point runtime hash field.
                            # Frozen source and weight dependencies verify their metrics.
                            audit=BASE/'metric_implementation_audit.json';assert load(audit)['status']=='PASS'
                            assert sha(audit)==load(ROOT/'audits/dependencies.json')[str(audit)]
                            r['metric_dependency_verification']=dict(status='PASS',basis='historical point validation plus frozen source and metric weight dependencies; no fabricated runtime hashes',dependencies_sha256=sha(ROOT/'audits/dependencies.json'),source_metric_audit=str(audit),source_metric_audit_sha256=sha(audit))
                        r.update(dataset=d,reused=True,reused_from=str(src),reused_sha256=sha(src),reuse_verified=True)
                        dst=point(d,m,v['video_index'],q);records.append((dst,r))
                        good.append(dict(target=str(dst),source=str(src),sha256=sha(src),checkpoint_sha256=r['checkpoint_sha256'],input_RGB_sha256=r['source_rgb_sha256'],protocol_source=str(V15/'protocol.json'),protocol_sha256=sha(V15/'protocol.json'),metric_module_hashes=evalcfg()['metric_module_hashes']))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(source=str(src),dataset=d,method=m,video=v['video_index'],QP=q,reason=repr(e)))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',verified=good,recompute=bad,points=len(good),metric_dependencies_verified=True))
    digest=sha(ROOT/'audits/baseline_reuse.json')
    for p,r in records:r['reuse_audit_sha256']=digest;dump(p,r)
    print('REUSE',len(good),'RECOMPUTE',len(bad),flush=True)
if __name__=='__main__':main()
