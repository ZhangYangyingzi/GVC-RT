"""Verify prior points before reuse; leave missing/mismatched points for real coding."""
from mixed_io import *
def main():
    from adapter import evaluator
    frozen();good=[];bad=[];records=[]
    # Historical adapters check complete byte counts, source frames, weights and assets.
    sys.path.insert(0,str(V614B));oldb=module('mixed_old_retention',V614B/'adapter.py')
    # Imports must bind the historical IO explicitly; local aliases otherwise shadow it.
    oldbio=module('mixed_old_retention_io',V614B/'retention_io.py')
    oldb.__dict__.update({k:v for k,v in oldbio.__dict__.items() if not k.startswith('__')})
    sys.path.insert(0,str(V614));olda=module('mixed_old_uvg',V614/'eval_adapter.py');oldaio=module('mixed_old_uvg_io',V614/'v614_io.py')
    olda.__dict__.update({k:v for k,v in oldaio.__dict__.items() if not k.startswith('__')})
    for d in (*DATASETS,'uvg_validation'):
        for v in sources(d):
            for m in METHODS[:4]:
                for q in range(10):
                    src=(V614/'parts'/m if d.startswith('uvg') else V614B/'parts'/d/m)/f'video_{v["video_index"]:02d}_qp{q}.json'
                    try:
                        r=load(src);oldv=dict(v,dataset='uvg') if d.startswith('uvg') else v
                        (olda if d.startswith('uvg') else oldb).validate(r,oldv,q,m)
                        assert r['evaluator_sha256']==sha(BASE/'evaluate.py')
                        assert r['loaded_receiver_module_hashes']==(evalcfg()['original_receiver_hashes'] if m=='original' else evalcfg()['deployment_receiver_hashes'][m])
                        if 'runtime_audit' in r:assert r['runtime_audit']['metric_module_hashes']==evalcfg()['metric_module_hashes']
                        # Historical immutable dependencies pin metric implementation and model files.
                        assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                        out=point(d,m,v['video_index'],q)
                        r.update(dataset=d,reused=True,reused_from=str(src),reused_sha256=sha(src),reuse_verified=True,original_dataset=oldv['dataset'])
                        records.append((out,r));good.append(dict(target=str(out),source=str(src),sha256=sha(src),checkpoint_sha256=r['checkpoint_sha256'],input_RGB_sha256=r['source_rgb_sha256'],protocol_source=str(V614/'protocol.json' if d.startswith('uvg') else V614B/'protocol.json'),metric_module_hashes=evalcfg()['metric_module_hashes']))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(source=str(src),dataset=d,method=m,video=v['video_index'],QP=q,reason=repr(e)))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',verified=good,recompute=bad,points=len(good),metric_dependencies_verified=True))
    digest=sha(ROOT/'audits/baseline_reuse.json')
    for p,r in records:r['reuse_audit_sha256']=digest;dump(p,r)
    print('REUSE',len(good),'RECOMPUTE',len(bad),flush=True)
if __name__=='__main__':main()
