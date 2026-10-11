from io21 import *
def main():
    from adapter import validate,cache_key,historical21_adapter
    old,oldio=historical21_adapter();oldio.frozen();good=[];bad=[]
    for d in DATASETS:
        for v in sources(d):
            assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
            for method,oldm in (('original','original'),('B1_initial','B1_reconstruction_1000')):
                for q in range(10):
                    p=oldio.point(d,oldm,v['video_index'],q,-1)
                    try:
                        r=load(p);old.validate(r,v,q,oldm,-1);assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                        r.update(kbps_20fps=r['bits_per_frame']*20/1000,original_rate_accounting_fps=v['rate_accounting_fps'],method=method,IP=-1,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,cache_key=cache_key(v,q,method,-1),source_method=oldm,source_protocol_sha256=sha(V22/'protocol.json'),reuse_dependency_sha256=sha(ROOT/'audits/dependencies.json'))
                        validate(r,v,q,method,-1);dump(point(d,method,v['video_index'],q,-1),r);good.append(dict(path=str(p),sha256=sha(p),method=method,dataset=d,sequence=v['name'],QP=q))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(path=str(p),reason=repr(e),method=method,dataset=d,sequence=v['name'],QP=q))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',expected=320,reused=len(good),records=good,rerun=bad));print('REUSE',len(good),'/320',flush=True)
if __name__=='__main__':main()
