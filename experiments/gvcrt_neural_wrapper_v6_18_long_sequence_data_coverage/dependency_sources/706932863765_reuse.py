"""Strict baseline reuse: same RGB, QP, source checkpoint and metric implementation."""
from v614_io import *
def main():
    from eval_adapter import validate
    command([PYTHON,'-B',str(Path(__file__).resolve())]);frozen();good=[];missing=[]
    assert load(BASE/'final_integrity.json')['status']=='PASS'
    for m,old in [('original','original'),('v62_initial','v62')]:
        for v in sources():
            for q in range(10):
                out=point('uvg',m,v['video_index'],q);src=BASE/'parts/uvg'/old/out.name
                try:
                    r=load(src);assert r['method']==old and r['evaluator_sha256']==sha(BASE/'evaluate.py')
                    r.update(method=m,reused=True,reused_from=str(src),reused_sha256=sha(src),original_method=old)
                    validate(r,v,q,m);dump(out,r);good.append(str(out))
                except (AssertionError,KeyError,FileNotFoundError) as e:missing.append(dict(source=str(src),target=str(out),reason=repr(e)))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',points=len(good),verified=good,recompute=missing,historical_dependencies_verified=True));print('REUSED',len(good),'MISSING',len(missing),flush=True)
if __name__=='__main__':main()
