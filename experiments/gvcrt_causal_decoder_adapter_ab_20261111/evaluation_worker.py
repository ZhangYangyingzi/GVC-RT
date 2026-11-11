"""One source/QP bundle; Original decoding remains in an isolated namespace."""
from common_s2 import *
import traceback


def main(specpath):
    env();spec=load(specpath);v=spec['video'];q=spec['QP'];row=v.get('source_row',v)
    from cache_codec import cache_paths
    key,p,rp=cache_paths(row,q,False)
    with lock('locks/native_'+digest(dict(row=row,QP=q))+'.lock',blocking=True):
        if not p.exists() or not rp.exists():
            r=ROOT/'tmp/eval_rows'/f'{key}.json';dump(r,row)
            subprocess.run([PYTHON,'-B','-u','cache_codec.py','--row',str(r),'--qp',str(q)],cwd=ROOT,env=os.environ,check=True)
            key,p,rp=cache_paths(row,q,False)
        proof=load(rp);assert sha(p)==proof['cache_sha256']
        import torch
        from evaluation import Evaluator
        data=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        ev=Evaluator();torch.cuda.reset_peak_memory_stats()
        for task in spec['tasks']:
            ev.point(v,q,task['model'],task['step'],data)
            ev.contexts.clear();torch.cuda.empty_cache()
        dump(ROOT/'audits/evaluation_bundles'/Path(specpath).name,dict(status='PASS',spec=spec,
            physical_GPU=physical_gpu(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),time=time.time()))
        del data,ev;torch.cuda.empty_cache()
        # Native caches are disposable and bounded to active bundles, not all96 points.
        if p.is_relative_to(ROOT) and p.exists():p.unlink()


if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception as e:
        env();dump(ROOT/'audits/evaluation_failures'/f'{Path(sys.argv[1]).stem}_{os.getpid()}.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),spec=load(sys.argv[1]),physical_GPU=physical_gpu(0),retryable_OOM='out of memory' in str(e).lower(),time=time.time()));raise
