"""Historical real-RANS engine, native-I bypass and official pooled FID features."""
from runtime import *
import csv
import os
import time
import textwrap

def write(path, rows):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def task_name(d,i,m,q):return f'{d}_{i}_{m}_{q}'

def runtime_for(method,step=1000):
    import torch
    sys.path.insert(0,str(ROOT.parent/'gvcrt_parallel_v5a3_v60'))
    e=module('causal_engine',ROOT.parent/'gvcrt_parallel_v5a3_v60/engine.py')
    ec=load(V32/'evaluation_config.json');cfg=dict(experiment='B',force_zero_thres=.12,methods={},checkpoints={})
    if method==BRANCHES[0]:cfg['methods'][method]=[None,None]
    else:
        cp=ROOT/'branches'/method/'checkpoints'/f'adaptation_step_{step:04d}.pt'
        s=torch.load(cp,map_location='cpu',weights_only=True)
        cfg['checkpoints'][method]=dict(path=str(cp),sha256=sha(cp),module_hashes=s['hashes']['modules'])
        cfg['methods'][method]=[method if method in BRANCHES[2:] else None,method]
        del s
    # Exact validated native-I protocol: apply P only on P frames.
    source=textwrap.dedent(inspect.getsource(e.Runtime.run))
    old='proxy=target if self.wrapper is None else self.wrapper(target)'
    assert source.count(old)==1
    source=source.replace(old,'proxy=target if self.wrapper is None or i==0 else self.wrapper(target)')
    ns=dict(e.__dict__);exec(compile(source,str(ROOT/'evaluate.py'),'exec'),ns)
    run=ns['run']
    class Runtime(e.Runtime):
        def models(self,h,w):
            im,pm=super().models(h,w)
            assert e.core.compression_hash(im,pm)==ec['compression_hash']
            assert e.core.module_hash(im)==ec['I_hash']
            if not hasattr(self,'receiver_hashes'):self.receiver_hashes=self.loaded_receiver_hashes
            assert self.receiver_hashes==self.loaded_receiver_hashes
            return im,pm
    Runtime.run=run
    rt=Runtime(cfg,method,torch.device('cuda:0'))
    # Official feature module already validated in v6.32b (no alternate FID implementation).
    sys.path.insert(0,str(V32))
    from training32 import official_model
    rt.metric_models=(*rt.metric_models[:2],official_model())
    return e,rt

def baseline_reuse():
    ev=load(V32/'eval_manifest.json');expected=load(V32/'evaluation_config.json')
    count=0
    for d,vs in ev.items():
        for v in vs:
            i=v['video_index']
            for q in range(10):
                p=V32/'tasks'/f'baseline_{d}_{i}_original_{q}.json';s=load(p);r=s['result']
                assert s['status']=='PASS' and r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
                assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices']
                assert r['compression_hash_before']==r['compression_hash_after']==expected['compression_hash']
                assert r['loaded_receiver_module_hashes']==expected['original_receiver_hashes']
                for kind in ('bitstream','feature','frame_metrics','transitions'):
                    assert sha(r[kind+'_path'])==r[kind+'_sha256'],(p,kind)
                assert r['real_bytes']==Path(r['bitstream_path']).stat().st_size==r['bytes_consumed']
                assert abs(r['bpp']-r['real_bytes']*8/(64*1920*1080))<1e-15
                name=task_name(d,i,BRANCHES[0],q)
                result=dict(r,method=BRANCHES[0],dataset=d,video=i,sequence=v['name'],QP=q)
                dump('evaluation/points/'+name+'.json',dict(status='PASS',result=result,reused=True,source_receipt=str(p),source_receipt_sha256=sha(p)))
                count+=1
    dump('evaluation/baseline_reuse_integrity.json',dict(status='PASS',points=count,all_artifacts_rehashed=True,no_new_measurement_claim=True))
    print('BASELINE REUSE PASS',count,flush=True)

def smoke():
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='7')
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(V32));from adapter32 import frames_for
    e,rt=runtime_for(BRANCHES[3],0);v=load(V32/'eval_manifest.json')['ulong'][0]
    r,fr=rt.run(frames_for(v),0,ROOT/'bitstreams/smoke.bin',ROOT/'features/smoke.npz')
    r['actual_qps']=[f['actual_qp'] for f in fr]
    old=load(ROOT/'evaluation/points'/f'ulong_0_{BRANCHES[0]}_0.json')['result']
    checks={k:r[k]==old[k] for k in ('real_bytes','bitstream_sha256','reconstruction_sha256','actual_qps') if k in r and k in old}
    assert all(checks.values()),checks
    dump('sanity/full_pipeline.json',dict(status='PASS',checks=checks,result=r,scope='step0 identity-initialized P + original receiver versus verified original 64-frame native-resolution point'))
    print('FULL PIPELINE PASS',checks,flush=True)

def main(method,gpu,shard,shards):
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(V32))
    from adapter32 import frames_for
    e,rt=runtime_for(method)
    tasks=[(d,v,q) for d,vs in load(V32/'eval_manifest.json').items() for v in vs for q in range(10)]
    for idx,(d,v,q) in enumerate(tasks):
        if idx%shards!=shard:continue
        name=task_name(d,v['video_index'],method,q);receipt=ROOT/'evaluation/points'/f'{name}.json'
        if receipt.exists():continue
        b=ROOT/'bitstreams'/f'{name}.bin';f=ROOT/'features'/f'{name}.npz';t=time.time()
        r,fr=rt.run(frames_for(v),q,b,f)
        r['actual_qps']=[x['actual_qp'] for x in fr]
        assert r['actual_qps']==load(V32/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
        assert r['real_RANS'] and r['state_sync_pass'] and r['independent_decode_pass'] and r['bytes_consumed']==b.stat().st_size
        fp='evaluation/raw/'+name+'.frames.csv';write(fp,fr)
        r.update(method=method,dataset=d,video=v['video_index'],sequence=v['name'],QP=q,
                 source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],
                 frame_metrics_path=str(ROOT/fp),frame_metrics_sha256=sha(ROOT/fp),transitions_path=str(f.with_suffix('.transitions.csv')),transitions_sha256=sha(f.with_suffix('.transitions.csv')))
        dump('evaluation/points/'+name+'.json',dict(status='PASS',result=r,reused=False,seconds=time.time()-t,gpu=gpu))
        print('EVAL',name,time.time()-t,flush=True)

if __name__=='__main__':
    import argparse,traceback
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--method',choices=BRANCHES[1:]);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),default=7);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
    try:
        if a.smoke:smoke()
        elif a.baseline:baseline_reuse()
        else:main(a.method,a.gpu,a.shard,a.shards)
    except Exception:
        dump(f'logs/evaluation_failure_{os.getpid()}.json',dict(traceback=traceback.format_exc()));raise
