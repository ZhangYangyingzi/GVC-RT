"""Atomic, shared FP32 frozen-prefix training cache. No projected u is stored."""
from common_s2 import *
import argparse


def identity(row):
    return dict(schema='causal_raw_h46_63_v1',B300_SHA256=B300_SHA,INIT_SHA256=INIT_SHA,
        PILOT_SHA256=PILOT_SHA,BASELINE_SHA256=BASELINE_SHA,
        original_weights=load(ROOT/'protocol.json')['original']['weights'],
        precision='codec FP16; full reconstruction FP32; TCG/LoRA FP32; s=.5; shared module head',
        row=row,actual_qps=row['actual_frame_qps'],history='reset before I0, replay P1..63; raw ada_final h46..63; first available left replicate',
        prefix_code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'runtime_tree').rglob('*.py'))},
        loader_code=sha(ROOT/'experiment_model.py'),model_context_code=sha(ROOT/'model_context.py'),convlora_code=sha(ROOT/'convlora.py'))


def paths(row):
    key=digest(identity(row));return ROOT/'cache/h'/f'{key}.pt',ROOT/'audits/h_cache'/f'{key}.json'


def ready(row):
    p,r=paths(row)
    if not p.exists() or not r.exists():return False
    a=load(r);return a['status']=='PASS' and a['bytes']==p.stat().st_size


def ensure(row,rank=0,context=None):
    import torch
    from experiment_model import construct,reconstruct
    from model_context import assert_reset
    from cache_codec import cache_paths
    p,rp=paths(row)
    with lock('locks/h_'+p.stem+'.lock',blocking=True):
        if ready(row):return load(rp)
        _,src,proof=cache_paths(row,row['external_qp'],True)
        assert src.exists() and proof.exists()
        proof=load(proof);assert sha(src)==proof['cache_sha256']
        d=torch.load(src,map_location='cpu',weights_only=True,mmap=True)
        assert d['info']['source_row']==row and d['info']['actual_qps']==row['actual_frame_qps']
        c=context or construct('B300',True,rank,with_adapter=False)
        gen=c['models']['generator'];captured=[]
        handle=gen.ada_final.register_forward_hook(lambda m,args,h:captured.append(h.detach()))
        assert_reset(c);hs=[];start=time.time()
        with torch.no_grad():
            for i in range(1,64):
                captured.clear();raw=reconstruct(c,d['features'][i],d['info']['actual_qps'][i])
                assert len(captured)==1 and captured[0].dtype==torch.float32
                assert captured[0].shape[1:]==(320,32,32)
                if i>=46:hs.append(captured[0].cpu().clone())
        handle.remove();assert_reset(c)
        head=gen.head
        head_state={n:t.detach().cpu().clone() for n,t in head.state_dict().items()}
        headp=ROOT/'checkpoints/frozen_FP32_head.pt'
        with lock('locks/head.lock',blocking=True):
            if headp.exists():assert tensor_hash(head_state.items())==torch.load(headp,map_location='cpu',weights_only=True)['tensor_SHA256']
            else:atomic_torch(headp,dict(state=head_state,tensor_SHA256=tensor_hash(head_state.items()),provenance=c['provenance'],s=.5))
        # Store only training-required GT/flow and h. Source codec tensor stays read-only.
        out=dict(h=torch.cat(hs,0),GT=d['GT'][48:64].clone(),
            GT_backward=[x.clone() for x in d['GT_backward'][49:64]],
            GT_valid=[x.clone() for x in d['GT_valid'][49:64]],identity=identity(row),
            info=d['info'],source_cache_path=str(src),source_cache_SHA256=proof['cache_sha256'],
            h_first=46,GT_first=48,flow_first=49)
        atomic_torch(p,out)
        rec=dict(status='PASS',path=str(p),SHA256=sha(p),bytes=p.stat().st_size,identity=out['identity'],
            h_shape=list(out['h'].shape),h_dtype=str(out['h'].dtype),h_bytes=out['h'].numel()*4,
            head_SHA256=sha(headp),source_cache_path=str(src),source_cache_SHA256=proof['cache_sha256'],
            physical_GPU=physical_gpu(rank),seconds=time.time()-start)
        dump(rp,rec);event('h_cache_ready',slot=row['plan_slot'],physical_GPU=physical_gpu(rank),seconds=rec['seconds'])
        return rec

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('row');a=p.parse_args();env();ensure(load(a.row))
