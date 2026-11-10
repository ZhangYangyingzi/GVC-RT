"""Only GT frames48..63; independent fixed Teacher targets shared by both groups."""
from common_s2 import *
from cache_codec import cache_paths
import argparse

def paths(row,qp):
    codec,_,_=cache_paths(row,qp,True);key=digest(dict(codec_identity=codec,teacher=TEACHER_SHA,frames=list(range(48,64)),RGB='uint8 -> CUDA FP32 /255',precision='FP32',normalization='none_extra'))
    return key,ROOT/'cache/teacher'/f'{key}.pt',ROOT/'audits/teacher_cache'/f'{key}.json'

def ensure(row,qp,rank=0):
    import torch
    env();key,p,rp=paths(row,qp)
    with lock('locks/teacher_'+key+'.lock',blocking=True):
        if p.exists() and rp.exists():
            a=load(rp);assert sha(p)==a['SHA256'] and a['Teacher_SHA256']==TEACHER_SHA;return a
        _,data_path,proof=cache_paths(row,qp,True);assert sha(data_path)==load(proof)['cache_sha256'];data=torch.load(data_path,map_location='cpu',weights_only=True,mmap=True)
        from teacher import Teacher
        teacher=Teacher(torch.device('cuda',rank));before=tensor_hash(teacher.state_dict().items());rng=rng_state();targets={};stats=[]
        for i in range(48,64):
            gt=data['GT'][i].permute(2,0,1)[None].to(rank).float()/255;value=teacher(gt);targets[i]=value.cpu()
            stats.append(dict(frame=i,shape=list(value.shape),min=float(value.min()),max=float(value.max()),mean=float(value.mean()),RMS=float(value.square().mean().sqrt()),SHA256=tensor_sha(value)))
        assert before==tensor_hash(teacher.state_dict().items()) and not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
        assert torch.equal(rng['torch'],torch.get_rng_state()) and torch.equal(rng['cuda'],torch.cuda.get_rng_state())
        atomic_torch(p,dict(targets=targets,Teacher_SHA256=TEACHER_SHA,codec_identity=data['info']['cache_identity'],teacher_model_SHA256=before,frames=list(range(48,64)),stats=stats))
        a=dict(status='PASS',SHA256=sha(p),bytes=p.stat().st_size,Teacher_SHA256=TEACHER_SHA,codec_identity=data['info']['cache_identity'],frames=list(range(48,64)),stats=stats);dump(rp,a)
        if not (ROOT/'teacher_audit.json').exists():dump(ROOT/'teacher_audit.json',teacher.audit)
        return a
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--qp',type=int,required=True);p.add_argument('--rank',type=int,default=0);a=p.parse_args();ensure(load(a.row),a.qp,a.rank)
