"""Fresh ori initialization; frozen differentiable generator; FP32 core masters."""
import random,math
import torch
from v68_io import *
def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
def core_parameters(pm,models):
    excluded={id(p) for k in ('bridge','generator') for p in models[k].parameters()}
    return {n:p for n,p in pm.named_parameters() if id(p) not in excluded}
def setup(device,teacher_required=True):
    cfg=load(ROOT/'config.json');torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    v4=module('v613_training_core',V4/'train.py')
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12)
    for cp in cfg['init_checkpoints'].values():assert sha(cp['path'])==cp['sha256']
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator=pm.recon_generation_net.decoder.float().eval().requires_grad_(False)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator);core=core_parameters(pm,models)
    for p in core.values():p.data=p.data.float();p.requires_grad_(True)
    from rate_forward import bind
    bind(v4,'scale_ste');native=v4.joint_ste_forward
    class Call(torch.nn.Module):
        def __init__(self):super().__init__();self.pm=pm
        def forward(self,proxy,qp):return native(self.pm,proxy,qp)
    call=Call()
    def adapted(model,proxy,qp):
        assert model is pm
        return torch.func.functional_call(call,{'pm.'+n:p.to(torch.float16) for n,p in core.items()},(proxy,qp),strict=False)
    v4.joint_ste_forward=adapted
    oc=cfg['optimizer']
    groups=[dict(name=k,params=list(models[k].parameters()),lr=oc[k+'_lr']) for k in ('wrapper','bridge')]
    groups.append(dict(name='compression_core',params=list(core.values()),lr=oc['compression_core_lr']))
    opt=torch.optim.AdamW(groups,weight_decay=oc['weight_decay']);assert not opt.state
    ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))
    assert set(ids)=={id(p) for m in (wrapper,pm) for p in m.parameters() if p.requires_grad}
    assert not any(p.requires_grad for m in (im,generator) for p in m.parameters())
    quality=v4.quality_models(device)
    from teacher import Teacher
    teacher=Teacher(device,ema=True,candidate='pretrain262144') if teacher_required else None
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);random.seed(cfg['seed'])
    return v4,im,pm,models,opt,quality,teacher
def deployment_hash(im,pm):
    return tensor_hash(dict([(f'i.{n}',p.half()) for n,p in im.named_parameters()]+[(f'p.{n}',p.half()) for n,p in pm.named_parameters() if not n.startswith('recon_generation_net.')]))
def precision(im,pm,models,v4):
    core=core_parameters(pm,models)
    return dict(compression_hash=deployment_hash(im,pm),I_hash=v4.module_hash(im),master_core_hash=state_hash({n:p.detach() for n,p in core.items()}),deployment_core_hash=state_hash({n:p.detach().half() for n,p in core.items()}),master_dtype='float32',deployment_dtype='native ori float16')
def inference_state(im,pm,models,v4):
    return dict(**{k:{n:p.detach().cpu().clone() for n,p in m.state_dict().items()} for k,m in models.items()},full_p={n:p.detach().cpu().half().clone() for n,p in pm.state_dict().items()},precision=precision(im,pm,models,v4))
def inventory(im,pm,models,opt,quality,teacher):
    groups={id(p):(g['name'],g['lr']) for g in opt.param_groups for p in g['params']}
    items=[('wrapper.'+n,p) for n,p in models['wrapper'].named_parameters()]+[('P.'+n,p) for n,p in pm.named_parameters()]+[('I.'+n,p) for n,p in im.named_parameters()]
    items += [(f'quality{i}.'+n,p) for i,m in enumerate(quality) for n,p in m.named_parameters()]
    if teacher:items += [('teacher.'+n,p) for n,p in teacher.named_parameters()]
    return [dict(name=n,dtype=str(p.dtype),requires_grad=p.requires_grad,group=groups.get(id(p),('frozen',0))[0],lr=groups.get(id(p),('frozen',0))[1],numel=p.numel()) for n,p in items]
def atomic_torch(path,state):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(state,tmp);tmp.replace(path)
