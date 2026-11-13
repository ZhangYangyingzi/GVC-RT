from util import *
import torch
import torch.nn.functional as F
TERMS=['LPIPS','DISTS_weighted','RGB_L1_weighted']
def quality_models():
    import lpips
    sys.path.append(str(ROOT/'vendor/runtime_dependencies'))
    from DISTS_pytorch import DISTS
    return (lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False),DISTS().cuda().eval().requires_grad_(False))
def losses(quality,out,gt):
    return [quality[0](out,gt,normalize=True).mean(),.5*quality[1](out,gt,require_grad=True).mean(),.1383589248954108*F.l1_loss(out,gt)]
def norm(gs):return sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5

def clip_forward(branch,data,clip,condition,intervention=None,checkpointing=True):
    seq=clip['video'];start=clip['start'];state=None
    # Candidate warmup, with its own state; no B300 out history is ever injected.
    with torch.no_grad():
        for i in range(start-2,start):
            f,p,h,x,q,m=data.tensors(seq,i)
            if m['scene_cut']:state=None
            hh,_=branch(f,p,h,m,condition,intervention)
            _,state=data.receiver.suffix(hh,x,q,state,False)
        state=state.detach()
    for i in range(start,start+clip['length']):
        f,p,h,x,q,m=data.tensors(seq,i)
        if m['scene_cut']:state=None
        hh,stats=branch(f,p,h,m,condition,intervention)
        out,state=data.receiver.suffix(hh,x,q,state,checkpointing)
        yield i,out,stats

def clip_loss(branch,data,quality,clip,condition,full=False):
    terms=[0.,0.,0.];stats=[]
    for i,out,st in clip_forward(branch,data,clip,condition):
        gt=data.gt(clip['video'],i)
        if full:out=out[...,:gt.shape[-2],:gt.shape[-1]]
        else:
            x,y=clip['x'],clip['y'];out=out[...,y:y+256,x:x+256];gt=gt[...,y:y+256,x:x+256]
        ts=losses(quality,out,gt)
        terms=[a+b/clip['length'] for a,b in zip(terms,ts)];stats.append({k:float(v) for k,v in st.items()})
    return terms,{k:sum(s[k] for s in stats)/len(stats) for k in stats[0]}

def step_gradient(branch,data,quality,clips,condition):
    total_frames=sum(c['length'] for c in clips);vals=[0.,0.,0.];stats=[]
    for clip in clips:
        ts,st=clip_loss(branch,data,quality,clip,condition)
        loss=sum(ts)*clip['length']/total_frames
        assert torch.isfinite(loss),'NONFINITE_LOSS'
        loss.backward()
        vals=[a+float(b.detach())*clip['length']/total_frames for a,b in zip(vals,ts)];stats.append(st)
    return dict(loss=sum(vals),**dict(zip(TERMS,vals)),**{k:sum(s[k] for s in stats)/len(stats) for k in stats[0]})

def diagnose(branch,data,quality,clips,condition,step,path):
    params=list(branch.parameters());names=[n for n,p in branch.named_parameters()]
    components=[[torch.zeros_like(p) for p in params] for _ in TERMS];vals=[0.,0.,0.];acts=[]
    for clip in clips:
        terms,stats=clip_loss(branch,data,quality,clip,condition);acts.append(stats)
        for k,t in enumerate(terms):
            vals[k]+=float(t.detach())/len(clips)
            gs=torch.autograd.grad(t/len(clips),params,retain_graph=k<2,allow_unused=True)
            for dst,g in zip(components[k],gs):
                if g is not None:dst.add_(g.detach())
    norms=[norm(gs) for gs in components];cos={}
    for i in range(3):
        for j in range(i+1,3):cos[TERMS[i]+'__'+TERMS[j]]=sum(float(a.double().mul(b.double()).sum()) for a,b in zip(components[i],components[j]))/(norms[i]*norms[j]) if norms[i]*norms[j] else None
    gs=[sum(x[i] for x in components) for i in range(len(params))];gn=norm(gs)
    result=dict(status='PASS',step=step,fixed_windows=clips,loss=sum(vals),terms=dict(zip(TERMS,vals)),gradient_norms=dict(zip(TERMS,norms)),gradient_cosines=cos,clip_before=gn,clip_after=min(gn,1.),upstream_gradient=norm([g for n,g in zip(names,gs) if not n.startswith('output.')]),output_gradient=norm([g for n,g in zip(names,gs) if n.startswith('output.')]),activation={k:sum(s[k] for s in acts)/len(acts) for k in acts[0]})
    dump(path,result);return result
