"""Exact common V6.12 objective AST plus channelwise GT continuous-LFQ cosine."""
import torch
from v68_io import *
from model_runtime import norm,core_parameters
class Objective:
    def __init__(self,v4,im,pm,models,quality,teacher,weight):
        self.v4=v4;self.teacher=teacher;self.weight=weight;self.models=models
        self.groups={k:list(models[k].parameters()) for k in ('wrapper','bridge')};self.groups['compression_core']=list(core_parameters(pm,models).values())
        self.parameters=[p for ps in self.groups.values() for p in ps]
        self.hook=models['generator'].register_forward_pre_hook(lambda _m,args:setattr(self,'latent',args[0]))
        source=(V66/'objective.py').read_text();start=source.index('    def consume(');end=source.index("    context['consume']=consume",start)
        source=source[:start]+'    consume = external_consume\n'+source[end:]
        source=source.replace("'consume(components,wp)'","'consume(components,wp,rec,frame)'")
        sys.path.insert(0,str(V66));ns=dict(__file__=str(V66/'objective.py'),external_consume=self.consume)
        exec(compile(source,str(ROOT/'objective.py'),'exec'),ns)
        cfg=load(ROOT/'config.json');self.fn=ns['bind'](v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
    def grads(self,term):
        gs=torch.autograd.grad(term,self.parameters,retain_graph=True,allow_unused=True);out={};index=0
        for name,ps in self.groups.items():
            vs=gs[index:index+len(ps)];index+=len(ps);out[name]=dict(norm=norm(vs),finite=all(bool(torch.isfinite(g).all()) for g in vs if g is not None),active=sum(g is not None for g in vs),parameters=len(ps))
        return out
    def consume(self,c,wp,rec,frame):
        t=self.v4.rate_trace;target=self.targets[self.index];self.index+=1
        assert self.latent.shape==target.shape
        cosine=(1-torch.nn.functional.cosine_similarity(self.latent.float(),target.float(),dim=1,eps=1e-8)).mean()
        common=c['loss'];self.extra['L_A']+=float(common.detach())/3;self.extra['L_cos']+=float(cosine.detach())/3
        if self.backward:self.cosine_gradient_frames.append(self.grads(cosine/3))
        self.extra['weighted_L_cos']+=float(cosine.detach())*self.weight/3
        for key in ('R_y0','R_y1','R_z','total_rate'):self.extra[key]+=float(t[key].detach())/3
        if self.weight:c['loss']=common+self.weight*cosine
        if self.measure:
            scales=[]
            for i,p in enumerate(t['passes']):
                g=torch.autograd.grad(t[f'R_y{i}'],p['s'],retain_graph=True,allow_unused=True)[0]
                scales.append(dict(pass_index=i,finite=g is None or bool(torch.isfinite(g).all()),norm=norm([g]),active=int(p['active'].sum())))
            latent_grad=torch.autograd.grad(c['D']/3,self.latent,retain_graph=True)[0]
            self.observations.append(dict(cosine=self.grads(cosine/3),weighted_cosine=self.grads(self.weight*cosine/3),reconstruction_latent_gradient=dict(norm=norm([latent_grad]),finite=bool(torch.isfinite(latent_grad).all())),Gaussian_scale=scales,estimated_rate=self.grads((t['R_y0']+t['R_y1']+t['R_z'])/(3*256*256))))
        if self.capture:self.outputs.append(rec.detach().cpu().clone())
        if self.backward:(c['loss']/3).backward()
    def __call__(self,frames,q,measure=False,capture=False,backward=True):
        self.measure=measure;self.capture=capture;self.backward=backward;self.index=0;self.observations=[];self.outputs=[];self.cosine_gradient_frames=[]
        self.extra={k:0. for k in ('L_A','L_cos','weighted_L_cos','R_y0','R_y1','R_z','total_rate')}
        before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        self.targets=[self.teacher(x) for x in frames[1:]]
        assert torch.equal(before[0],torch.get_rng_state()) and torch.equal(before[1],torch.cuda.get_rng_state())
        result,actual=self.fn(frames,q);result.update(self.extra);assert self.index==3
        return result,actual
    def close(self):self.hook.remove();self.targets=[];self.outputs=[];self.v4.rate_trace.clear()
