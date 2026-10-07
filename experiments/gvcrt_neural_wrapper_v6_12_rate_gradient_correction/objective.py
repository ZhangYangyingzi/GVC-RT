"""Exact V6.6 B objective; observational Gaussian and grouped rate gradients."""
from v68_io import *
from model_runtime import norm,core_parameters
import torch

def tensor_stats(g):
    if g is None:return dict(is_none=True,finite=True,norm=0.,nonzero=0,elements=0,max_abs=0.)
    x=g.detach()
    return dict(is_none=False,finite=bool(torch.isfinite(x).all()),norm=norm([x]),nonzero=int(torch.count_nonzero(x)),elements=x.numel(),max_abs=float(x.abs().max()) if x.numel() else 0.)

class Objective:
    def __init__(self,v4,im,pm,models,quality,cfg):
        self.v4=v4;self.models=models;self.core=core_parameters(pm,models)
        self.groups={k:list(m.parameters()) for k,m in models.items()}
        self.groups['compression_core']=[p for p in self.core.values() if p.requires_grad]
        self.params=[p for ps in self.groups.values() for p in ps]
        source=(V66/'objective.py').read_text()
        start=source.index('    def consume(');end=source.index("    context['consume']=consume",start)
        source=source[:start]+"    consume = external_consume\n"+source[end:]
        source=source.replace("'consume(components,wp)'","'consume(components,wp,rec)'")
        sys.path.insert(0,str(V66));ns=dict(__file__=str(V66/'objective.py'),external_consume=self.consume)
        exec(compile(source,str(ROOT/'objective.py'),'exec'),ns)
        self.fn=ns['bind'](v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
    def grouped_grad(self,term):
        gs=torch.autograd.grad(term,self.params,retain_graph=True,allow_unused=True)
        out={};offset=0
        for name,ps in self.groups.items():
            group=gs[offset:offset+len(ps)];offset+=len(ps)
            out[name]=dict(norm=norm(group),parameters=len(ps),active=sum(x is not None for x in group),finite=all(bool(torch.isfinite(x).all()) for x in group if x is not None))
        return out
    def consume(self,c,wp,rec):
        t=self.v4.rate_trace
        for key in ('R_y0','R_y1','R_z','total_rate'):self.rate_sums[key]+=float(t[key].detach())/3
        if self.capture:
            self.outputs.append(rec.detach().cpu().clone())
            self.traces.append({k:t[k].detach().cpu().clone() for k in ('R_y0','R_y1','R_z','total_rate','z_hard','hard0','hard1','active0','active1','y_hat')})
        if self.measure:
            scales=[]
            for index,g in enumerate(t['passes']):
                derivative=torch.autograd.grad(t[f'R_y{index}'],g['s'],retain_graph=True,allow_unused=True)[0] if g['s'].requires_grad else None
                scales.append(dict(pass_index=index,active_count=int(g['active'].sum()),scale_gradient=tensor_stats(derivative),native_scale_dtype=str(g['s'].dtype)))
            div=rec.shape[-2]*rec.shape[-1]*3
            self.gradient_measurements['frames'].append(dict(Gaussian_scale=scales,
                R_y=self.grouped_grad((t['R_y0']+t['R_y1'])/div),R_z=self.grouped_grad(t['R_z']/div),units='bpp divided by three P frames'))
        (c['loss']/3).backward()
    def __call__(self,frames,q,measure=False,capture=False):
        self.measure=measure;self.capture=capture;self.outputs=[];self.traces=[]
        self.gradient_measurements=dict(frames=[]);self.rate_sums={k:0. for k in ('R_y0','R_y1','R_z','total_rate')}
        result,actual=self.fn(frames,q);result.update(self.rate_sums)
        return result,actual
    def close(self):self.outputs=[];self.traces=[];self.v4.rate_trace.clear()
