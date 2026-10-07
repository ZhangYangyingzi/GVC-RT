"""Original V6.6 B AST; observation-only gradient and output instrumentation."""
from v68_io import *
from model_runtime import norm,core_parameters
import torch

class Objective:
    def __init__(self,v4,im,pm,models,quality,cfg):
        self.models=models;self.core=core_parameters(pm,models)
        source=(V66/'objective.py').read_text()
        start=source.index('    def consume(');end=source.index("    context['consume']=consume",start)
        source=source[:start]+"    consume = external_consume\n"+source[end:]
        source=source.replace("'consume(components,wp)'","'consume(components,wp,rec)'")
        sys.path.insert(0,str(V66));ns=dict(__file__=str(V66/'objective.py'),external_consume=self.consume)
        exec(compile(source,str(ROOT/'objective.py'),'exec'),ns)
        self.fn=ns['bind'](v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
    def consume(self,c,wp,rec):
        if self.capture:self.outputs.append(rec.detach().cpu().clone())
        if self.measure and any(p.requires_grad for p in self.core.values()):
            ps=list(self.core.values())
            for label,term in [('L_B',c['loss']/3),('rate',c['beta_rate']/3)]:
                gs=torch.autograd.grad(term,ps,retain_graph=True,allow_unused=True)
                self.gradient_measurements[label].append(norm(gs))
        (c['loss']/3).backward()
    def __call__(self,frames,q,measure=False,capture=False):
        self.measure=measure;self.capture=capture;self.outputs=[];self.gradient_measurements=dict(L_B=[],rate=[])
        return self.fn(frames,q)
    def close(self):self.outputs=[]
