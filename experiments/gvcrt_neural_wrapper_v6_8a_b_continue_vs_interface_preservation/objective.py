"""Reuse V6.6 B objective AST verbatim; add only a P-interface cosine term."""
import ast
from v68_io import *
from model_runtime import Capture,norm,cosine,disjoint_states
FIELDS=('raw_alignment_loss','weighted_alignment_loss','interface_cosine_similarity','teacher_interface_norm','student_interface_norm')
def reference(v4,im,pm,models,quality,cfg):
    sys.path.insert(0,str(V66));m=module('v68_B_objective_reference',V66/'objective.py')
    return m.bind(v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
class Objective:
    def __init__(self,v4,im,pm,models,quality,cfg,teacher=None,weight=0.0,calibration=False):
        import torch
        import torch.nn.functional as F
        self.pm=pm;self.models=models;self.teacher=teacher;self.weight=weight;self.calibration=calibration;self.capture=Capture(pm) if teacher else None
        self.params=[p for k in ('wrapper','bridge') for p in models[k].parameters()];self.nw=len(list(models['wrapper'].parameters()))
        def enhance(c):
            if teacher is None:return c
            zs=self.capture.latent().float().flatten(1);zt=self.targets[self.index].detach().float().flatten(1);self.index+=1
            assert zs.shape==zt.shape and not zt.requires_grad
            cs=F.cosine_similarity(zs,zt,dim=1,eps=1e-8).mean();align=1-cs
            self.main=c['loss'];self.align=align
            c.update(raw_alignment_loss=align,weighted_alignment_loss=weight*align,interface_cosine_similarity=cs,teacher_interface_norm=zt.norm(dim=1).mean(),student_interface_norm=zs.norm(dim=1).mean())
            # Preserve precisely the same graph when lambda=0, not an added zero branch.
            if weight!=0:c['loss']=c['loss']+weight*align
            return c
        def consume(c,wp):
            if not calibration:(c['loss']/3).backward();return
            for label,term in [('main',self.main),('align',self.align),('weighted',weight*self.align)]:
                gs=torch.autograd.grad(term/3,self.params,retain_graph=True,allow_unused=True)
                for dst,g in zip(self.acc[label],gs):
                    if g is not None:dst.add_(g.detach())
        # Change only instrumentation and backward consumer in the existing B binder.
        source=(V66/'objective.py').read_text();start=source.index('    def consume(');end=source.index("    context['consume']=consume",start)
        source=source[:start]+"    consume = external_consume\n    context['enhance'] = external_enhance\n"+source[end:]
        needle="components['loss'] = components['loss'] + structure_weight*struct"
        assert source.count(needle)==1;source=source.replace(needle,needle+"\\ncomponents = enhance(components)")
        needle2='sums.update(proxy_MS_SSIM=0.0,structure_loss_raw=0.0,weighted_structure_loss=0.0)'
        if teacher:source=source.replace(needle2,needle2[:-1]+','+','.join(k+'=0.0' for k in FIELDS)+')')
        sys.path.insert(0,str(V66));ns=dict(__file__=str(V66/'objective.py'),external_consume=consume,external_enhance=enhance)
        exec(compile(source,str(ROOT/'objective.py'),'exec'),ns)
        self.fn=ns['bind'](v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
    def __call__(self,frames,q):
        import torch
        self.index=0
        if self.teacher:self.targets,actual=self.teacher.targets(frames,q)
        if self.calibration:self.acc={k:[torch.zeros_like(p) for p in self.params] for k in ('main','align','weighted')}
        sums,qs=self.fn(frames,q)
        if self.teacher:
            assert qs==actual and self.index==3 and disjoint_states(self.pm,self.teacher.pm);self.teacher.assert_frozen();self.capture.clear()
        else:sums.update({k:None for k in FIELDS})
        if self.calibration:
            self.gradients=self.acc
            for group,sl in [('wrapper',slice(0,self.nw)),('bridge',slice(self.nw,None))]:
                for label in self.acc:sums['g_'+group+'_'+label]=norm(self.acc[label][sl])
                sums['cosine_'+group]=cosine(self.acc['main'][sl],self.acc['align'][sl])
            sums['cosine_generator']=None;sums['generator_alignment_gradient']='NOT_APPLICABLE: loss is upstream of generator parameters'
        return sums,qs
    def close(self):
        if self.capture:self.capture.close()
