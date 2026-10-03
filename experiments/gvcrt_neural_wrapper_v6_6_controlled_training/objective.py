"""Instrument the unchanged V6.4 objective AST with one proxy MS-SSIM term."""
import ast,math,types
from v66_io import *
def bind(v4,im,pm,wrapper,quality,cfg,weight=0.0,calibration=False,original=False):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(V64));base=module('v66_objective_reference',V64/'objective.py')
    if original:return base.bind(v4,im,pm,wrapper,quality,cfg)
    sys.path.insert(0,str(V62));from fullqp_common import beta_q,lambda_q,frame_qp
    # Preserve the established MS-SSIM calculation, returning its tensor for autograd.
    corefile=ROOT.parent/'gvcrt_neural_wrapper_v2_joint/core.py'
    ss=next(n for n in ast.parse(corefile.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='ms_ssim_rgb')
    assert ast.unparse(ss.body[-1])=='return float(result)';ss.body[-1]=ast.Return(value=ast.Name(id='result',ctx=ast.Load()))
    # Exactly F.pad(x,(0,1,0,1),mode='reflect'), with deterministic slice/cat backward.
    def reflect_pad(x):
        horizontal=torch.cat((x,x[:,:,:,-2:-1]),dim=3)
        return torch.cat((horizontal,horizontal[:,:,-2:-1,:]),dim=2)
    class DeterministicPad(ast.NodeTransformer):
        def visit_Call(self,n):
            if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='F' and n.func.attr=='pad':
                assert ast.literal_eval(n.args[1])==(0,1,0,1)
                return ast.Call(func=ast.Name(id='reflect_pad',ctx=ast.Load()),args=[n.args[0]],keywords=[])
            return self.generic_visit(n)
    ss=DeterministicPad().visit(ss)
    context=dict(v4=v4,im=im,pm=pm,wrapper=wrapper,quality=quality,cfg=cfg,torch=torch,F=F,math=math,args=types.SimpleNamespace(branch='schedule_s1p0'),beta_q=beta_q,lambda_q=lambda_q,frame_qp=frame_qp,load=load,ROOT=V62,structure_weight=weight)
    context['reflect_pad']=reflect_pad
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ss],type_ignores=[])),str(corefile),'exec'),context)
    tensor_ssim=context['ms_ssim_rgb'];ssim_verified=False
    def checked_ssim(x,y):
        nonlocal ssim_verified
        result=tensor_ssim(x,y)
        if not ssim_verified:
            with torch.no_grad():
                assert torch.equal(reflect_pad(x),F.pad(x,(0,1,0,1),mode='reflect'))
                assert math.isclose(float(result.detach()),v4.ms_ssim_rgb(x,y),rel_tol=1e-7,abs_tol=1e-8)
            ssim_verified=True
        return result
    context['ms_ssim_rgb']=checked_ssim
    def consume(components,wp):
        if not calibration:(components['loss']/3).backward();return
        terms=(components['D']+components['beta_rate'],components['lambda_proxy_proxy_L1'],components['structure_loss_raw'])
        for label,term in zip(('main','l1','struct_unit'),terms):
            grad=torch.autograd.grad(term/3,wp,retain_graph=True,allow_unused=True)
            for i,g in enumerate(grad):
                if g is not None:acc[label][i].add_(g.detach())
    context['consume']=consume
    node=base.original_ast()
    class Instrument(ast.NodeTransformer):
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=='sums' for t in n.targets):
                extra=ast.parse("sums.update(proxy_MS_SSIM=0.0,structure_loss_raw=0.0,weighted_structure_loss=0.0)").body[0]
                return [n,extra]
            if any(isinstance(t,ast.Name) and t.id=='components' for t in n.targets):
                extra=ast.parse("ssim = ms_ssim_rgb(proxy,frame)\nstruct = 1 - ssim\ncomponents.update(proxy_MS_SSIM=ssim,structure_loss_raw=struct,weighted_structure_loss=structure_weight*struct)\ncomponents['loss'] = components['loss'] + structure_weight*struct").body
                return [n,*extra]
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward':
                return ast.parse('consume(components,wp)').body[0]
            return self.generic_visit(n)
    node=Instrument().visit(node);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/'objective.py'),'exec'),context)
    fn=context['run_clip'];acc={}
    def run(frames,q):
        nonlocal acc
        if calibration:acc={k:[torch.zeros_like(p) for p in wrapper.parameters()] for k in ('main','l1','struct_unit')}
        sums,actual=fn(frames,q)
        if calibration:sums.update({f'g_{k}':math.sqrt(sum(float(g.double().square().sum()) for g in gs)) for k,gs in acc.items()})
        return sums,actual
    return run
