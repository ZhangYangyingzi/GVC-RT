"""CPU synthetic checks; not a replacement for required real-codec GPU smoke."""
import math,types
from v67_io import *
from objective import bind
def main():
    import torch
    torch.set_num_threads(2);torch.manual_seed(7);torch.use_deterministic_algorithms(True)
    real=module('v67_cpu_reference',V4/'train.py')
    class Scale(torch.nn.Module):
        def __init__(self,value):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(value))
        def forward(self,x):return self.scale*x
    class PM:
        def __init__(self):self.dec=torch.nn.Identity();self.bridge=Scale(.93);self.generator=Scale(.92)
        def clear_dpb(self):pass
        def set_curr_poc(self,n):pass
        def add_ref_frame(self,*args):pass
        def shift_qp(self,q,i):return q+[0,2,1][i]
    pm=PM();wrapper=Scale(.91);im=types.SimpleNamespace(compress=lambda x,q:{'x_hat':x})
    def forward(pm,proxy,q):return pm.generator(pm.bridge(pm.dec(proxy))),proxy.square().mean()*65536*.05,None
    def perceptual(rec,x,quality):
        lp=(rec-x).abs().mean();ds=(rec-x).square().mean();return lp+ds,lp,ds
    v4=types.SimpleNamespace(codec_input=lambda x:x,joint_ste_forward=forward,perceptual=perceptual,ms_ssim_rgb=real.ms_ssim_rgb)
    cfg={'lambda_proxy':.01};frames=[torch.rand(1,3,256,256) for _ in range(4)];params=list(wrapper.parameters())+list(pm.bridge.parameters())+list(pm.generator.parameters())
    def zero():
        for p in params:p.grad=None
    rows=[]
    for branch in BRANCHES:
        zero();ref,actual=bind(v4,im,pm,wrapper,None,cfg,original=True)(frames,0);gr=[p.grad.clone() for p in params]
        zero();sums,qa=bind(v4,im,pm,wrapper,None,cfg,branch=branch)(frames,0)
        assert qa==actual and all(math.isclose(sums[k],v,rel_tol=1e-6,abs_tol=1e-8) for k,v in ref.items())
        assert all(torch.allclose(p.grad,g,atol=1e-7,rtol=1e-6) for p,g in zip(params,gr))
        zero();sums,_=bind(v4,im,pm,wrapper,None,cfg,branch=branch,weight=.2)(frames,0)
        extra=sums['proxy_L1'] if branch==D else 1-sums['output_MS_SSIM']
        assert math.isclose(sums['raw_extra_guard'],extra,rel_tol=1e-6,abs_tol=1e-7)
        assert math.isclose(sums['loss'],ref['loss']+.2*extra,rel_tol=1e-6,abs_tol=1e-7)
        zero();bind(v4,im,pm,wrapper,None,cfg,branch=branch,guard_only=True)(frames,0)
        norms=[0.0 if p.grad is None else float(p.grad.abs().sum()) for p in params]
        assert norms[0]>0
        assert all(n>0 for n in norms) if branch==E else norms[1:]==[0.,0.]
        rows.append(dict(branch=branch,base_gradient_equivalence=True,extra_gradient_norms=norms))
    zero();sums,_=bind(v4,im,pm,wrapper,None,cfg,calibration=True)(frames,0)
    assert all(sums['g_'+k]>0 for k in ('main','B','D','E'))
    assert all(math.isfinite(z) and -1<=z<=1 for k,z in sums.items() if k.startswith('cos_'))
    assert all(p.grad is None for p in params)
    sys.path.insert(0,str(V66));old=module('v67_cpu_old_objective',V66/'objective.py')
    oldsum,_=old.bind(v4,im,pm,wrapper,None,cfg,calibration=True)(frames,0)
    assert math.isclose(sums['g_main'],oldsum['g_main'],rel_tol=1e-6)
    assert math.isclose(sums['g_B'],oldsum['g_struct_unit'],rel_tol=1e-6)
    assert math.isclose(.01*sums['g_D'],oldsum['g_l1'],rel_tol=1e-6)
    # Latest user policy allows sharing occupied GPUs, but only physical GPUs 4-7.
    scheduler=module('v67_cpu_scheduler',ROOT/'run_pipeline.py');original=scheduler.subprocess.check_output
    scheduler.subprocess.check_output=lambda args,**kw:'GPU4, 123\n' if '--query-compute-apps=gpu_uuid,pid' in args else '4, GPU4, 39000\n5, GPU5, 36000\n0, GPU0, 40000\n'
    try:assert scheduler.free_memory()=={4:39000,5:36000}
    finally:scheduler.subprocess.check_output=original
    dump(ROOT/'audits/cpu_protocol_tests.json',dict(status='PASS',synthetic_only=True,not_real_codec_smoke=True,rows=rows,calibration_gradients_match_V66=True,gradient_cosines_finite=True,shared_GPU_4567_filter_pass=True))
    print('CPU PROTOCOL TESTS PASS',flush=True)
if __name__=='__main__':main()
