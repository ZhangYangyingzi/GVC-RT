"""Independent direct-sigma probe against FP64 analytic interval derivatives."""
import math
import torch
from v68_io import *

def run(device):
    from src.models.entropy_models import GaussianEncoder
    from rate_forward import gaussian_component
    encoder=GaussianEncoder();rows=[]
    cases=[('interior',[0.,.25,1.,2.],[.3,.6,1.2,2.],None),
           ('inactive',[0.,1.],[.2,1.],[False,False]),
           ('empty',[],[],[]),
           ('boundaries',[0.,0.,0.,1.,1.],[.05,.11,.12,16.,20.],None),
           ('floor',[100.],[.3],None)]
    for name,qs,ss,mask in cases:
        outputs={}
        for mode in ('legacy','scale_ste'):
            q=torch.tensor(qs,device=device,dtype=torch.float32).requires_grad_()
            sigma=torch.tensor(ss,device=device,dtype=torch.float32).requires_grad_()
            active=sigma.detach()>.12 if mask is None else torch.tensor(mask,device=device,dtype=torch.bool)
            bits,g=gaussian_component(q,sigma,active,encoder,mode)
            dq,ds=torch.autograd.grad(bits,(q,sigma),allow_unused=True)
            fixed_bits,_=gaussian_component(q.detach(),sigma,active,encoder,mode)
            fixed_ds=torch.autograd.grad(fixed_bits,sigma,allow_unused=True)[0] if fixed_bits.requires_grad else None
            assert (ds is None and fixed_ds is None) or (ds is not None and fixed_ds is not None and torch.equal(ds,fixed_ds))
            assert torch.isfinite(bits) and (ds is None or torch.isfinite(ds).all()) and torch.isfinite(dq).all()
            outputs[mode]=dict(bits=bits.detach(),dq=dq.detach(),ds=None if ds is None else ds.detach(),stable=g['stable'].detach(),q=g['quantized'].detach(),prob=g['probability'].detach())
        old,new=outputs['legacy'],outputs['scale_ste']
        assert torch.equal(old['bits'],new['bits']) and torch.equal(old['dq'],new['dq'])
        assert old['ds'] is None or bool((old['ds']==0).all())
        record=dict(case=name,forward_exact=True,symbol_gradient_exact=True,literal_detached_symbol_probe=True,
            legacy_sigma_gradient=None if old['ds'] is None else old['ds'].cpu().tolist(),
            scale_ste_sigma_gradient=None if new['ds'] is None else new['ds'].cpu().tolist(),
            stable=new['stable'].cpu().tolist(),symbol_gradient=new['dq'].cpu().tolist(),
            active=active.cpu().tolist(),estimated_bits=float(new['bits']))
        if name=='interior':
            assert bool((new['prob']>1e-5).all()) and new['ds'] is not None and bool(torch.isfinite(new['ds']).all()) and bool((new['ds']!=0).all())
            x=new['q'].double();s=new['stable'].double();u=(x+.5)/s;l=(x-.5)/s
            phi=lambda z:torch.exp(-z.square()/2)/math.sqrt(2*math.pi)
            cdf=lambda z:(1+torch.erf(z/math.sqrt(2)))/2
            prob=cdf(u)-cdf(l)
            analytic_scale=-(l*phi(l)-u*phi(u))/(s*prob*math.log(2))
            analytic_symbol=-(phi(u)-phi(l))/(s*prob*math.log(2))
            assert torch.allclose(new['ds'].double(),analytic_scale,atol=1e-6,rtol=1e-5)
            assert torch.allclose(new['dq'].double(),analytic_symbol,atol=1e-6,rtol=1e-5)
            record.update(analytic_scale_FP64=analytic_scale.cpu().tolist(),analytic_symbol_FP64=analytic_symbol.cpu().tolist(),
                max_scale_derivative_error=float((new['ds'].double()-analytic_scale).abs().max()),
                max_symbol_derivative_error=float((new['dq'].double()-analytic_symbol).abs().max()),analytic_atol=1e-6,analytic_rtol=1e-5)
        rows.append(record)
    result=dict(status='PASS',cases=rows,analytic_reference='Derivative of continuous Gaussian interval probability at stable table values; NOT finite differences of discrete indexing',symbols_fixed_when_measuring_sigma=True)
    dump(ROOT/'audits/rate_gradient_probe.json',result);return result
