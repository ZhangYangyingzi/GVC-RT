from io32 import *
def main():
 import torch
 from distribution32 import reference,losses,moments
 torch.set_num_threads(2);torch.manual_seed(20261032);z=torch.randn(31,8,dtype=torch.float64,requires_grad=True);r=reference(torch.randn(37,8,dtype=torch.float64));mu,M,_=moments(torch.randn(41,8,dtype=torch.float64));old=dict(mu=mu.detach(),M=M.detach(),counter=0)
 a,_=losses(z,old,r);ma=.95*mu+.05*z.mean(0);mm=.95*M+.05*z.T@z/31;sg=(mm-ma[:,None]*ma[None,:]);bg=sg+r['epsilon']*torch.eye(8,dtype=torch.float64);e,v=torch.linalg.eigh(r['bar']);root=(v*e.sqrt()[None,:])@v.T;prod=root@bg@root;prod=(prod+prod.T)/2;fd=(ma-r['mu']).square().sum()+bg.trace()+r['bar'].trace()-2*torch.linalg.eigvalsh(prod).sqrt().sum()
 g1=torch.autograd.grad(a['FD'],z,retain_graph=True)[0];g2=torch.autograd.grad(fd,z)[0];err=float((g1-g2).norm()/g2.norm());assert abs(float(a['FD'].detach()-fd.detach()))<1e-10 and err<1e-8
 w=torch.randn(5,8,dtype=torch.float64,requires_grad=True);x=torch.randn(31,5,dtype=torch.float64);out=x@w;ls,_=losses(out,old,r);direct=torch.autograd.grad(ls['FD'],w)[0];leaf=out.detach().requires_grad_();ll,_=losses(leaf,old,r);dz=torch.autograd.grad(ll['FD'],leaf)[0];vjp=torch.autograd.grad(x@w,w,grad_outputs=dz)[0];re=float((direct-vjp).norm()/direct.norm());assert re<=1e-6
 dump(ROOT/'audits/distribution_math.json',dict(status='PASS',dimension=8,FD_Cholesky_SVD_vs_symmetric_eigen=dict(loss_error=abs(float(a['FD'].detach()-fd.detach())),gradient_relative_L2=err,epsilon_identical=True),two_pass_toy=dict(relative_L2=re,tolerance=1e-6,status='PASS'),no_jitter=True,no_PSD_clamp=True,training_backend='float64 Cholesky cross nuclear norm, mathematically equivalent square-root trace',not_a_substitute_for_real_network_calibration=True))
if __name__=='__main__':main()
