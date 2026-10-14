"""Fixed FP64, ddof0 moment EMA and regularized FD / mean-CORAL."""
import torch
BETA=.95
def sym(x):return (x+x.T)*.5
def moments(z):
 z=z.double();assert z.ndim==2 and z.shape[0]>0 and torch.isfinite(z).all()
 mu=z.mean(0);M=z.T@z/z.shape[0];return mu,M,sym(M-mu[:,None]*mu[None,:])
def candidate(z,old):
 mu,M,_=moments(z);mu=BETA*old['mu'].detach()+.05*mu;M=BETA*old['M'].detach()+.05*M
 return mu,M,sym(M-mu[:,None]*mu[None,:])
def reference(z):
 mu,M,S=moments(z);d=z.shape[1];eps=1e-4*max(float(S.trace()/d),1e-6)
 bar=S+eps*torch.eye(d,dtype=torch.float64,device=z.device)
 e,v=torch.linalg.eigh(bar);assert torch.isfinite(e).all() and float(e.min())>0
 return dict(mu=mu.detach(),M=M.detach(),Sigma=S.detach(),epsilon=eps,bar=bar.detach(),sqrt=((v*e.sqrt()[None,:])@v.T).detach(),n=len(z))
def losses(z,old,ref):
 mu,M,S=candidate(z,old);d=z.shape[1];dm=(mu-ref['mu']).square().sum()
 bar=S+ref['epsilon']*torch.eye(d,dtype=torch.float64,device=z.device);prod=sym(ref['sqrt']@bar@ref['sqrt'])
 eig=torch.linalg.eigvalsh(prod)
 # Fixed GT regularization makes this positive definite. No negative clamp/fallback.
 assert torch.isfinite(eig).all() and float(eig.min())>0,('nonpositive regularized symmetric product',float(eig.min()))
 dc=bar.trace()+ref['bar'].trace()-2*eig.sqrt().sum()
 out=dict(FD=dm+dc,mean=dm/d,CORAL=(S-ref['Sigma']).square().sum()/(4*d*d),D_mu=dm,D_cov_regularized=dc)
 assert all(torch.isfinite(v).all() for v in out.values())
 return out,dict(mu=mu.detach(),M=M.detach(),counter=old.get('counter',0)+1)
def commit(old,next_state):
 assert next_state['counter']==old.get('counter',0)+1
 return {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in next_state.items()}
