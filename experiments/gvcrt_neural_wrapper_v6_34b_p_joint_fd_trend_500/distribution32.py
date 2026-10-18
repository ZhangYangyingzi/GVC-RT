"""FP64 ddof0 moments and exactly regularized Wasserstein distance.
The cross-square-root trace equals nuclear_norm(L_real.T @ L_generated).
No jitter, PSD clipping, or constant replacement is used.
"""
import torch
BETA=.95
def sym(x):return (x+x.T)*.5
def moments(z):
 z=z.double();assert z.ndim==2 and len(z)>0 and torch.isfinite(z).all()
 mu=z.mean(0);M=z.T@z/len(z);return mu,M,sym(M-mu[:,None]*mu[None,:])
def candidate(z,old):
 mu,M,_=moments(z);mu=BETA*old['mu'].detach()+.05*mu;M=BETA*old['M'].detach()+.05*M
 return mu,M,sym(M-mu[:,None]*mu[None,:])
def reference(z):
 mu,M,S=moments(z);d=z.shape[1];eps=1e-4*max(float(S.trace()/d),1e-6);bar=S+eps*torch.eye(d,dtype=torch.float64,device=z.device)
 L=torch.linalg.cholesky(bar)
 return dict(mu=mu.detach(),M=M.detach(),Sigma=S.detach(),epsilon=eps,bar=bar.detach(),chol=L.detach(),n=len(z))
def losses(z,old,ref):
 mu,M,S=candidate(z,old);d=z.shape[1];dm=(mu-ref['mu']).square().sum();bar=S+ref['epsilon']*torch.eye(d,dtype=torch.float64,device=z.device)
 L=torch.linalg.cholesky(bar);cross=ref['chol'].T@L;sv=torch.linalg.svdvals(cross)
 assert torch.isfinite(sv).all() and sv.min()>0
 dc=bar.trace()+ref['bar'].trace()-2*sv.sum()
 out=dict(FD=dm+dc,mean=dm/d,CORAL=(S-ref['Sigma']).square().sum()/(4*d*d),D_mu=dm,D_cov_regularized=dc)
 assert all(torch.isfinite(v).all() for v in out.values())
 return out,dict(mu=mu.detach(),M=M.detach(),counter=old.get('counter',0)+1)
def commit(old,next_state):
 assert next_state['counter']==old.get('counter',0)+1
 return {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in next_state.items()}
def to(x,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
