"""One original-grid sampling pass per projected K/V; causal shared metadata."""
import torch
import torch.nn.functional as F
def grid(h,w,device):
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    return torch.stack((x,y),0)[None].float()
def sample(x,coords):
    h,w=x.shape[-2:];g=torch.stack((2*(coords[:,0]+.5)/w-1,2*(coords[:,1]+.5)/h-1),-1)
    return F.grid_sample(x.float(),g,mode='bilinear',padding_mode='zeros',align_corners=False)
def bounds(c,h,w):return (c[:,0:1]>=0)&(c[:,0:1]<=w-1)&(c[:,1:2]>=0)&(c[:,1:2]<=h-1)
def valid_flow(back,forward):
    h,w=back.shape[-2:];coords=grid(h,w,back.device)+back;fw=sample(forward,coords)
    return bounds(coords,h,w)&((back+fw).square().sum(1,keepdim=True)<=.01*(back.square().sum(1,keepdim=True)+fw.square().sum(1,keepdim=True))+.5)
OFFSETS=[(dx,dy) for dy in [-1,0,1] for dx in [-1,0,1]]
def geometry(back,valid,error,h,w,available=True):
    centers=grid(h,w,back.device)*8+3.5;rh,rw=back.shape[-2:]
    coords=grid(h,w,back.device)+sample(back,centers)/8
    external=(sample(valid.float(),centers)>.999)&bounds(centers,rh,rw)
    if not available:external=external&False
    return dict(coords=coords.detach(),external=external.float().detach(),fb_error=sample(error,centers).detach())
def candidates(x,coords):
    b,c,h,w=x.shape
    cc=torch.cat([coords+coords.new_tensor([dx,dy]).view(1,2,1,1) for dx,dy in OFFSETS],-1)
    return sample(x,cc).reshape(b,c,h,9,w).permute(0,3,1,2,4)
def shared_valid(meta):
    coords=meta['coords'];h,w=coords.shape[-2:];base=grid(h,w,coords.device)
    return torch.stack([bounds(coords+coords.new_tensor([dx,dy]).view(1,2,1,1),h,w)&bounds(base+base.new_tensor([dx,dy]).view(1,2,1,1),h,w) for dx,dy in OFFSETS],1)&meta['external'][:,None].bool()
