"""Pixel-center backward sampling v1; distinct from historical PWC sampler."""
import torch
import torch.nn.functional as F

VERSION='MCResidual_RAFT_pixelcenter_v1'

def alignment_field(data,i,device):
    """Pad native RGB flow/mask to codec geometry before feature rescaling."""
    h,w=data['info']['native_shape'];ph,pw=data['info']['padded_shape']
    flow=data['base_backward'][i].to(device).detach()
    valid=data['base_valid'][i].to(device).detach()
    return (F.pad(flow,(0,pw-w,0,ph-h)),F.pad(valid,(0,pw-w,0,ph-h),value=False))

def backwarp(value,flow):
    assert flow.shape[1]==2 and value.shape[-2:]==flow.shape[-2:]
    b,_,h,w=flow.shape
    y,x=torch.meshgrid(torch.arange(h,device=flow.device,dtype=torch.float32),torch.arange(w,device=flow.device,dtype=torch.float32),indexing='ij')
    px=x[None]+flow[:,0].float();py=y[None]+flow[:,1].float()
    grid=torch.stack((2*(px+.5)/w-1,2*(py+.5)/h-1),-1)
    sampled=F.grid_sample(value.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    valid=((px>=0)&(px<=w-1)&(py>=0)&(py<=h-1)).unsqueeze(1)
    return sampled,valid

def flow_valid(backward,forward):
    warped,bounds=backwarp(forward,backward)
    residual=(backward+warped).square().sum(1,keepdim=True)
    threshold=.01*(backward.square().sum(1,keepdim=True)+warped.square().sum(1,keepdim=True))+.5
    return (bounds & (residual<=threshold)).detach()

def warp_feature(previous,flow,valid):
    h,w=previous.shape[-2:];rh,rw=flow.shape[-2:]
    small=F.interpolate(flow.float(),size=(h,w),mode='bilinear',align_corners=False)
    small=small*small.new_tensor([w/rw,h/rh]).reshape(1,2,1,1)
    mask=F.interpolate(valid.float(),size=(h,w),mode='nearest')
    warped,bounds=backwarp(previous,small.detach())
    return warped*mask.detach()*bounds

def sobel(x):
    # Derivative unit = normalized 3x3 Sobel, replicate boundary.
    k=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    c=x.shape[1];xx=F.pad(x.float(),(1,1,1,1),mode='replicate')
    return torch.cat([F.conv2d(xx,kk.expand(c,1,3,3),groups=c) for kk in (k,k.t())],1)

def temporal_errors(current,previous,target,previous_gt,flow,valid):
    count=valid.float().sum()
    if count.item()==0:return None,None,float(valid.float().mean())
    errors=[]
    for op in (lambda x:x,sobel):
        a=op(current);b=op(previous);g=op(target);p=op(previous_gt)
        b,_=backwarp(b,flow);p,_=backwarp(p,flow)
        errors.append(((a-b)-(g-p)).abs().mul(valid).sum()/(count*a.shape[1]))
    return errors[0],errors[1],float(valid.float().mean())

def checks(device):
    x=torch.arange(77,device=device,dtype=torch.float32).reshape(1,1,7,11)
    zero=torch.zeros(1,2,7,11,device=device);y,v=backwarp(x,zero)
    z=zero.clone();z[:,0]=1;yy,vv=backwarp(x,z)
    assert torch.allclose(y,x,rtol=0,atol=1e-5)
    assert torch.allclose(yy[:,:,:,:-1],x[:,:,:,1:],rtol=0,atol=1e-5)
    assert not vv[:,:,:,-1].any() and vv[:,:,:,:-1].all()
    t=x.clone().requires_grad_();warped,_=backwarp(t,z);warped.sum().backward();assert t.grad.norm()>0
    f=torch.ones(1,1,14,22,device=device);out=warp_feature(f,z,torch.ones_like(v))
    assert out.shape==f.shape and out[:,:,:,-2:].eq(0).all()
    return dict(status='PASS',version=VERSION,identity_max_abs=float((y-x).abs().max()),one_pixel_positive_dx_samples_right=True,align_corners=False,padding='zeros',flow_units='RGB pixels; resized by feature/RGB width,height',FB_threshold='.01*(||b||^2+||f_warp||^2)+0.5',invalid_history_zero=True,warp_history_gradient=True)
