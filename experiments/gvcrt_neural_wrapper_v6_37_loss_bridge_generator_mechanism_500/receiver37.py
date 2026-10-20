"""Unmodified actual bridge path; intervention only at the decoder's input."""
from io32 import *
def runtime_original():
 from adapter35 import runtime
 return runtime(dict(method='original',stage='baseline'))
def payloads():
 cfg=load(ROOT/'evaluation_config.json');out={}
 for name,key in [('B_basicLoss_noP_500','B_basicLoss_noP_500'),('F_fullLoss_noP_500','N_oursLoss_noP_500')]:out[name]=cfg['checkpoints'][key]
 cp=ROOT/'branches/D_basicDISTS_noP_500/checkpoint_index.json'
 if cp.exists() and '500' in load(cp):out['D_basicDISTS_noP_500']=load(cp)['500']['inference']
 return out
def receiver_bank(pm,names):
 import copy,torch
 bank={};spec=payloads()
 for name in names:
  m=copy.deepcopy(pm.recon_generation_net)
  if name!='O_original':
   p=spec[name];assert sha(p['path'])==p['sha256'];s=torch.load(p['path'],map_location='cpu',weights_only=True);m.mlp.load_state_dict(s['bridge'],strict=True);m.decoder.load_state_dict(s['generator'],strict=True)
  m.eval().requires_grad_(False);bank[name]=m
 return bank
def forward(m,f,q):
 cap=[];h=m.decoder.register_forward_pre_hook(lambda _,a:cap.append((a[0],a[1])))
 try:x=m(f,q)
 finally:h.remove()
 assert len(cap)==1 and cap[0][0].shape[1]==18
 return x,cap[0][0]
def metadata(t):
 return next(v for v in load(ROOT/'eval_manifest.json')[t['dataset']] if v['video_index']==t['video'])
def sourcepoint(t,m):
 return next(x for x in read(ROOT/'results/reused_points.csv') if (x['dataset'],int(x['video']),int(x['QP']),x['method'])==(t['dataset'],t['video'],t['qp'],m))
def tag(t):return f'{t["dataset"]}_{t["video"]}_q{t["qp"]}'
def th(t):
 import torch
 assert torch.isfinite(t).all()
 return dict(sha256=hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),shape=list(t.shape),dtype=str(t.dtype))
def difference(a,b):
 import torch
 da=(a.double()-b.double());den=float(b.double().norm());rel=float(da.norm()/max(den,1e-12));return dict(bitwise=torch.equal(a,b),maxabs=float(da.abs().max()),relative_L2=rel,reference_zero=den==0,epsilon=1e-12)
def latent_stats(z):
 a=z.detach().double().flatten(2)[0];mu=a.mean(1);c=a-mu[:,None];cov=c@c.T/a.shape[1]
 return dict(channels=18,spatial_samples=a.shape[1],variance_ddof=0,mean=mu.cpu().tolist(),variance=cov.diag().cpu().tolist(),RMS=a.square().mean(1).sqrt().cpu().tolist(),L2=a.square().sum(1).sqrt().cpu().tolist(),covariance18x18=cov.cpu().tolist(),sum=a.sum(1).cpu().tolist(),cross_sum=(a@a.T).cpu().tolist())
def latent_delta(a,b):
 import torch
 x=a.detach().double().flatten();y=b.detach().double().flatten();d=y-x;nx=float(x.norm());ny=float(y.norm());eps=1e-12
 return dict(relative_L2=float(d.norm()/max(nx,eps)),cosine_distance=1-float(x.dot(y))/max(nx*ny,eps),reference_norm=nx,target_norm=ny,denominator_zero=nx==0 or ny==0,epsilon=eps,mean_delta=float(y.mean()-x.mean()),variance_delta=float(y.var(unbiased=False)-x.var(unbiased=False)))
def parse(path,eco):
 import io
 stream=io.BytesIO(Path(path).read_bytes());helper=eco.SPSHelper()
 for i in range(64):
  start=stream.tell();h=eco.read_header(stream)
  while h['nal_type']==eco.NalType.NAL_SPS:helper.add_sps_by_id(eco.read_sps_remaining(stream,h['sps_id']));h=eco.read_header(stream)
  sp=helper.get_sps_by_id(h['sps_id']);q,bits=eco.read_ip_remaining(stream);assert (h['nal_type']==eco.NalType.NAL_I)==(i==0)
  yield i,sp,q,bits,(stream.tell()-start)*8
 assert stream.tell()==Path(path).stat().st_size
