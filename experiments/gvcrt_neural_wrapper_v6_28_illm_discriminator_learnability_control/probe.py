from runtime import *
import argparse,traceback

def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,required=True);x=a.parse_args();device=init(x.gpu);torch.cuda.reset_peak_memory_stats();cfg=load(ROOT/'config.json')
 from illm_runtime import local_ce,official_modules
 official_modules();import types,importlib
 pkg=types.ModuleType('neuralcompression.loss_fn');pkg.__path__=[str(V26/'vendor/NeuralCompression-3f122808255a2278eea049119367eb3abc434862/neuralcompression/loss_fn')];sys.modules['neuralcompression.loss_fn']=pkg
 official=importlib.import_module('neuralcompression.loss_fn._oasis_loss').OASISDiscriminatorLoss()
 r=torch.randn(1,1025,32,32,requires_grad=True);f=torch.randn_like(r,requires_grad=True);y=torch.randint(1024,(1,32,32));lo=official(r,f,y);ll=.5*(local_ce(r,y+1)+local_ce(f,torch.zeros_like(y)));go=torch.autograd.grad(lo,(r,f),retain_graph=True);gl=torch.autograd.grad(ll,(r,f));assert torch.allclose(lo,ll,atol=1e-6) and all(torch.allclose(a,b,atol=1e-8) for a,b in zip(go,gl))
 dump(ROOT/'audits/OASIS_equivalence.json',dict(status='PASS',source=str(V26/'vendor/NeuralCompression-3f122808255a2278eea049119367eb3abc434862/neuralcompression/loss_fn/_oasis_loss.py'),official_loss=float(lo),adapted_loss=float(ll),absolute_error=float((lo-ll).abs()),gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(go,gl)),CPU_reference=True,adaptation='log_softmax/gather deterministic GPU equivalent of official CE; same y+1 real and0fake'))
 # Fixed source cache equivalence with V27, not an extra codec benchmark.
 first=load(ROOT/'plans/tiny.json')['windows'][0]['row_id'];raw=buildcache(first,device);cache_peak=torch.cuda.max_memory_reserved()/2**20;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 hashes={};records=[]
 for b in BRANCHES28:
  d=createD(device);hashes[b]=tensor_hash(d.state_dict());original={k:v.clone() for k,v in d.state_dict().items()};before=dict(d.named_buffers());buf=tensor_hash(before);pp=pairs(raw,[49,63]);pp=pp*4;opt=optfor(d,b);assert not opt.state
  out=Dstep(d,opt,pp,b,device);main_out=Dstep(d,opt,pairs(raw),b,device);assert tensor_hash(d.state_dict())!=hashes[b] and tensor_hash(dict(d.named_buffers()))==buf
  changes={n:sum(float((p.detach()-original[n]).double().square().sum()) for n,p in d.named_parameters() if n.startswith(prefix))**.5 for prefix,n in [('input_sequence','condition_adapter'),('_unet','image_unet')]}
  c=condition(raw['conditions'][0].to(device),b);assert torch.isfinite(c).all()
  if cfg['branches'][b]['condition']=='zero':assert not c.any()
  if cfg['branches'][b]['condition']=='rms':assert abs(float(c.square().mean().sqrt())-1)<1e-5
  d.eval().requires_grad_(False);rec=raw['reconstructions'][0].to(device).requires_grad_(True);dh=tensor_hash(d.state_dict());loss=local_ce(d(rec,c),raw['labels'][0].to(device)+1);gg=torch.autograd.grad(loss,rec)[0];assert torch.isfinite(gg).all() and gg.norm()>0 and tensor_hash(d.state_dict())==dh
  records.append(dict(branch=b,status='PASS',**out,parameter_change=changes,readonly_CE_G_image_gradient=float(gg.norm()),buffers_unchanged=True,optimizer_fresh=True,FP32_D=True,no_G_updates=True));del d,opt,rec,gg;torch.cuda.empty_cache()
 assert len(set(hashes.values()))==1
 dump(ROOT/'audits/preflight.json',dict(status='PASS',D_initial_hashes=hashes,branches=records,cache_peak_MiB=cache_peak,D_peak_MiB=torch.cuda.max_memory_reserved()/2**20,physical_gpu=x.gpu,logical_gpu=0,no_G_updates=True,no_new_codec_evaluation=True,frame_average_equivalence=True))
 print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'audits/preflight_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
