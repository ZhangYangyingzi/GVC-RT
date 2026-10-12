"""D-only runtime: strict D0, exact labels, immutable B1 cached causal forward."""
from io21 import *
import torch,random,numpy as np,fcntl,contextlib
V27=ROOT.parent/'gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit'
BRANCHES28=('A_raw_current','B_rms_current','C_zero_current','D_raw_official_optimizer')
def init(gpu):
 assert gpu in (4,5,6,7);os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(20261028);torch.cuda.manual_seed_all(20261028);random.seed(20261028);np.random.seed(20261028);return torch.device('cuda:0')
def rowkey(row):return 'main_'+str(row['adaptation_step']).zfill(4) if 'panel_id' not in row else 'val_'+row['panel_id']
def rowfor(key):
 if key.startswith('main_'):return load(ROOT/'plans/shared.json')['updates'][int(key[5:])-1]
 return next(x for x in load(ROOT/'plans/validation.json')['updates'] if key=='val_'+x['panel_id'])
def cachepath(key):return ROOT/'cache/raw'/f'{key}.pt'
def condition(c,branch):
 c=c.detach().float();mode=load(ROOT/'config.json')['branches'][branch]['condition']
 if mode=='zero':return torch.zeros_like(c)
 if mode=='rms':return c/c.square().mean().sqrt().detach().clamp_min(1e-6)
 return c

def createD(device):
 from illm_runtime import official_modules
 _,_,u=official_modules()
 with torch.random.fork_rng(devices=[0]):torch.manual_seed(20261028);d=u.ConditionalUnetDiscriminator(output_channels=1025,output_downsampling_factor=8,context_in=1024,context_out=16,norm_type='identity').to(device).float()
 item=load(ROOT/'config.json')['D_source_checkpoint'];s=torch.load(item['path'],map_location='cpu',weights_only=True);assert s['adaptation_step']==0 and s['D_updates']==0;state={k.removeprefix('discriminator.'):v for k,v in s['projected'].items() if k.startswith('discriminator.')};d.load_state_dict(state,strict=True);assert tensor_hash(d.state_dict())==tensor_hash(state);return d

def optfor(d,b):
 c=load(ROOT/'config.json')['branches'][b];klass=getattr(torch.optim,c['optimizer']);return klass(d.parameters(),lr=c['lr'],betas=tuple(float(v) for v in c['betas']),weight_decay=0.)
def state_rng():
 n=np.random.get_state();return dict(python=random.getstate(),numpy=[n[0],torch.from_numpy(n[1].astype('int64')),n[2],n[3],n[4]],torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
 random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),n[2],n[3],n[4]));torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def safe_cache(key):
 p=cachepath(key);meta=ROOT/'cache/status'/f'{key}.json'
 if not meta.exists():return None
 m=load(meta);assert m['status']=='PASS' and p.stat().st_size==m['bytes'];s=torch.load(p,map_location='cpu',weights_only=True);r=rowfor(key);assert s['protocol_sha256']==sha(ROOT/'protocol.json') and s['B1_sha256']==EXPECTED and s['labeler_sha256']==load(ROOT/'config.json')['labeler_sha256'] and s['row_hash']==hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest();return s

def buildcache(key,device):
 lk=ROOT/'cache/locks'/f'{key}.lock';lk.parent.mkdir(parents=True,exist_ok=True)
 with lk.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  old=safe_cache(key)
  if old:return old
  from window_training import replay
  from illm_runtime import labeler
  row=rowfor(key);frames=replay(row,'C_decoder_illm',device);source=None;legacy=None
  vals=load(ROOT/'plans/validation.json')['updates']
  matches=[r for r in vals if r['cropped_RGB_sha256']==row['cropped_RGB_sha256'] and r['source_frame_indices']==row['source_frame_indices'] and r['external_qp']==row['external_qp']]
  if matches:
   legacy=V27/'cache/diagnostic/B1'/f"{matches[0]['panel_id']}.pt"
   if legacy.exists():
    prior=load(V27/'diagnostics/cache_status'/f"{matches[0]['panel_id']}.json");assert prior['status']=='PASS' and sha(legacy)==prior['cache_sha256']
    x=torch.load(legacy,map_location='cpu',weights_only=True);assert x['model_sha256']==EXPECTED and x['source_cropped_RGB_sha256']==row['cropped_RGB_sha256'] and x['source_indices']==row['source_frame_indices'] and x['state_unchanged'] and x['no_grad'];assert x['model_hashes']==load(ROOT/'config.json')['source_checkpoint']['module_hashes'];assert x['plan_sha256'] in load(ROOT/'audits/validation_plan_revisions/revision.json')['accepted_plan_hashes'];source=str(legacy);recs=x['reconstructions'];conds=x['conditions'];warm=x['warmup'];del x
  codec_hash=None
  if not source:
   v4,im,pm,models,quality=load_training(device)
   for m in models.values():m.requires_grad_(False)
   before={k:v4.module_hash(m) for k,m in models.items()};codec_hash=v4.compression_hash(im,pm);assert codec_hash==load(ROOT/'config.json')['source_checkpoint']['compression_hash'];assert all(not p.requires_grad for m in (im,pm,models['wrapper']) for p in m.parameters())
   buffers=tensor_hash({str(i)+'.'+k:v for i,m in enumerate((im,pm,models['wrapper'])) for k,v in m.named_buffers()})
   cap=[];lat=[];h=pm.dec.register_forward_hook(lambda m,a,o:cap.append(o));lh=models['bridge'][0].register_forward_pre_hook(lambda m,a:lat.append(a[0].detach()));common=module('cache_qp',V62/'fullqp_common.py');recs=[];conds=[];warm=[]
   try:
    with torch.no_grad():
     pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),row['external_qp'])['x_hat'])
     for pos in range(1,64):
      qa=common.frame_qp(pm,row['external_qp'],pos);rec,_,_=v4.joint_ste_forward(pm,models['wrapper'](frames[pos]),qa);assert len(cap)==1 and len(lat)==1;c=lat.pop();assert c.shape==torch.Size([1,1024,16,16]);pm.add_ref_frame(cap.pop().detach(),(rec.detach()*2-1).half())
      assert not pm.dpb[0].feature.requires_grad and not rec.requires_grad
      if pos>=49:assert qa==row['actual_qps'][pos-49];recs.append(rec.cpu());conds.append(c.cpu())
      else:warm.append(dict(frame=pos,QP=qa,reference_sha256=hashlib.sha256(rec.cpu().numpy().tobytes()).hexdigest()))
   finally:h.remove();lh.remove()
   assert before=={k:v4.module_hash(m) for k,m in models.items()} and codec_hash==v4.compression_hash(im,pm);assert buffers==tensor_hash({str(i)+'.'+k:v for i,m in enumerate((im,pm,models['wrapper'])) for k,v in m.named_buffers()})
   del v4,im,pm,models,quality
   torch.cuda.empty_cache()
  lab=labeler(device);lh=tensor_hash(lab.state_dict());ys=[]
  with torch.no_grad():
   for gt in frames[49:64]:y=lab(gt).reshape(1,32,32).long();assert y.min()>=0 and y.max()<1024;ys.append(y.cpu())
  assert lh==tensor_hash(lab.state_dict());del lab
  raw=dict(GT=[f.cpu() for f in frames[49:64]],reconstructions=recs,conditions=conds,labels=ys,actual_QPs=row['actual_qps'],source_indices=row['source_frame_indices'],cropped_RGB_sha256=row['cropped_RGB_sha256'],full_RGB_sha256=row['full_RGB_sha256'],source_video=row['source_path'],source_video_sha256=row['source_sha256'],crop=[row['crop_x'],row['crop_y'],256],B1_sha256=EXPECTED,labeler_sha256=load(ROOT/'config.json')['labeler_sha256'],row_hash=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),protocol_sha256=sha(ROOT/'protocol.json'),legacy_source=source,legacy_sha256=sha(legacy) if source else None,warmup=warm,parameters_buffers_unchanged=True,no_grad=True,causal_reference=True,G_updates=0)
  atomic_torch(cachepath(key),raw);dump(ROOT/'cache/status'/f'{key}.json',dict(status='PASS',path=str(cachepath(key)),sha256=sha(cachepath(key)),bytes=cachepath(key).stat().st_size,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,reuse_source=source,legacy_sha256=raw['legacy_sha256'],row_hash=raw['row_hash']));return raw

def pairs(raw,positions=None):
 inds=[p-49 for p in positions] if positions else range(15)
 return [(raw['GT'][i],raw['reconstructions'][i],raw['conditions'][i],raw['labels'][i],i+49) for i in inds]
def tiny_pairs():
 out=[]
 for w in load(ROOT/'plans/tiny.json')['windows']:out.extend(pairs(safe_cache(w['row_id']),w['positions']))
 assert len(out)==8;return out

def gradients(d):
 out={}
 for name,ps in [('condition_adapter',d.input_sequence.parameters()),('image_unet',d._unet.parameters())]:
  ps=list(ps);g=[p.grad for p in ps if p.grad is not None];out[name+'_grad_norm']=sum(float(x.detach().double().square().sum()) for x in g)**.5;out[name+'_grad_finite']=all(bool(torch.isfinite(x).all()) for x in g)
 return out

def Dstep(d,opt,pp,branch,device,details=False):
 from illm_runtime import local_ce
 d.train().requires_grad_(True);opt.zero_grad(set_to_none=True);losses=[0.,0.];den=len(pp);assert den in (8,15)
 for gt,rec,c,y,_ in pp:
  gt=gt.to(device);rec=rec.to(device).detach();y=y.to(device);c=condition(c.to(device),branch);assert not c.requires_grad and not rec.requires_grad
  real=d(gt,c);fake=d(rec,c);lr=local_ce(real,y+1);lf=local_ce(fake,torch.zeros_like(y));loss=(lr+lf)*(.5/den);assert bool(torch.isfinite(loss));loss.backward();losses[0]+=float(lr.detach())/den;losses[1]+=float(lf.detach())/den
 g=gradients(d);assert all(g[k] for k in g if k.endswith('finite'))
 assert all(p.dtype==torch.float32 and (p.grad is None or p.grad.dtype==torch.float32) for p in d.parameters())
 # No clipping, loss scaling or AMP: only the fixed D optimizer.
 opt.step();assert all(bool(torch.isfinite(p).all()) for p in d.parameters())
 return dict(CE_real=losses[0],CE_fake=losses[1],D_loss=sum(losses)*.5,frames=den,normalization_denominator=den,**g)
