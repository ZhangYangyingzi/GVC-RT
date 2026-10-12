"""Read-only discriminator, image/condition and full-window gradient probes."""
from io21 import *
import argparse,fcntl,random,math,traceback

def ccache(model,panel):return ROOT/'cache/diagnostic'/model/(panel+'.pt')
def row_for(panel):return next(r for r in load(ROOT/'diagnostic_plan.json')['updates'] if r['panel_id']==panel)
def checkpoint(step):return load(V26/'branches/C_decoder_illm/checkpoint_index.json')[str(step)]
def load_g(step,device):
 import torch
 v4,im,pm,models,quality=load_training(device)
 if step is not None:
  item=checkpoint(step);assert sha(item['path'])==item['sha256'];s=torch.load(item['path'],map_location='cpu',weights_only=True)
  for k,m in models.items():m.load_state_dict(s[k],strict=True)
  assert {k:v4.module_hash(m) for k,m in models.items()}==item['module_hashes'];del s
 models['wrapper'].eval().requires_grad_(False)
 return v4,im,pm,models,quality

def reconstruct(model,panel,device):
 import torch
 from window_training import replay
 path=ccache(model,panel)
 if path.exists():
  s=torch.load(path,map_location='cpu',weights_only=True);assert s['plan_sha256']==sha(ROOT/'diagnostic_plan.json') and s['model_sha256']==(EXPECTED if model=='B1' else checkpoint(int(model[1:]))['sha256']);return s
 row=row_for(panel);frames=replay(row,'C_decoder_illm',device);step=None if model=='B1' else int(model[1:]);v4,im,pm,models,quality=load_g(step,device);before={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);captures=[];latent=[];hh=pm.dec.register_forward_hook(lambda m,a,o:captures.append(o));lh=models['bridge'][0].register_forward_pre_hook(lambda m,a:latent.append(a[0].detach()));common=module('diag_qp',V62/'fullqp_common.py');q=row['external_qp'];recframes=[];conditions=[];warm=[]
 try:
  with torch.no_grad():
   pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),q)['x_hat'])
   for pos in range(1,64):
    latent.clear();qa=common.frame_qp(pm,q,pos);rec,_,_=v4.joint_ste_forward(pm,models['wrapper'](frames[pos]),qa);assert len(captures)==1 and len(latent)==1
    condition=latent.pop();assert condition.shape==torch.Size([1,1024,16,16]) and not condition.requires_grad
    pm.add_ref_frame(captures.pop().detach(),(rec.detach()*2-1).half());assert not pm.dpb[0].feature.requires_grad
    if pos>=49:recframes.append(rec.cpu());conditions.append(condition.cpu())
    else:warm.append(dict(frame=pos,QP=qa,reference_sha256=hashlib.sha256(rec.cpu().contiguous().numpy().tobytes()).hexdigest()))
 finally:hh.remove();lh.remove()
 assert before=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm)
 s=dict(reconstructions=recframes,conditions=conditions,warmup=warm,plan_sha256=sha(ROOT/'diagnostic_plan.json'),model_sha256=EXPECTED if model=='B1' else checkpoint(step)['sha256'],source_cropped_RGB_sha256=row['cropped_RGB_sha256'],source_indices=row['source_frame_indices'],model_hashes=before,state_unchanged=True,no_grad=True,causal_reference=True,optimizer_updates=0);atomic_torch(path,s);return s

def classification(logits,y):
 import torch
 from illm_runtime import local_ce
 target=y+1;probs=logits.softmax(1);top=logits.topk(5,dim=1).indices;ctop=logits[:,1:].topk(5,dim=1).indices;pf=float(probs[:,0].mean());content=1-pf
 return dict(CE_content=float(local_ce(logits,target)),CE_fake=float(local_ce(logits,torch.zeros_like(y))),p_fake=pf,content_probability_mass=content,all_class_top1=float((top[:,0]==target).float().mean()),all_class_top5=float((top==target[:,None]).any(1).float().mean()),content_only_top1=float((ctop[:,0]==y).float().mean()),content_only_top5=float((ctop==y[:,None]).any(1).float().mean()),fake_class_accuracy=float((top[:,0]==0).float().mean())),dict(GT=torch.bincount(y.flatten(),minlength=1024).tolist(),predicted_all=torch.bincount(top[:,0].flatten(),minlength=1025).tolist(),predicted_content=torch.bincount(ctop[:,0].flatten(),minlength=1024).tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--panel',required=True);ap.add_argument('--step',type=int,choices=(0,250,500,1000));ap.add_argument('--cache-only',action='store_true');a=ap.parse_args()
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8');import torch,numpy as np
 torch.set_num_threads(2);torch.manual_seed(20261027);torch.cuda.manual_seed_all(20261027);random.seed(20261027);np.random.seed(20261027);torch.use_deterministic_algorithms(True)
 device=torch.device('cuda:0');row=row_for(a.panel);out=ROOT/'diagnostics/points'/f'{a.panel}_s{a.step}.json';lockpath=ROOT/'diagnostics/locks'/f'{a.panel}_{a.step}_{a.cache_only}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True);lk=lockpath.open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not a.cache_only and out.exists() and load(out).get('status')=='PASS':return
 torch.cuda.reset_peak_memory_stats()
 if a.cache_only:
  reconstruct('B1',a.panel,device);dump(ROOT/'diagnostics/cache_status'/f'{a.panel}.json',dict(status='PASS',cache_sha256=sha(ccache('B1',a.panel)),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20));print('CACHE PASS',a.panel,flush=True);return
 from window_training import replay,make_runner
 from illm_runtime import create_illm
 from gradient_tools import stats,dists_details
 fixed=reconstruct('B1',a.panel,device);shuffled=torch.load(ccache('B1',row['shuffle_panel_id']),map_location='cpu',weights_only=True);matched=reconstruct('C'+str(a.step),a.panel,device);frames=replay(row,'C_decoder_illm',device)
 disc=create_illm(device);pretrained_feature_hash=tensor_hash(disc.feature.state_dict());item=checkpoint(a.step);state=torch.load(item['path'],map_location='cpu',weights_only=True);disc.load_state_dict(state['projected'],strict=True);assert tensor_hash(disc.feature.state_dict())==pretrained_feature_hash;assert tensor_hash(disc.state_dict())==tensor_hash(state['projected']);del state;disc.g_mode();dh=disc.d_hash();fh=tensor_hash(disc.feature.state_dict());per=[];hist=[]
 # Independent labels: strict official pretrained weights, no mutable VQ counters.
 gtlabels=[disc.labels(f).clone() for f in frames[49:64]]
 assert all(y.shape==torch.Size([1,32,32]) and y.dtype==torch.long and 0<=y.min()<=y.max()<1024 for y in gtlabels)
 kernel=torch.exp(-torch.arange(-2,3,device=device,dtype=torch.float32).square()/2);kernel/=kernel.sum();kk=(kernel[:,None]*kernel[None,:])[None,None].repeat(3,1,1,1)
 import torch.nn.functional as F
 with torch.no_grad():
  for i,(gt,y) in enumerate(zip(frames[49:64],gtlabels)):
   for panel,rec,cc in [('fixed',fixed['reconstructions'][i],fixed['conditions'][i]),('matched',matched['reconstructions'][i],matched['conditions'][i])]:
    images={'real':gt,'reconstruction':rec.to(device)}
    if panel=='fixed':images['blurred_real']=F.conv2d(F.pad(gt,(2,2,2,2),mode='reflect'),kk,groups=3)
    c=cc.to(device).detach();conds={'matched':c}
    if panel=='fixed':conds.update(zero=torch.zeros_like(c),shuffled=shuffled['conditions'][i].to(device).detach())
    for image,x in images.items():
     for cm,condition in conds.items():
      logits=disc.scores(x,condition)[0];values,counts=classification(logits,y);identity=dict(panel_id=a.panel,panel_type=row['panel'],domain=row['domain'],external_QP=row['external_qp'],checkpoint=a.step,reconstruction_panel=panel,frame_index=49+i,source_frame_index=row['source_frame_indices'][49+i],image=image,condition=cm,actual_QP=row['actual_qps'][i]);per.append(dict(identity,**values));hist.append(dict(identity,**counts))
 # Full matched no-update gradient replay, identical 15-frame loss average and FP32 recovery.
 v4,im,pm,models,quality=load_g(a.step,device);gb={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality];params=[p for m in models.values() for p in m.parameters() if p.requires_grad];imagegrads=[];oldadv=disc.adversarial
 def captured_adv(x,c):
  assert not c.requires_grad;s=oldadv(x,c);g=torch.autograd.grad(s,x,retain_graph=True)[0];imagegrads.append(stats([g],[x]));return s
 disc.adversarial=captured_adv
 dists=[];oldonce=quality[1].forward_once;oldforward=quality[1].forward
 def traced_forward(x,y,**kw):
  if dists:return oldforward(x,y,**kw)
  assert kw.get('require_grad') is True;score,layers,terms=dists_details(quality,x,y,True) if False else (None,None,None)
  captures=[]
  def grab(z):f=oldonce(z);captures.append(f);return f
  quality[1].forward_once=grab
  try:s=oldforward(x,y,**kw)
  finally:quality[1].forward_once=oldonce
  ds=quality[1];weights=ds.alpha.sum()+ds.beta.sum();al=torch.split(ds.alpha/weights,ds.chns,1);be=torch.split(ds.beta/weights,ds.chns,1);u,v=captures[0][-1],captures[1][-1];um=u.mean((2,3),keepdim=True);vm=v.mean((2,3),keepdim=True);uv=(u-um).square().mean((2,3),keepdim=True);vv=(v-vm).square().mean((2,3),keepdim=True);cov=(u*v).mean((2,3),keepdim=True)-um*vm;term=-((al[-1]*(2*um*vm+1e-6)/(um.square()+vm.square()+1e-6)).sum(1)+(be[-1]*(2*cov+1e-6)/(uv+vv+1e-6)).sum(1)).mean();gs=torch.autograd.grad(term*.5/15,params,retain_graph=True,allow_unused=True);off=0
  for k in ('bridge','generator'):
   ps=list(models[k].parameters());dists.append(dict(module=k,deepest_term=stats(gs[off:off+len(ps)],ps)));off+=len(ps)
  assert all(t['deepest_term']['norm']>0 and t['deepest_term']['finite'] for t in dists);return s
 quality[1].forward=traced_forward
 runner=make_runner(v4,im,pm,models,quality,disc);sums,actual,tr,warm,fakes=runner(frames,row['external_qp'],'C_decoder_illm');quality[1].forward=oldforward;disc.adversarial=oldadv
 assert actual==row['actual_qps'] and len(warm)==48 and len(tr)==15;assert [x['reconstruction_sha256'] for x in tr]==[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in matched['reconstructions']]
 preclip=sum(float(p.grad.double().square().sum()) for p in params if p.grad is not None)**.5;clip=min(1.,1/(preclip+1e-6));modulegrad={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()};assert all(g['finite'] for g in modulegrad.values())
 assert sums['adv_B1_ratio']<=.05+1e-8;assert all(g['finite'] and g['norm']>0 for g in imagegrads)
 assert gb=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality] and dh==disc.d_hash() and fh==tensor_hash(disc.feature.state_dict())
 ff=out.with_suffix('.frames.csv');write(ff,per);hp=out.with_suffix('.histograms.jsonl');hp.parent.mkdir(parents=True,exist_ok=True);hp.write_text(''.join(json.dumps(x)+'\n' for x in hist))
 grad=dict(sums,preclip_norm=preclip,clip_coefficient=clip,module_gradients=modulegrad,image_gradients=imagegrads,DISTS_deep=dists,optimizer_steps=0,condition_detached=True,states_unchanged=True)
 dump(out,dict(status='PASS',panel_id=a.panel,panel_type=row['panel'],domain=row['domain'],QP=row['external_qp'],checkpoint=a.step,checkpoint_SHA256=item['sha256'],D_state_hash=dh,labeler_hash=fh,plan_SHA256=sha(ROOT/'diagnostic_plan.json'),frame_csv=str(ff),frame_csv_SHA256=sha(ff),histogram_path=str(hp),histogram_SHA256=sha(hp),gradient=grad,parameters_buffers_unchanged=True,no_training=True,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,physical_gpu=a.gpu,logical_gpu=0));print('DIAGNOSTIC PASS',a.panel,a.step,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs'/f'diagnostic_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
