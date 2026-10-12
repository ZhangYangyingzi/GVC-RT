from runtime import *
import argparse,traceback,math
from illm_runtime import local_ce

def moments(t):
 t=t.detach().float();return dict(mean=float(t.mean()),std=float(t.std(unbiased=False)),RMS=float(t.square().mean().sqrt()),min=float(t.min()),max=float(t.max()),finite=bool(torch.isfinite(t).all()))
def classify(logits,y):
 logits=logits.detach();p=logits.softmax(1);top=logits.topk(5,1).indices;co=logits[:,1:].topk(5,1).indices;target=y+1
 return dict(CE_content=float(local_ce(logits,target)),CE_fake=float(local_ce(logits,torch.zeros_like(y))),CE_content_only=float(local_ce(logits[:,1:],y)),p_fake=float(p[:,0].mean()),all_class_top1=float((top[:,0]==target).float().mean()),all_class_top5=float((top==target[:,None]).any(1).float().mean()),content_top1=float((co[:,0]==y).float().mean()),content_top5=float((co==y[:,None]).any(1).float().mean()),fake_accuracy=float((top[:,0]==0).float().mean()),fake_prediction_fraction=float((top[:,0]==0).float().mean())),dict(GT=torch.bincount(y.flatten(),minlength=1024).cpu().tolist(),predicted_all=torch.bincount(top[:,0].flatten(),minlength=1025).cpu().tolist(),predicted_content=torch.bincount(co[:,0].flatten(),minlength=1024).cpu().tolist())
def run_panel(a,device,d,item):
 out=ROOT/'validation'/a.branch/a.stage/f'step_{a.step:05d}'/a.panel;out.mkdir(parents=True,exist_ok=True);lk=(out/'validation.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'integrity.json').exists() and load(out/'integrity.json')['status']=='PASS':return
 before=tensor_hash(d.state_dict());torch.cuda.reset_peak_memory_stats()
 if a.panel=='tiny_fit':
  data=[]
  for w in load(ROOT/'plans/tiny.json')['windows']:
   row=rowfor(w['row_id']);data.extend((row,pp) for pp in pairs(safe_cache(w['row_id']),w['positions']))
 else:
  row=rowfor(a.panel);data=[(row,pp) for pp in pairs(safe_cache(a.panel))]
 records=[];hists=[];scales=[];activation=[]
 hh=d.input_sequence.register_forward_hook(lambda m,i,o:activation.append(('condition_adapter',moments(o))))
 # First actual image/context convolution in the official U-Net.
 firstconv=next(m for m in d._unet.modules() if isinstance(m,torch.nn.Conv2d));fh=firstconv.register_forward_hook(lambda m,i,o:activation.append(('early_unet',moments(o))))
 try:
  for row,(gt,rec,c,y,pos) in data:
   gt=gt.to(device);rec=rec.to(device).detach();c=c.to(device).detach();y=y.to(device);cc=condition(c,a.branch);assert cc.shape==torch.Size([1,1024,16,16]) and not cc.requires_grad
   for kind,x in [('real',gt),('reconstruction',rec)]:
    activation.clear()
    with torch.no_grad():logits=d(x,cc);values,hist=classify(logits,y)
    identity=dict(branch=a.branch,stage=a.stage,checkpoint=a.step,panel=row.get('panel','tiny_fit'),panel_id=row.get('panel_id',rowkey(row)),domain=row['domain'],external_QP=row['external_qp'],frame_index=pos,source_frame_index=row['source_frame_indices'][pos],actual_QP=row['actual_qps'][pos-49],image=kind)
    records.append(dict(identity,**values));hists.append(dict(identity,**hist));scales.append(dict(identity,image_scale=moments(x),raw_condition=moments(c),transformed_condition=moments(cc),activations=dict(activation)))
   # Read-only CE_G image gradient: frozen D and no G model loaded.
   rec=rec.requires_grad_(True);loss=local_ce(d(rec,cc),y+1);g=torch.autograd.grad(loss,rec)[0];assert torch.isfinite(g).all();records[-1]['CE_G_image_grad_norm']=float(g.norm());records[-1]['CE_G_image_grad_finite']=True
 finally:hh.remove();fh.remove()
 assert before==tensor_hash(d.state_dict()) and all(s['image_scale']['finite'] and s['raw_condition']['finite'] and s['transformed_condition']['finite'] and all(v['finite'] for v in s['activations'].values()) for s in scales)
 write(out/'frames.csv',records)
 for name,rows in [('histograms',hists),('scales',scales)]:
  with (out/(name+'.jsonl')).open('w') as f:
   for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
 dump(out/'integrity.json',dict(status='PASS',checkpoint_sha256=item['sha256'],D_hash=before,parameters_buffers_unchanged=True,frames=len(data),images=len(records),condition_mode=load(ROOT/'config.json')['branches'][a.branch]['condition'],condition_detached=True,same_real_fake_condition=True,G_updates=0,D_updates=0,CE_G_readonly=True,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,physical_gpu=a.gpu,logical_gpu=0,files={str(p.name):sha(p) for p in out.iterdir() if p.suffix in ('.csv','.jsonl')}));print('VALIDATION PASS',a.branch,a.stage,a.step,a.panel,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--branch',choices=BRANCHES28,required=True);p.add_argument('--stage',choices=('TINY','MAIN'),required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--panel',nargs='+',required=True);a=p.parse_args();device=init(a.gpu);panels=a.panel
 item=load(ROOT/'branches'/a.branch/a.stage/'checkpoint_index.json')[str(a.step)];assert sha(item['path'])==item['sha256'];s=torch.load(item['path'],map_location='cpu',weights_only=True);d=createD(device);d.load_state_dict(s['discriminator'],strict=True);d.eval().requires_grad_(False);del s
 for panel in panels:a.panel=panel;run_panel(a,device,d,item)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs'/f'validation_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
