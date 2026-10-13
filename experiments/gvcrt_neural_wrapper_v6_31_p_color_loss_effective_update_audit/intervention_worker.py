"""Cross sender/receiver cells; 54 audited aliases and 81 new real RANS points."""
from audit31 import *
import argparse,fcntl,resource,traceback
def main(t,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from adapter31 import backend,runtime,frames_for,decode
 from color31 import color,residual
 settings=backend();torch.cuda.reset_peak_memory_stats();start=time.time()
 lf=receipt(t).with_suffix('.lock').open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if valid(t):return
 v=video(t['dataset'],t['video']);cell=t['method'];art={};frames=frames_for(v)
 if cell in ('R00','R11'):
  m='original' if cell=='R00' else 'B1_initial';ct=dict(t,stage='color',method=m);assert valid(ct),('color prerequisite',uid(ct))
  cr=load(receipt(ct));oldp=oldpoint(t['dataset'],m,t['video'],t['qp']);r=load(oldp)
  assert cr['decode']['reconstruction_sha256']==r['reconstruction_sha256'] and cr['decode']['bytes_consumed']==r['real_bytes']
  assert cr['decode']['actual_qps']==r['actual_qps'];assert sha(r['bitstream_path'])==r['bitstream_sha256']
  dump(receipt(t),dict(status='PASS',task=t,signature=signature(t),artifacts={str(oldp):sha(oldp),r['bitstream_path']:r['bitstream_sha256'],r['feature_path']:r['feature_sha256']},result=r,reused=True,reuse_from=str(oldp),reuse_sha256=sha(oldp),color_decode_receipt=str(receipt(ct)),color_decode_sha256=sha(receipt(ct)),physical_GPU=gpu,peak_memory_MiB=1.,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024));print('REUSED',uid(t),flush=True);return
 eng,rt=runtime(cell);before=[tensor_hash(m.state_dict()) for m in (*rt.quality,*rt.metric_models)];wh=None if rt.wrapper is None else eng.core.module_hash(rt.wrapper)
 b=ROOT/'bitstreams'/t['dataset']/cell/f'video{t["video"]:02d}_qp{t["qp"]}.bin';f=ROOT/'features/intervention'/(uid(t)+'.npz')
 r,fr=rt.run(frames,t['qp'],b,f)
 assert r['frames']==64 and r['num_transitions']==63 and r['independent_decode_pass'] and r['state_sync_pass'] and r['bytes_consumed']==b.stat().st_size
 assert r['actual_qps'] if 'actual_qps' in r else True
 actual=[x['actual_qp'] for x in fr];assert actual==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(t['qp'])][:64]
 r.update(dataset=t['dataset'],video=t['video'],sequence=v['name'],cell=cell,QP=t['qp'],actual_qps=actual,native_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,kbps_20fps=r['bits_per_frame']*20/1000,rate_accounting_fps=v['rate_accounting_fps'],input_rgb_sha256=v['rgb_sha256'])
 colors=[];clips=[]
 for i,x,_ in decode(eng,rt,b,dict(r,reconstruction_sha256=r['reconstruction_sha256'])):
  gt=frames[i].cuda();colors.append(dict(dataset=t['dataset'],video=t['video'],cell=cell,QP=t['qp'],frame=i,**color(x,gt)))
  if cell=='R11_center':
   px,c=residual(rt.wrapper,gt,center=True);assert torch.equal(px,rt.wrapper(gt))
   clips.append(dict(frame=i,used_for_encoding=i>0,**c))
 assert before==[tensor_hash(m.state_dict()) for m in (*rt.quality,*rt.metric_models)] and wh==(None if rt.wrapper is None else eng.core.module_hash(rt.wrapper))
 base=ROOT/'raw/intervention'/uid(t)
 for suffix,data in (('.frames.csv',fr),('.color.csv',colors)):
  p=base.with_suffix(suffix);write(p,data);art[str(p)]=sha(p)
 if clips:
  p=base.with_suffix('.P.csv');write(p,clips);art[str(p)]=sha(p)
 for p in (b,f,f.with_suffix('.transitions.csv')):art[str(p)]=sha(p)
 dump(receipt(t),dict(status='PASS',task=t,signature=signature(t),artifacts=art,result=r,reused=False,decode=rt.decode_audit,wrapper_hash=wh,frozen_metric_hashes=before,backend=settings,physical_GPU=gpu,seconds=time.time()-start,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024));print('DONE',uid(t),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
 try:main(json.loads(a.task),a.gpu)
 except Exception:error(dict(task=json.loads(a.task),traceback=traceback.format_exc()));raise
