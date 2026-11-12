from util import *
import argparse

def video_rows():
 out=[]
 for r in [manifest()['dev'][k] for k in [0,2]]:
  v=dict(r);v.update(id=r['id']+'_video64',frames=64,external_qp=5,source_frame_indices=list(range(r['start'],r['start']+64)),metric_source_id=r['id']);out.append(v)
 return out

def cache():
 from cache_data import make
 rows=video_rows();dump(ROOT/'video_manifest.json',dict(rows=rows,reason='64-frame existing development clip for original-fps visual inspection; two cases fixed before quality results, no selection',fast_quality_metrics_frames=16,visual_frames=64))
 with lock('media64_cache.lock'):make('media64_cache',rows)

def render():
 import torch,gc
 from restore import restore_b300
 from model import AlignedLatent,bridge,decode
 setup();seed();c=restore_b300();stats=readtensor(ROOT/'cache/statistics.pt');models=[('Original',0,'Original'),('B300',0,'B300'),('S_current',300,'S300'),('U_unaligned',300,'U300'),('T_aligned',300,'T300'),('T_aligned',100,'T100')]
 for row in video_rows():
  d=waitcache(row['id'],'media64_render',5);h,w=d['native_shape'];id=row['metric_source_id']
  for label,step,short in models:
   dest=ROOT/'cache/media64'/f'{label}_{step:04d}_{id}_q5.pt'
   if dest.with_suffix('.pt.json').exists():continue
   m=None
   if label not in ['Original','B300']:
    cp=ROOT/'checkpoints'/label/f'step_{step:04d}.pt'
    while not cp.with_suffix('.pt.json').exists():heartbeat('media64_render','WAITING_FOR_CHECKPOINT',checkpoint=str(cp));time.sleep(3)
    m=AlignedLatent(stats,label).cuda();m.load_state_dict(readtensor(cp)['model']);m.eval();m.reset()
   c['models']['generator'].reset_temporal_buffer();frames=[];heats=[]
   with torch.no_grad():
    for i in range(64):
     heartbeat('media64_render','RENDERING_64_NATIVE',model=short,id=id,frame=i)
     if i==0 or label=='Original':y=d['original'][i].cuda();a=None
     else:
      f=d['features'][i].cuda().float();l=bridge(c,f);lp,a=m(l,f,d['actual_qps'][i],id,i) if m is not None else (l,None);y=decode(c,lp,d['actual_qps'][i])[...,:h,:w]
     frames.append(y.mul(255).round().clamp(0,255).byte().cpu())
     if label=='T_aligned' and i in [1,32,63]:heats.append(dict(frame=i,q=a['q'].cpu(),u=a['u'].cpu(),correction=(a['q']*a['delta']).square().mean(1,keepdim=True).sqrt().cpu()))
   save(dest,dict(frames=torch.cat(frames),model=short,source_frames=row['source_frame_indices'],fps=row['source_fps'],stream_sha256=d['stream_sha256'],independent_causal_roll=True))
   if heats:save(ROOT/'cache/heatmaps64'/f'T_aligned_{step:04d}_{id}_q5.pt',dict(frames=heats))
   del frames,m;gc.collect();torch.cuda.empty_cache()
 dump(ROOT/'receipts/media64.complete.json',dict(status='COMPLETE',frames_per_case=64,cases=[r['metric_source_id'] for r in video_rows()],time=time.time()))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['cache','render']);p=a.parse_args()
 if p.phase=='cache':cache()
 else:render()
