"""Complete only missing beta1 video assets; numerical evaluation is preserved."""
from util import *
import argparse,csv,torch,gc
from model import DeformableHistory,bridge,decode
from reference import coexist_original
from stream import raw_frames,stream_file,decode_frames,ref_sidecar
from evaluate import VideoSink,tag,proofpath

def gt_video(row,arrays):
 meta=ROOT/'audits/video_assets'/f"{row['id']}_GT.json";dest=ROOT/'videos'/row['id']/'GT_native.mkv'
 with lock('GT_video_'+row['id']+'.lock',True):
  if meta.exists():return load(meta)
  sink=VideoSink(dest,row['height'],row['width'],row['source_fps'])
  for a in arrays:
   raw=a.tobytes();sink.p.stdin.write(raw);sink.hash.update(raw);sink.frames+=1
  v=sink.close();v.update(label='GT',id=row['id'],fps=row['source_fps'],source_frames=row['source_frame_indices'],width=row['width'],height=row['height']);dump(meta,v);return v

def run(shard):
 setup();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);rt,im,pm=coexist_original();st=stats();fixed=load(ROOT/'evaluation/fixed_points.json')['fixed_points'];selected=[devrows()[k] for k in [0,2]];jobs=[]
 for row in selected:
  for step in fixed:
   for label in ['A_zero_base','B_flow_guided']:jobs.append(dict(id=row['id'],step=step,label=label,qp=5,beta=1.))
 for j in jobs[shard::2]:
  p=proofpath(j);proof=load(p)
  if proof.get('video') and Path(proof['video']['path']).exists():continue
  row=next(r for r in selected if r['id']==j['id']);arrays=raw_frames(row);gt_video(row,arrays);stream,meta=stream_file(rt,im,pm,row,5,arrays);assert sha(stream)==proof['stream_sha256'];m=DeformableHistory(st,j['label']).cuda();cp=ROOT/'checkpoints'/j['label']/f"step_{j['step']:04d}.pt";ck=readtensor(cp);assert sha(cp)==proof['checkpoint_sha256'];m.load_state_dict(ck['model']);m.eval();m.reset();g=c['models']['generator'];g.reset_temporal_buffer();h,w=row['height'],row['width'];sink=VideoSink(ROOT/'videos'/row['id']/f'{tag(j)}.mkv',h,w,row['source_fps']);prev=None;frames=list(csv.DictReader((ROOT/'metrics/frames'/f'{tag(j)}.csv').open()));comparisons=[]
  with torch.no_grad():
   for i,q,f,ref,sps,_ in decode_frames(rt,im,pm,stream,meta):
    heartbeat('render'+str(shard),'RENDERING_MISSING_VIDEO',job=j,frame=i)
    if i==0:y=ref[...,:h,:w]
    else:
     l=bridge(c,f.float());back,e=ref_sidecar(row,5,i,ref,prev,None,None);lp,_=m(l,back,e,q,row['id'],i,1.);y=decode(c,lp,q)[...,:h,:w]
    if i in [1,32,63]:
     gt=torch.from_numpy(arrays[i]).permute(2,0,1)[None].cuda().float()/255;mse=float((y-gt).square().mean());err=abs(mse-float(frames[i]['RGB_MSE']));assert err<=1e-12,('VIDEO_EVALUATION_MISMATCH',j,i,mse,frames[i]['RGB_MSE']);comparisons.append(dict(frame=i,MSE_difference=err))
    sink.write(y);prev=ref.detach()
  video=sink.close();video.update(fps=row['source_fps'],width=w,height=h,source_frames=row['source_frame_indices'],sparse_float_MSE_matches_evaluation=comparisons);proof['video']=video;dump(p,proof);event('MISSING_VIDEO_COMPLETED',job=j,path=video['path']);del m,arrays;gc.collect();torch.cuda.empty_cache()
 heartbeat('render'+str(shard),'COMPLETE')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);p=a.parse_args()
 with lock('render'+str(p.shard)+'.lock'):run(p.shard)
