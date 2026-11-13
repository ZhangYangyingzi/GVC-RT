"""Explicit validation-only15-frame evaluation with independent suffix rollout."""
from util import *
import torch,statistics,gc
from objective import highpass,quality_models
from temporal import temporal_errors,sobel
from data import Data,cache_dir
from model import load_cref,load_native,G11,Receiver
class VideoSink:
 def __init__(self,path,h,w,fps):
  self.path=path;path.parent.mkdir(parents=True,exist_ok=True);self.tmp=path.with_suffix('.part.mkv');self.hash=hashlib.sha256();self.frames=0;self.proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{w}x{h}','-framerate',fps,'-i','pipe:0','-c:v','ffv1','-level','3','-g','1','-pix_fmt','bgr0','-color_range','pc','-colorspace','rgb','-threads','2',str(self.tmp)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
 def write(self,x):
  raw=x[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy().tobytes();self.proc.stdin.write(raw);self.hash.update(raw);self.frames+=1
 def close(self):
  self.proc.stdin.close();error=self.proc.stderr.read();assert self.proc.wait()==0,error.decode();self.tmp.replace(self.path);return dict(path=str(self.path),bytes=self.path.stat().st_size,sha256=sha(self.path),RGB8_sha256=self.hash.hexdigest(),frames=self.frames)

def fps(row):
 if row.get('source_fps'):return row['source_fps']
 p=ROOT/'sources/validation_source_recovery.json'
 if p.exists():
  match=next((r for r in load(p)['sources'] if r['sequence']==row['sequence']),None)
  if match:return match['fps']
 raise RuntimeError(('VALIDATION_FPS_NOT_VERIFIED',row['sequence']))
class Evaluator:
 def __init__(self,quality=None):
  self.q=quality or quality_models();sys.path.insert(0,str(ROOT/'vendor/flo'));import flolpips_compat
  self.pwc,self.flo=flolpips_compat.load_models(torch.device('cuda:0'));self.metric_hashes={k:thash(v.state_dict().items()) for k,v in [('LPIPS',self.q[0]),('DISTS',self.q[1]),('PWC',self.pwc),('FloLPIPS',self.flo)]};self.original=None
 @torch.no_grad()
 def point(self,label,step,row,data,branch=None,checkpoint_sha=None,media=True):
  assert row['split']=='validation' and row['sequence'] in {r['sequence'] for r in videos('validation')},'FINAL_TEST_FORBIDDEN';seq=row['sequence'];dest=ROOT/'metrics/points'/f'{label}_s{step:04d}_{seq}.json'
  if dest.exists():
   d=load(dest);assert d['protocol_sha256']==protocol_sha() and d['checkpoint_sha256']==checkpoint_sha;return d['metrics']
  wait_file(cache_dir(seq)/'completion.json','WAITING_FOR_EVAL_CACHE');src=data.source(seq);h,w=src.info['native_shape'];state=None;previous=prevgt=None;frames=[];pairs=[];sinks={};t=time.perf_counter()
  if media:
   rate=fps(row);sinks[label]=VideoSink(ROOT/'videos'/seq/f'{label}_s{step:04d}_frames49-63.mkv',h,w,rate)
   if label=='C_ref':sinks['GT']=VideoSink(ROOT/'videos'/seq/'GT_frames49-63.mkv',h,w,rate)
  if label=='Original' and self.original is None:
   from original_reference import build_original
   self.original=build_original()
  for i in range(47,64):
   heartbeat('VALIDATION',label=label,validation_step=step,video=seq,frame=i)
   d=data.tensors(seq,i);aux=None
   if d['meta']['scene_cut']:state=None;previous=prevgt=None
   if label=='Original':
    f=src.feature(i).cuda().float();q=src.info['actual_qps'][i];pm=self.original[2];out=(pm.recon_generation_net(f,pm.q_scale_recon[q:q+1].float()).clamp(-1,1)+1)/2
   else:
    hh=d['h_raw'] if label=='B300' else d['h_s']
    if label=='G11':hh,aux=branch(hh,d['current_c'],d['previous_c'],d['meta'])
    out,state=data.receiver.suffix(hh,d['x'],d['q'],state,False)
   if i<49:continue
   gt=data.gt(seq,i);out=out[...,:h,:w];err=out-gt;hp,hg=highpass(out),highpass(gt);mse=float(err.square().mean());r=dict(label=label,step=step,sequence=seq,dataset=row['domain'],frame=i,source_frame=row['source_frame_indices'][i],actual_QP=src.info['actual_qps'][i],bytes=src.info['real_bytes'],bpp=src.info['bpp'],PSNR=-10*math.log10(max(mse,1e-15)),RGB_MSE=mse,RGB_L1=float(err.abs().mean()),LPIPS=float(self.q[0](out,gt,normalize=True).mean()),DISTS=float(self.q[1](out,gt,require_grad=False).mean()),highpass_signed_error_MSE=float((hp-hg).square().mean()),signed_Sobel_L1=float((sobel(out)-sobel(gt)).abs().mean()),highpass_energy=float(hp.square().mean()),GT_highpass_energy=float(hg.square().mean()),highpass_cosine=float((hp*hg).mean()/(hp.square().mean().sqrt()*hg.square().mean().sqrt()+1e-12)))
   if aux is not None:r.update(offset_RMS=float(aux['offset'].square().mean().sqrt()),offset_max_abs=float(aux['offset'].abs().max()),modulation_mean=float(aux['modulation'].mean()),point_valid=float(aux['point_validity'].float().mean()),v_mean=float(aux['v'].mean()),delta_h_RMS=float(aux['delta'].square().mean().sqrt()),correction_RMS=float(aux['correction'].square().mean().sqrt()))
   if previous is not None:
    flow,valid=data.supervision(seq,i);a,b,v=temporal_errors(out,previous,gt,prevgt,flow,valid);fd=self.pwc(prevgt,gt)-self.pwc(previous,out);fr=float(self.flo(prevgt,previous,fd,normalize=True).mean());pairs.append(dict(label=label,step=step,sequence=seq,dataset=row['domain'],from_frame=i-1,to_frame=i,FloLPIPS=fr,MC_RGB=float(a) if a is not None else None,MC_Sobel=float(b) if b is not None else None,valid_fraction=v,valid_pixels=int(valid.sum())));r.update({k:pairs[-1][k] for k in ['FloLPIPS','MC_RGB','MC_Sobel','valid_fraction']})
   frames.append(r);previous,prevgt=out,gt
   for key,sink in sinks.items():sink.write(gt if key=='GT' else out)
  assert len(frames)==15;assert len(pairs)<=14
  metrics=dict(label=label,step=step,sequence=seq,dataset=row['domain'],split='validation',frames=15,transitions=len(pairs),QP=5,bytes=src.info['real_bytes'],bpp=src.info['bpp'])
  for k in set().union(*(r.keys() for r in frames)):
   if k not in metrics and k not in ['frame','source_frame','actual_QP']:
    vs=[r[k] for r in frames if isinstance(r.get(k),(int,float))];metrics[k]=statistics.mean(vs) if vs else None
  metrics['PSNR']=-10*math.log10(metrics['RGB_MSE'])
  for k in ['FloLPIPS','MC_RGB','MC_Sobel','valid_fraction']:
   vs=[r[k] for r in pairs if r[k] is not None];metrics[k]=statistics.mean(vs) if vs else None
  assert all(not isinstance(v,float) or math.isfinite(v) for v in metrics.values());assets=[]
  for label,sink in sinks.items():assets.append(sink.close()|dict(label=label,step=0 if label=='GT' else step,sequence=seq,fps=fps(row),width=w,height=h,frames_source=list(range(49,64)),range='full RGB8',quantization='round(clamp(RGB*255,0,255))'))
  name=f"{metrics['label']}_s{step:04d}_{seq}";csv_write(ROOT/'metrics/frames'/f'{name}.csv',frames);csv_write(ROOT/'metrics/transitions'/f'{name}.csv',pairs)
  dump(dest,dict(status='PASS',protocol_sha256=protocol_sha(),checkpoint_sha256=checkpoint_sha,metrics=metrics,stream_sha256=src.info['stream_SHA256'],codec_feature_hashes=[data.tensors(seq,i)['codec_feature_sha256'] for i in range(47,64)],independent_rollout=True,metric_hashes=self.metric_hashes,videos=assets,seconds=time.perf_counter()-t,reset_pairs_excluded=14-len(pairs)))
  return metrics
 def validation(self,label,step,data,branch=None,checkpoint_sha=None):return [self.point(label,step,r,data,branch,checkpoint_sha,media=(label!='G11' or step>=100)) for r in videos('validation')]
