"""Fact-only tables, dataset-specific three-point figures and lossless media."""
from util import *
import math,statistics,collections
import numpy as np

def numbers(rows):
 out=[]
 for r in rows:
  row={}
  for k,v in r.items():
   try:row[k]=float(v) if v!='' else None
   except (TypeError,ValueError):row[k]=v
  out.append(row)
 return out

def aggregate(rows,keys):
 buckets={}
 for r in rows:buckets.setdefault(tuple(r[k] for k in keys),[]).append(r)
 out=[]
 for key,rs in buckets.items():
  d=dict(zip(keys,key));d['source_count']=len(rs)
  for k in set().union(*(r.keys() for r in rs)):
   vals=[r[k] for r in rs if isinstance(r.get(k),(int,float))]
   if k not in d and vals:d[k]=statistics.mean(vals)
  out.append(d)
 return out

def plots(rows):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 groups=['Original','B300','S_current','U_unaligned','T_aligned'];colors=['#666666','#1f77b4','#d58500','#9467bd','#d62728'];made=[]
 rd=[r for r in rows if r['label'] in ['Original','B300'] or r['step']==300]
 for dataset in sorted(set(r['dataset'] for r in rd)):
  rs=[r for r in rd if r['dataset']==dataset]
  expected=None
  for label in groups:
   for qp in [0,5,9]:
    ids={r['id'] for r in rs if r['label']==label and r['qp']==qp}
    if expected is None:expected=ids
    assert ids==expected,('RD_SOURCE_MISMATCH',dataset,label,qp)
  for metric in ['PSNR','LPIPS','DISTS','FloLPIPS']:
   fig,ax=plt.subplots(figsize=(6,4))
   for label,color in zip(groups,colors):
    pts=[]
    for qp in [0,5,9]:
     rr=[r for r in rs if r['label']==label and r['qp']==qp];pts.append((statistics.mean(r['bpp'] for r in rr),statistics.mean(r[metric] for r in rr)))
    pts.sort();ax.plot([p[0] for p in pts],[p[1] for p in pts],marker='o',color=color,label=label+(' (300 updates)' if label not in ['Original','B300'] else ''))
   ax.set(xlabel='Actual bitstream bpp',ylabel=metric,title=f'{dataset}: {len(expected)} fixed development sources, QP 0/5/9');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();p=ROOT/'figures'/f'{dataset}_RD_{metric}.png';fig.savefig(p,dpi=180);fig.savefig(p.with_suffix('.svg'));plt.close(fig);made.append(str(p))
 for metric in ['LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel']:
  fig,ax=plt.subplots(figsize=(6,4));dev=[r for r in rows if r['dataset']=='ulong' and r['qp']==5]
  for label,color in zip(groups[2:],colors[2:]):
   pts=[]
   for step in [0,50,100,200,300]:
    rr=[r for r in dev if r['label']==label and r['step']==step];assert len(rr)==4;pts.append(statistics.mean(r[metric] for r in rr))
   ax.plot([0,50,100,200,300],pts,'o-',label=label,color=color)
  b=statistics.mean(r[metric] for r in dev if r['label']=='B300');ax.axhline(b,color=colors[1],linestyle='--',label='B300');ax.set(xlabel='Committed joint updates',ylabel=metric,title='ULong fixed four-source development mean, QP5');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();p=ROOT/'figures'/f'ULong_trajectory_{metric}.png';fig.savefig(p,dpi=180);plt.close(fig);made.append(str(p))
 return made

def ffv1(path,frames,fps):
 t,h,w,c=frames.shape;assert c==3 and frames.dtype==np.uint8;path.parent.mkdir(exist_ok=True,parents=True)
 raw=frames.tobytes();temp=path.with_name(path.stem+'.part.mkv')
 cmd=['ffmpeg','-y','-v','error','-threads','2','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',fps,'-i','pipe:0','-an','-c:v','ffv1','-level','3','-g','1','-pix_fmt','bgr0','-color_range','pc','-colorspace','rgb','-threads','2',str(temp)]
 subprocess.run(cmd,input=raw,check=True);temp.replace(path)
 decoded=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
 assert decoded==raw,('FFV1_NOT_LOSSLESS',str(path))
 mp4=path.with_suffix('.mp4');tmp=mp4.with_name(mp4.stem+'.part.mp4');subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(path),'-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-threads','2',str(tmp)],check=True);tmp.replace(mp4)
 return dict(path=str(path),mp4=str(mp4),sha256=sha(path),mp4_sha256=sha(mp4),bytes=path.stat().st_size,frames=t,width=w,height=h,fps=fps,codec='FFV1 bgr0 RGB8 lossless, full range',RGB8_roundtrip_exact=True,RGB8_sha256=hashlib.sha256(raw).hexdigest())

def videos():
 from PIL import Image,ImageDraw,ImageFont
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 rows=[];png=[];models=[('Original',0,'Original'),('B300',0,'B300'),('S_current',300,'S300'),('U_unaligned',300,'U300'),('T_aligned',300,'T300'),('T_aligned',100,'T100')]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 for row in [manifest()['dev'][k] for k in [0,2]]:
  row=dict(row);row['source_frame_indices']=list(range(row['start'],row['start']+64))
  id=row['id'];d=waitcache(id+'_video64','delivery',5);source=d['GT'].numpy();allframes={'GT':source};fps=row['source_fps'];x,y,rw,rh=row['ROI']
  for label,step,short in models:
   p=ROOT/'cache/media64'/f'{label}_{step:04d}_{id}_q5.pt';rec=readtensor(p);allframes[short]=rec['frames'].permute(0,2,3,1).numpy()
  for short,arr in allframes.items():
   path=ROOT/'videos'/id/f'{short}_native.mkv';r=ffv1(path,arr,fps);r.update(model=short,id=id,source_frames=list(range(row['start'],row['start']+64)),category=row['category']);rows.append(r);event('VIDEO_COMMITTED',path=str(path),lossless_verified=True)
   for i in [1,32,63]:
    p=ROOT/'figures/fixed_frames'/id/f'{short}_frame{i:03d}_source{row["source_frame_indices"][i]:06d}.png';p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arr[i]).save(p);png.append(str(p))
  panel=[]
  for i in range(64):
   out=Image.new('RGB',(rw*len(allframes),rh+48),(20,20,20));draw=ImageDraw.Draw(out)
   for k,(short,arr) in enumerate(allframes.items()):draw.text((k*rw+8,12),short,font=font,fill='white');out.paste(Image.fromarray(arr[i,y:y+rh,x:x+rw]),(k*rw,48))
   panel.append(np.asarray(out))
  r=ffv1(ROOT/'videos'/id/'GT_Original_B300_S300_U300_T300_T100_native_ROI_panels.mkv',np.stack(panel),fps);r.update(model='GT | Original | B300 | S300 | U300 | T300 | T100',id=id,ROI=row['ROI'],ROI_scale=1,title_band_extra_pixels=48);rows.append(r)
  for step in [100,300]:
   heat=readtensor(ROOT/'cache/heatmaps64'/f'T_aligned_{step:04d}_{id}_q5.pt')
   for a in heat['frames']:
    if a['frame'] not in [1,32,63]:continue
    vals=[(a['q'][0,0].numpy(),'q',0,1),(a['u'][0].square().sum(0).sqrt().numpy(),'displacement_grid8',0,8),(a['correction'][0,0].numpy(),'gated_latent_correction_RMS',0,.05)]
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,(z,name,vmin,vmax) in zip(axes,vals):im=ax.imshow(z,cmap='viridis',vmin=vmin,vmax=vmax);ax.set_title(name);ax.axis('off');fig.colorbar(im,ax=ax,fraction=.045)
    fig.suptitle(f'{id}, T{step}, local frame {a["frame"]}; fixed display scales');fig.tight_layout();p=ROOT/'figures'/f'{id}_T{step}_frame{a["frame"]:03d}_mechanism.png';fig.savefig(p,dpi=150);plt.close(fig);png.append(str(p))
 dump(ROOT/'audits/videos.json',dict(status='PASS',videos=rows,fixed_frame_png=png,metrics_before_RGB8_quantization=True,video_content='native frame dimensions and fps; ROI panels keep pixel scale with separate title band'))
 text=['# Video index','',f'Experiment: `{ROOT}`','', 'Fixed cases are predeclared development diagnostics, not independent final tests. Each is the complete 64-frame existing development clip at its original resolution and source frame rate. MP4 is a lossy playback copy; FFV1 RGB8 was decoded and compared byte-for-byte.','', 'Model legend: GT; independent Original GVC-RT; frozen B300; S300 (current-only added branch); U300 (real history and matching evidence, zero latent displacement); T300 (predicted alignment); T100 (fixed trajectory checkpoint, not selected model).','',f'B300 composite parameter SHA256: `{load(ROOT/"audits/extended_checks.json")["final_composite_parameter_SHA256"]}`','']
 for label,step,short in models:
  if label not in ['Original','B300']:text.append(f'{short}: `{ROOT/"checkpoints"/label/f"step_{step:04d}.pt"}`, SHA256 `{sha(ROOT/"checkpoints"/label/f"step_{step:04d}.pt")}`.');text.append('')
 for r in rows:text.extend([f'- {r["id"]} / {r["model"]}: `{r["path"]}`; playback `{r["mp4"]}`; {r["width"]}×{r["height"]}, {r["fps"]} fps, {r["frames"]} frames.',''])
 text+=['Fixed frame PNGs and mechanism maps: `figures/fixed_frames/` and `figures/*_mechanism.png`.','', 'No additional historically repeated three-clip montage was presented as an independent final test. All large media remain local; the path/size/SHA256 list is in `large_artifacts.json`.']
 (ROOT/'VIDEO_INDEX.md').write_text('\n'.join(text)+'\n');return rows,png

def report(rows,paired,figures):
 profile=load(ROOT/'audits/runtime_profile.json');checks=load(ROOT/'audits/extended_checks.json');p=load(ROOT/'protocol.json');cal=load(ROOT/'audits/calibration.json');coverage=manifest()['coverage'];agg=aggregate(rows,['dataset','label','step','qp']);csvwrite(ROOT/'metrics/aggregate_metrics.csv',agg)
 def fmt(v):return f'{v:.8g}' if isinstance(v,(int,float)) else str(v)
 lines=['# Bridge aligned latent pilot: factual execution record','', 'Completed one paired seed (20261112): 500 matcher pretraining updates and 300 committed updates each for S/U/T. No full-dataset training, second seed, architecture/loss search, convergence claim, efficacy claim or test135 run.','',f'B300 composition: Original → pilot → B1500 TCG → A2000 LoRA → B300 last (52 incremental keys). LoRA gain 0.5. Composite parameter SHA256 `{checks["final_composite_parameter_SHA256"]}`. All codec/entropy/Bridge/de-tokenizer/TCG/LoRA parameters frozen; per-group final hashes are in `checkpoints/*/freeze_audit.json`.','',f'Training coverage: {coverage["source_groups"]} distinct approved ULong source groups, {coverage["unique_temporal_windows"]} unique nonoverlapping 16-frame temporal windows, {coverage["unique_spatial_crops"]} fixed original-RGB 256×256 spatial crops, {coverage["qp_instances"]} QP instances (external0/5/9, original actual-QP schedule retained). Original frames were cropped before frozen codec/Bridge; no whole-frame latent crop equivalence claim.','', 'Training consumes 1200 fixed 4-frame segments per group (4800 supervised frame occurrences), reusing 384 fixed window×QP instances. Per-instance/per-window repeat counts and exact spatial/temporal identities are in `metrics/training_coverage.csv` and `training_plan.json`. Four segments are accumulated equally per update. Candidate-specific causal warmup is replayed from the window I-frame reset, TCG gradients are kept inside each segment, and all state is detached between segments.','', 'Only the new aligner (U/T), reliability head and residual adapter train. S retains B300 TCG and has the same fusion capacity; S has no new historical content, U retains matching information while forcing latent warp displacement to zero. Parameter counts: '+json.dumps(profile['new_parameter_counts'])+'. The matcher/RMS/QP statistics and optimizer settings are fixed in protocol.json.','',f'Loss: LPIPS_Alex(normalize=True)+0.5 DISTS+0.1383589248954108 RGB_L1 + {cal["lambda_temp"]:.12g}*(GT-flow-masked motion-compensated RGB change error +0.5 signed /8 Sobel change error); U/T additionally use0.1 multiscale pseudo-flow Charbonnier. This is the verified existing baseline spatial objective, not a claim of an unchanged original-paper loss. Lambda is0.1 times the median reconstructed-RGB output gradient ratio on eight fixed train segments. GT RAFT flow is a pseudo-label, never inference input.','', 'All candidate and B300 reconstruction/metrics/new modules use FP32 PyTorch reference operators; codec keeps audited FP16. Decoder and perceptual parameters are frozen but their input gradients are preserved. Zero-output branches matched B300 bitwise across4 synthetic and7 real consecutive P frames. Correlation direction, fractional sampling, displacement units, all-invalid complete fallback, original-only history cache and cross-frame TCG gradients passed. Candidate reconstruction leaves codec DPB untouched. Grid-sample CUDA backward uses deterministic warn_only; seed pairing does not imply bitwise deterministic backward.','', 'Evaluation scope: four predeclared ULong development clips, each16 consecutive frames from a reset I frame, full native1920×1080 at source fps; QP5 at0/50/100/200/300. At300, externalQP0/5/9 additionally cover the same four ULong sources and one preexisting UVG-validation Bosphorus source (native120fps). These are short clips, not full original videos. No verified legal HEVC/MCL development split was available without using reserved/final lists; those curves are absent. Selection used existing GT content categories, never candidate errors; existing development use does not make these an independent final test.','', '|Dataset|Model|step|QP|sources|PSNR|LPIPS|DISTS|FloLPIPS|MC RGB|MC Sobel|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in sorted(agg,key=lambda r:(r['dataset'],r['qp'],r['label'],r['step'])):
  if r['qp']==5:lines.append('|'+ '|'.join(fmt(r[k]) for k in ['dataset','label','step','qp','source_count','PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel'])+'|')
 lines+=['','All unrounded per-video/step/QP values, per-frame metrics and signed paired differences are in metrics/. Dataset R-D curves use identical sequence sets within each dataset and preserve Original/B300; the three observed QPs are connected without extrapolation. No cross-dataset averaged curve or full-rate-range conclusion is made.','', '|ULong QP5, step300|ΔPSNR vs B300|ΔLPIPS|ΔDISTS|ΔFloLPIPS|ΔMC RGB|ΔMC Sobel|','|---|---:|---:|---:|---:|---:|---:|']
 for g in ['S_current','U_unaligned','T_aligned']:
  rs=[r for r in paired if r['dataset']=='ulong' and r['qp']==5 and r['step']==300 and r['label']==g and r['reference']=='B300'];lines.append('|'+g+'|'+'|'.join(fmt(statistics.mean(r['delta_'+k] for r in rs)) for k in ['PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel'])+'|')
 lines+=['','Matcher pretraining is counted separately. Pseudo-label EPE samples are fixed four train pairs and one pair per ULong dev source at local frame8; they are not dense flow ground truth or full-clip average EPE. Raw0/100/250/500 values are in metrics/aligner_*.csv.','', '|Matcher step|train EPE8|dev EPE8|','|---:|---:|---:|']
 for step in [0,100,250,500]:
  rr=numbers(list(csv.DictReader((ROOT/'metrics'/f'aligner_{step:04d}.csv').open())));lines.append(f'|{step}|'+ '|'.join(fmt(statistics.mean(r['EPE_8'] for r in rr if r['split']==split)) for split in ['train','dev'])+'|')
 lines+=['', 'Mechanism statistics (q quantiles/near0/1, displacement, valid history fraction, delta RMS and per-channel latent mean/std drift) are observations, not calibrated reliability or proof of useful history. Invalid/newly exposed-region error uses the frozen teacher consistency mask as a proxy, not true occlusion labels. FloLPIPS uses the audited official PWC+Alex implementation; MC residuals use the fixed GT-RAFT protocol.','', f'Extra persistent state at this native padded shape: {profile["new_state_bytes_FP32"]} bytes for one raw f and one uncorrected Bridge latent plus metadata. No corrected latent is stored as future history. Runtime breakdown and actual RAM-bitstream end-to-end decode definition are in audits/runtime_profile.json.','', '|Pipeline|codec ms/frame|external complete reconstruction ms/frame|end-to-end ms/frame|','|---|---:|---:|---:|']
 for k,r in profile['actual_bitstream_pipeline'].items():lines.append(f'|{k}|{r["codec_ms_per_frame"]:.6g}|{r["external_reconstruction_ms_per_frame"]:.6g}|{r["end_to_end_ms_per_frame"]:.6g}|')
 lines+=['', 'End-to-end timing includes RAM bitstream parsing, entropy/feature decoding, Bridge, candidate and complete de-tokenizer (batch1), including I-frame handling. P-frame codec DPB stores original features with frame=None; bitwise feature identity with independent Original decode was checked. Candidate RGB/latent never enters DPB. Component branch latency alone is not decode speed; this FP32 reference is not an optimized deployment speed claim.','', 'Timing is separated: cache construction and teacher seconds are recorded per cache; training update seconds exclude cache generation/checkpoint serialization/diagnostics/evaluation; evaluation has its own receipts. Per-group updates and measured durations are in metrics/execution_times.json.','', 'Implementation exceptions: initial Original snapshot omitted protocol-verified C++ sources (stopped with0 caches); inherited restore reset deterministic warn_only (stopped with0 optimizer updates); a coarse mask had no valid positions at attempted pretrain update50 (49 committed, resumed after safe empty-set reduction). Full all-invalid fallback metadata was completed before joint update0. These are logged in audits/implementation_corrections.json; no quality-driven architecture/loss change was made.','', 'Limitations: one seed,32 training source groups,128 fixed windows,16-frame development clips, only three QPs for the final display, one legal UVG validation source, and no HEVC/MCL development curves. The two visual diagnostics independently roll through64 frames; quantitative checkpoint results remain16-frame pilot clips.300 updates is a cap, not convergence. No final135 test, human temporal blind study or significance test was performed.','', 'Locations: protocol.json; source_manifest.json; data_manifest.json; training_plan.json; additional_development_manifest.json; audits/; metrics/; figures/; VIDEO_INDEX.md; REPRODUCE.md. Large caches/weights/videos are local and listed in large_artifacts.json; ordinary Git upload excludes them.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
 assert load(ROOT/'receipts/evaluation.complete.json')['points']==len(__import__('evaluate').jobs())
 rows=numbers(list(csv.DictReader((ROOT/'metrics/per_video_metrics.csv').open())));paired=[];metrics=['PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel','edge_strength','high_frequency_error_MSE','invalid_RGB_L1']
 for r in rows:
  if r['label'] in ['Original','B300']:continue
  refs=['B300']+(['S_current'] if r['label'] in ['U_unaligned','T_aligned'] else [])+(['U_unaligned'] if r['label']=='T_aligned' else [])
  for ref in refs:
   matches=[v for v in rows if v['id']==r['id'] and v['qp']==r['qp'] and v['label']==ref and (ref=='B300' or v['step']==r['step'])];assert len(matches)==1
   b=matches[0];paired.append({k:r[k] for k in ['dataset','id','qp','label','step']}|dict(reference=ref)|{'delta_'+k:r[k]-b[k] for k in metrics if r.get(k) is not None and b.get(k) is not None})
 csvwrite(ROOT/'metrics/paired_differences.csv',paired)
 coverage=collections.Counter(s['id'] for b in plan()['joint'] for s in b);csvwrite(ROOT/'metrics/training_coverage.csv',[dict(id=r['id'],source_group=r['source_group'],start=r['start'],crop=json.dumps(r['crop']),external_qp=r['external_qp'],segment_repeats_per_group=coverage[r['id']],source_frame_indices=json.dumps(r['source_frame_indices'])) for r in manifest()['train']])
 timings={}
 for g in ['S_current','U_unaligned','T_aligned']:timings[g]=load(ROOT/'checkpoints'/g/'freeze_audit.json')
 timings['aligner']=load(ROOT/'audits/aligner_frozen_reference.json');timings['cache_events']=[json.loads(s) for s in (ROOT/'logs/events.jsonl').read_text().splitlines() if json.loads(s)['event']=='CACHE_COMMITTED'];dump(ROOT/'metrics/execution_times.json',timings)
 figs=plots(rows);media,png=videos();report(rows,paired,figs)
 artifacts=[]
 for base in ['cache','checkpoints','videos']:
  for p in (ROOT/base).rglob('*'):
   if p.is_file() and p.suffix in ['.pt','.bin','.mkv','.mp4']:
    proof=p.with_suffix(p.suffix+'.json');r=load(proof) if proof.exists() else {};assert not r or p.stat().st_size==r['bytes'];artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=r.get('sha256') or sha(p),uploaded=False,hash_source='atomic receipt' if r else 'file',reproduce='REPRODUCE.md and source/plan/protocol manifests'))
 dump(ROOT/'large_artifacts.json',dict(large_artifacts_uploaded=False,artifacts=artifacts))
 dump(ROOT/'receipts/local_delivery.complete.json',dict(status='LOCAL_COMPLETE',time=time.time(),figures=figs,fixed_png_count=len(png),lossless_videos=len(media),evaluation_points=len(rows),report=str(ROOT/'REPORT.md')))
 event('LOCAL_DELIVERY_COMPLETE',evaluation_points=len(rows),videos=len(media))

if __name__=='__main__':main()
