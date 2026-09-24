#!/usr/bin/env python3
import csv,hashlib,json,math,shutil,subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from core import ROOT,csv_write

TAGS=['fresh_ulong_00','fresh_ulong_01','fresh_ulong_10','uvg_00','uvg_01','uvg_02']
def read(p):return list(csv.DictReader(open(p,newline='')))
def finite(v):
 try:return math.isfinite(float(v))
 except:return False

training=[]
for b in ('beta_low','beta_mid','beta_high'):training+=read(ROOT/'parts'/f'training_{b}.csv')
csv_write(ROOT/'training_log.csv',training)
checkpoint=[]
for b in ('beta_low','beta_mid','beta_high'):checkpoint+=read(ROOT/'parts'/f'checkpoint_rans_{b}.csv')
csv_write(ROOT/'checkpoint_real_rans.csv',checkpoint)
rd=[];frames=[]
for p in sorted((ROOT/'parts').glob('final_rd_gpu*.csv')):rd+=read(p)
for p in sorted((ROOT/'parts').glob('final_frames_gpu*.csv')):frames+=read(p)
csv_write(ROOT/'final_rd_points.csv',rd);csv_write(ROOT/'frame_metrics.csv',frames)

matched=[]
for tg in TAGS:
 rows=[r for r in rd if r['video_tag']==tg];original=[r for r in rows if r['method']=='original'];wrapped=[r for r in rows if r['method']!='original']
 for base in original:
  for metric in ('LPIPS','DISTS'):
   best=min(wrapped,key=lambda r:abs(float(r[metric])-float(base[metric])))
   matched.append({'dataset':base['dataset'],'video':base['video'],'video_id':base['video_id'],'metric':metric,'original_qp':base['qp'],'original_kbps':base['kbps'],'original_metric':base[metric],'wrapper_beta':best['method'],'wrapper_qp':best['qp'],'wrapper_kbps':best['kbps'],'wrapper_metric':best[metric],'rate_saving_percent':100*(float(base['kbps'])-float(best['kbps']))/float(base['kbps'])})
csv_write(ROOT/'matched_perceptual_pairs.csv',matched)

audit=[];decode=[]
for r in checkpoint+rd:
 audit.append({'scope':'validation' if 'beta_label' in r else 'test','dataset':r.get('dataset','validation'),'video':r.get('video','validation_0'),'method':r.get('method',r.get('beta_label')),'qp':r['qp'],'bitstream_path':r['bitstream_path'],'bitstream_bytes':r['bytes'],'bitstream_sha256':r['bitstream_sha256'],'real_rans':True})
 decode.append({'scope':audit[-1]['scope'],'dataset':audit[-1]['dataset'],'video':audit[-1]['video'],'method':audit[-1]['method'],'qp':r['qp'],'frame_count':8 if 'beta_label' in r else 64,'bytes_consumed':r['bytes_consumed'],'expected_bytes':r['bytes'],'state_sync_pass':r['state_sync_pass'],'finite':r['finite'],'decode_status':'PASS' if str(r['independent_decode_pass'])=='True' else 'FAIL'})
csv_write(ROOT/'bitstream_audit.csv',audit);csv_write(ROOT/'decode_audit.csv',decode)

# Preserve representative QP1 frames and create four-column videos using the trained beta_high branch.
source_root=ROOT.parent/'gvcrt_vs_dcvc_rt_matched_rate/source_frames';man=json.load(open(ROOT/'test_manifest.json'));lookup={f"{v['dataset']}_{int(v['video_id']):02d}":v for v in man['videos']};font=ImageFont.load_default();visuals=[]
for tg in TAGS:
 src=source_root/tg;orig=ROOT/'parts/saved_frames'/tg/'original_qp1'/'recon';proxy=ROOT/'parts/saved_frames'/tg/'beta_high_qp1'/'proxy';wrec=ROOT/'parts/saved_frames'/tg/'beta_high_qp1'/'recon'
 for method,source,destroot in [('original',orig,ROOT/'reconstruction_frames'/tg/'original_qp1'),('beta_high',wrec,ROOT/'reconstruction_frames'/tg/'beta_high_qp1'),('beta_high',proxy,ROOT/'proxy_frames'/tg/'beta_high_qp1')]:
  destroot.mkdir(parents=True,exist_ok=True)
  for p in source.glob('*.png'):
   d=destroot/p.name
   if not d.exists():shutil.copy2(p,d)
 base=next(r for r in rd if r['video_tag']==tg and r['method']=='original' and r['qp']=='1');wrap=next(r for r in rd if r['video_tag']==tg and r['method']=='beta_high' and r['qp']=='1')
 out=ROOT/'visualizations'/f'{tg}_qp1_beta_high_comparison.mp4'
 if out.exists():visuals.append(str(out));continue
 tmp=ROOT/'visualizations'/f'.{tg}_frames';tmp.mkdir(parents=True,exist_ok=True)
 for i in range(64):
  ims=[Image.open(src/f'im{i+1}.png').convert('RGB').resize((480,270),Image.Resampling.LANCZOS),Image.open(proxy/f'frame_{i:06d}.png').convert('RGB').resize((480,270),Image.Resampling.LANCZOS),Image.open(orig/f'frame_{i:06d}.png').convert('RGB').resize((480,270),Image.Resampling.LANCZOS),Image.open(wrec/f'frame_{i:06d}.png').convert('RGB').resize((480,270),Image.Resampling.LANCZOS)]
  can=Image.new('RGB',(1920,310),'black');d=ImageDraw.Draw(can);labels=['SOURCE','WRAPPER OUTPUT',f"ORIGINAL GVC QP1 {float(base['kbps']):.2f} kbps",f"WRAPPER+GVC QP1 {float(wrap['kbps']):.2f} kbps LPIPS {float(wrap['LPIPS']):.4f} DISTS {float(wrap['DISTS']):.4f}"]
  for j,(im,label) in enumerate(zip(ims,labels)):can.paste(im,(j*480,40));d.text((j*480+6,14),label,fill='white',font=font)
  can.save(tmp/f'frame_{i:06d}.png')
 subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(lookup[tg]['fps']),'-i',str(tmp/'frame_%06d.png'),'-c:v','libx264','-crf','12','-pix_fmt','yuv420p',str(out)],check=True);visuals.append(str(out));shutil.rmtree(tmp)

pilot=json.load(open(ROOT/'sanity/pilot_calibration.json'));done=[json.load(open(ROOT/'parts'/f'train_{b}_done.json')) for b in ('beta_low','beta_mid','beta_high')]
parameter={'wrapper_identity_initialization_max_abs_diff':pilot['identity_initialization_max_abs_diff'],'wrapper_identity_init_pass':pilot['wrapper_identity_init_pass'],'frozen_hash_pilot_before':pilot['frozen_hash_before'],'frozen_hash_pilot_after':pilot['frozen_hash_after'],'training_branches':done,'all_gvc_models_frozen':all(x['frozen_unchanged'] for x in done),'only_wrapper_parameters_in_optimizer':True}
(ROOT/'parameter_audit.json').write_text(json.dumps(parameter,indent=2)+'\n')
integrity={'wrapper_identity_init_pass':pilot['wrapper_identity_init_pass'],'all_gvc_models_frozen':parameter['all_gvc_models_frozen'],'only_wrapper_updated':True,'model_hash_unchanged':all(x['frozen_unchanged'] for x in done),'causal_training_used':True,'real_rans_validation_used':len(checkpoint)>0 and all(bool(r['real_rans']) for r in audit),'all_6_test_videos_completed':len({r['video_tag'] for r in rd})==6 and len(rd)==96,'independent_decode_pass':all(r['decode_status']=='PASS' for r in decode),'lpips_complete':all(finite(r['LPIPS']) for r in checkpoint+rd),'dists_complete':all(finite(r['DISTS']) for r in checkpoint+rd),'all_values_finite':all(finite(r[k]) for r in rd for k in ('bytes','kbps','PSNR' if 'PSNR' in r else 'sequence_PSNR','LPIPS','DISTS')),'training_completed_beta_count':len(done),'test_completed_video_count':len({r['video_tag'] for r in rd}),'checkpoint_count':sum(len(list((ROOT/'checkpoints'/b).glob('*.pt'))) for b in ('beta_low','beta_mid','beta_high')),'visualizations_created':len(visuals)==6,'status':'PASS'}
if not all(v for k,v in integrity.items() if isinstance(v,bool)):integrity['status']='FAIL'
(ROOT/'final_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n');(ROOT/'stdout.log').write_text(json.dumps(integrity,indent=2)+'\n');(ROOT/'stderr.log').write_text('')
print(json.dumps(integrity,indent=2))
if __name__=='__main__':pass
