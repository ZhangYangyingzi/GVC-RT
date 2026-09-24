#!/usr/bin/env python3
import argparse,csv,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
def rc_dir(codec,rc):
 if codec=='gvc':return f'GVC_qp{rc[2:]}'
 a,b=rc.replace('I','').split('_P');return f'DCVC_q{a}' if a==b else f'DCVC_qi{a}_qp{b}'
def label(im,text):
 x=im.copy();d=ImageDraw.Draw(x);d.rectangle((0,0,760,54),fill=(0,0,0));d.text((12,10),text,fill=(255,255,255),font=ImageFont.load_default(size=30));return x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mod',type=int,default=1);ap.add_argument('--rem',type=int,default=0);a=ap.parse_args()
 pairs=list(csv.DictReader(open(ROOT/'matched_rate_pairs.csv')));metrics=list(csv.DictReader(open(ROOT/'matched_pair_metrics.csv')))
 for pi,p in enumerate(pairs):
  if pi%a.mod!=a.rem:continue
  if p['match_status']=='NO_CLOSE_REAL_POINT':continue
  tag=f"{p['dataset']}_{int(p['video_id']):02d}";gd=ROOT/'reconstructions'/tag/rc_dir('gvc',p['gvc_qp']);dd=ROOT/'reconstructions'/tag/rc_dir('dcvc',p['dcvc_rate_control']);sd=ROOT/'source_frames'/tag
  prefix=(f"ULong{int(p['video_id']):02d}" if p['dataset']=='fresh_ulong' else p['video'])+f"_{p['gvc_qp']}_vs_{p['dcvc_rate_control']}"
  trip=ROOT/'parts'/f'triptych_{prefix}';trip.mkdir(parents=True,exist_ok=True)
  m=next(x for x in metrics if x['dataset']==p['dataset'] and int(x['video_id'])==int(p['video_id']) and x['gvc_qp']==p['gvc_qp'])
  for fi in range(64):
   ims=[Image.open(sd/f'im{fi+1}.png').convert('RGB'),Image.open(dd/f'im{fi+1:05d}.png').convert('RGB'),Image.open(gd/f'im{fi+1:05d}.png').convert('RGB')]
   labs=['SOURCE',f"DCVC-RT {float(p['dcvc_kbps']):.3f} kbps  PSNR {float(m['dcvc_sequence_PSNR']):.3f}",f"GVC-RT {float(p['gvc_kbps']):.3f} kbps  PSNR {float(m['gvc_sequence_PSNR']):.3f}"]
   ims=[label(x,t) for x,t in zip(ims,labs)];canvas=Image.new('RGB',(5760,1080));
   for j,x in enumerate(ims):canvas.paste(x,(j*1920,0))
   canvas.save(trip/f'frame_{fi:06d}.png',compress_level=1)
   if fi in (0,16,32,48,63):canvas.save(ROOT/'comparison_frames'/f'{prefix}_frame_{fi:06d}.png',compress_level=1)
  fps=str(p['fps']);mkv=ROOT/'visualizations'/f'{prefix}_matched_triptych.mkv';mp4=ROOT/'visualizations'/f'{prefix}_matched_triptych_view_only.mp4'
  subprocess.run(['ffmpeg','-y','-v','error','-framerate',fps,'-i',str(trip/'frame_%06d.png'),'-c:v','ffv1','-level','3','-pix_fmt','yuv444p',str(mkv)],check=True)
  subprocess.run(['ffmpeg','-y','-v','error','-framerate',fps,'-i',str(trip/'frame_%06d.png'),'-c:v','libx264','-preset','slow','-crf','10','-pix_fmt','yuv420p',str(mp4)],check=True)
  print(prefix,flush=True)
if __name__=='__main__':main()
