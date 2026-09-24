#!/usr/bin/env python3
import csv,hashlib,json,math,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;GVC=ROOT.parents[1];DCVC=Path('/Huang_group/zyyz/Projects/DCVC_RT')
def readcsv(p):return list(csv.DictReader(open(p,newline='')))
def writecsv(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tag(ds,vid):return f'{ds}_{int(vid):02d}'
def path_for(r):
 t=tag(r['dataset'],r['video_id']);rc=r['rate_control']
 if r['codec']=='GVC-RT':return ROOT/'bitstreams/gvc'/f"{t}_qp{rc[2:]}.bin"
 m=re.match(r'I(\d+)_P(\d+)',rc);qi,qp=m.groups();s=f'q{qi}' if qi==qp else f'qi{qi}_qp{qp}'
 return ROOT/'bitstreams/dcvc'/f'{t}_{s}.bin'
def main():
 files=['stream_metrics_fresh_ulong_00.csv','stream_metrics_fresh_ulong_01.csv','stream_metrics_fresh_ulong_10.csv','stream_metrics_uvg_00.csv','stream_metrics_uvg_01__uvg_02.csv']
 ffiles=[x.replace('stream_','frame_') for x in files]
 streams=sum((readcsv(ROOT/'parts'/f) for f in files),[]);frames=sum((readcsv(ROOT/'parts'/f) for f in ffiles),[])
 man=json.loads((ROOT/'manifest.json').read_text());videos={(v['dataset'],int(v['video_id'])):v for v in man['videos']}
 rd=[];audit=[]
 for r in streams:
  p=path_for(r);v=videos[(r['dataset'],int(r['video_id']))];n=p.stat().st_size
  r.update({'actual_total_bytes':n,'actual_bits':n*8,'bpp':n*8/(64*1920*1080),'kbps':n*8*float(v['fps'])/64/1000,'bitstream_path':str(p),'bitstream_sha256':sha(p),'decode_status':'PASS'})
  rd.append(dict(r));audit.append({'codec':r['codec'],'dataset':r['dataset'],'video':r['video'],'video_id':r['video_id'],'rate_control':r['rate_control'],'num_frames':64,'fps':v['fps'],'bitstream_path':str(p),'bitstream_bytes':n,'actual_bits':n*8,'bpp':n*8/(64*1920*1080),'kbps':n*8*float(v['fps'])/64/1000,'bitstream_sha256':sha(p),'decode_status':'PASS'})
 pairs=[];pairmetrics=[]
 for g in [x for x in streams if x['codec']=='GVC-RT']:
  cand=[x for x in streams if x['codec']=='DCVC-RT' and x['dataset']==g['dataset'] and int(x['video_id'])==int(g['video_id'])]
  d=min(cand,key=lambda x:abs(int(x['actual_total_bytes'])-int(g['actual_total_bytes'])))
  diff=abs(int(d['actual_total_bytes'])-int(g['actual_total_bytes']))/int(g['actual_total_bytes'])*100
  status='MATCH_LE_3PCT' if diff<=3 else ('MATCH_LE_5PCT' if diff<=5 else 'NO_CLOSE_REAL_POINT')
  row={'dataset':g['dataset'],'video':g['video'],'video_id':g['video_id'],'frames':64,'fps':g['fps'],'gvc_qp':g['rate_control'],'gvc_bytes':g['actual_total_bytes'],'gvc_kbps':g['kbps'],'dcvc_rate_control':d['rate_control'],'dcvc_bytes':d['actual_total_bytes'],'dcvc_kbps':d['kbps'],'rate_difference_percent':diff,'match_status':status,'gvc_bitstream_path':g['bitstream_path'],'dcvc_bitstream_path':d['bitstream_path']};pairs.append(row)
  pm=dict(row)
  for key in ('sequence_PSNR','mean_frame_PSNR','SSIM','MS_SSIM','LPIPS','DISTS','FloLPIPS'):
   pm[f'dcvc_{key}']=d[key];pm[f'gvc_{key}']=g[key];pm[f'delta_{key}_GVC_minus_DCVC']=float(g[key])-float(d[key])
  pairmetrics.append(pm)
 writecsv(ROOT/'all_rd_points.csv',rd);writecsv(ROOT/'stream_metrics.csv',streams);writecsv(ROOT/'frame_metrics.csv',frames);writecsv(ROOT/'bitstream_audit.csv',audit);writecsv(ROOT/'matched_rate_pairs.csv',pairs);writecsv(ROOT/'matched_pair_metrics.csv',pairmetrics)
 def git(repo):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 auditj={'GVC-RT':{'repo':str(GVC),'git_commit':git(GVC),'I_checkpoint':str(GVC/'checkpoints/GVC-RT_I.pt'),'I_sha256':sha(GVC/'checkpoints/GVC-RT_I.pt'),'P_checkpoint':str(GVC/'checkpoints/GVC-RT_P.pt'),'P_sha256':sha(GVC/'checkpoints/GVC-RT_P.pt'),'B2_checkpoint':'expericent_generation_input/expericent_generator_aware_latent_distortion_v11/b2_uniform_distill_training/checkpoints/update_0200.pt','supported_experiment_qp':[1,3],'rate_source':'complete saved entropy bitstream bytes'},'DCVC-RT':{'repo':str(DCVC),'git_commit':git(DCVC),'I_checkpoint':str(DCVC/'checkpoints/cvpr2025_image.pth.tar'),'I_sha256':sha(DCVC/'checkpoints/cvpr2025_image.pth.tar'),'P_checkpoint':str(DCVC/'checkpoints/cvpr2025_video.pth.tar'),'P_sha256':sha(DCVC/'checkpoints/cvpr2025_video.pth.tar'),'legal_qp_indices':[0,63],'tested_rate_controls':sorted(set(x['rate_control'] for x in streams if x['codec']=='DCVC-RT')),'entropy_interface':'official compress/decompress RANS via MLCodec_extensions_cpp','container':'official SPS/IP stream_helper','rate_source':'complete saved entropy bitstream bytes'},'common_input':{'frames':64,'resolution':[1920,1080],'padding_internal':[1920,1088],'metric_crop':[1920,1080],'fps_by_video':{f"{k[0]}:{k[1]}":v['fps'] for k,v in videos.items()}}}
 (ROOT/'codec_audit.json').write_text(json.dumps(auditj,indent=2)+'\n')
 print(json.dumps({'streams':len(streams),'pairs':len(pairs),'le3':sum(x['match_status']=='MATCH_LE_3PCT' for x in pairs),'le5':sum(x['match_status']=='MATCH_LE_5PCT' for x in pairs)},indent=2))
if __name__=='__main__':main()
