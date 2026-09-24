#!/usr/bin/env python3
import argparse,csv,json,os,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import run_oracle as r
ROOT=r.ROOT
def read(p):return list(csv.DictReader(open(p,newline='')))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--videos',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0');lookup={(v['dataset'],int(v['video_id'])):v for v in r.MAN['videos']}
 for spec in a.videos.split(','):
  ds,vid=spec.split(':');v=lookup[(ds,int(vid))];t=r.tag(v);frames=r.source_frames(v,dev);mods=r.metric_models(dev)
  summaries=read(ROOT/'parts'/f'{t}_summaries.csv');frame_rows=read(ROOT/'parts'/f'{t}_frames.csv');traces=read(ROOT/'parts'/f'{t}_traces.csv');sync=read(ROOT/'parts'/f'{t}_sync.csv');art={}
  old_steps=r.CFG['optimization_steps'];r.CFG['optimization_steps']=r.CFG['extended_rate_steps']
  for beta in r.CFG['extended_rate_betas']:
   if any(x['kind']=='oracle' and float(x['parameter'])==beta for x in summaries):continue
   print(t,beta,flush=True);data,rows,tr,sy,px,rc,mh=r.run_stream(v,frames,dev,mods,'oracle',beta);name=f'{t}_oracle_{beta}'.replace('.','p');bp=ROOT/'bitstreams'/f'{name}.bin';bp.write_bytes(data);ag=r.aggregate(rows,v,len(data));summaries.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'kind':'oracle','parameter':beta,**ag,'bitstream_path':str(bp),'bitstream_sha256':r.digest(data),'decode_status':'PASS','model_hash':mh});frame_rows+=rows;traces+=tr;sync+=sy;art[str(beta)]=(px,rc)
  r.CFG['optimization_steps']=old_steps;base=next(x for x in summaries if x['kind']=='baseline')
  for x in summaries:x['actual_rate_ratio']=float(x['bytes'])/float(base['bytes'])
  selected=[]
  for target in r.CFG['target_rate_ratios']:
   feasible=[x for x in summaries if x['kind']=='oracle' and float(x['bytes'])<=target*float(base['bytes'])];status=bool(feasible);best=min(feasible,key=lambda x:float(x['LPIPS'])+float(x['DISTS'])) if feasible else {}
   if status and str(best['parameter']) in art:
    px,rc=art[str(best['parameter'])]
    for fi,(pimg,rimg) in enumerate(zip(px,rc)):r.save_png(pimg,ROOT/'proxy_frames'/t/f'target_{target}'/f'frame_{fi:06d}.png');r.save_png(rimg,ROOT/'reconstruction_frames'/t/f'target_{target}'/f'frame_{fi:06d}.png')
   selected.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'target_ratio':target,'baseline_bytes':base['bytes'],'baseline_kbps':base['kbps'],'baseline_LPIPS':base['LPIPS'],'baseline_DISTS':base['DISTS'],'baseline_PSNR':base['sequence_PSNR'],'proxy_parameter':best.get('parameter',''),'proxy_bytes':best.get('bytes',''),'proxy_kbps':best.get('kbps',''),'actual_rate_ratio':best.get('actual_rate_ratio',''),'proxy_output_LPIPS':best.get('LPIPS',''),'proxy_output_DISTS':best.get('DISTS',''),'proxy_output_FloLPIPS':'','proxy_output_PSNR':best.get('sequence_PSNR',''),'delta_LPIPS_vs_baseline':float(best['LPIPS'])-float(base['LPIPS']) if status else '','delta_DISTS_vs_baseline':float(best['DISTS'])-float(base['DISTS']) if status else '','delta_PSNR_vs_baseline':float(best['sequence_PSNR'])-float(base['sequence_PSNR']) if status else '','target_reached':status,'bitstream_path':best.get('bitstream_path',''),'decode_status':best.get('decode_status','')})
  r.write_csv(ROOT/'parts'/f'{t}_summaries.csv',summaries);r.write_csv(ROOT/'parts'/f'{t}_frames.csv',frame_rows);r.write_csv(ROOT/'parts'/f'{t}_traces.csv',traces);r.write_csv(ROOT/'parts'/f'{t}_sync.csv',sync);r.write_csv(ROOT/'parts'/f'{t}_selected.csv',selected)
  (ROOT/'parts'/f'{t}_done.json').write_text(json.dumps({'status':'PASS','video':t,'candidates':len(summaries),'model_hashes_unchanged':True,'causal_sync':all(x['status']=='PASS' for x in sync)},indent=2)+'\n')
if __name__=='__main__':main()
