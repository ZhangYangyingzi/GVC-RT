#!/usr/bin/env python3
import argparse,json,os
from pathlib import Path
import torch
from core import ROOT, quality_models, csv_write
from eval_core import video_frames,load_wrapper,run_stream

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--beta-label',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0');cfg=json.load(open(ROOT/'config.json'));beta=float(cfg['beta_weights'][a.beta_label]);quality=quality_models(dev)
 split=json.load(open(ROOT.parents[1]/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11/data_split.json'));v=split['splits']['validation'][0];frames=video_frames(v['local_path'],8);fps=float(v['probe']['avg_frame_rate'].split('/')[0])/float(v['probe']['avg_frame_rate'].split('/')[1]);rows=[]
 for cp in sorted((ROOT/'checkpoints'/a.beta_label).glob('*.pt')):
  wrapper=load_wrapper(cp,dev);step=5000 if cp.name=='final.pt' else int(cp.stem.split('_')[-1])
  for qp in (1,3):
   bp=ROOT/'bitstreams'/'validation'/f'{a.beta_label}_{cp.stem}_qp{qp}.bin';s,_=run_stream(frames,qp,wrapper,dev,quality,fps,bp)
   rows.append({'beta_label':a.beta_label,'beta':beta,'checkpoint':str(cp),'checkpoint_name':cp.name,'step':step,'qp':qp,**s});print(a.beta_label,cp.name,qp,s['bytes'],flush=True)
 csv_write(ROOT/'parts'/f'checkpoint_rans_{a.beta_label}.csv',rows)
 grouped={}
 for r in rows:grouped.setdefault(r['checkpoint'],[]).append(r)
 score=lambda rs:sum(float(x['LPIPS'])+float(x['DISTS'])+beta*float(x['bpp']) for x in rs)/len(rs)
 best=min(grouped,key=lambda k:score(grouped[k]));result={'beta_label':a.beta_label,'beta':beta,'selected_checkpoint':best,'selection_score':score(grouped[best]),'real_rans_used':True,'status':'PASS'};(ROOT/'parts'/f'checkpoint_selection_{a.beta_label}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
