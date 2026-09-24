#!/usr/bin/env python3
import argparse,csv,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from core import ROOT,csv_write,REPO

sys.path.insert(0,str(REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src'))
def load(p,dev):
 a=np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8);return torch.from_numpy(a.copy()).permute(2,0,1).float().div_(255).to(dev)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--tags',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0')
 from flolpips_metric import flolpips_for_videos
 from flolpips_compat import load_models,reference_flows
 flow,metric=load_models(dev);source_root=ROOT.parent/'gvcrt_vs_dcvc_rt_matched_rate/source_frames';rows=[]
 for tag in a.tags.split(','):
  src=torch.stack([load(source_root/tag/f'im{i}.png',dev) for i in range(1,65)]);flows=reference_flows(src,flow)
  for method in ('original','beta_low','beta_mid','beta_high'):
   d=ROOT/'parts/saved_frames'/tag/f'{method}_qp1'/'recon';rec=torch.stack([load(d/f'frame_{i:06d}.png',dev) for i in range(64)]);score,_=flolpips_for_videos(src,rec,flow,metric,flows);rows.append({'video_tag':tag,'method':method,'qp':1,'FloLPIPS':float(score)});del rec;torch.cuda.empty_cache();print(tag,method,float(score),flush=True)
   
  del src,flows;torch.cuda.empty_cache()
 csv_write(ROOT/'parts'/f'flolpips_gpu{a.gpu}.csv',rows)
if __name__=='__main__':main()
