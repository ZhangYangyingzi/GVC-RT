#!/usr/bin/env python3
import argparse,json,os
from pathlib import Path
import torch
from core import ROOT,quality_models,csv_write
from eval_core import matched_frames,load_wrapper,run_stream

def tag(v):return f"{v['dataset']}_{int(v['video_id']):02d}"
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--tags',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0');man=json.load(open(ROOT/'test_manifest.json'));lookup={tag(v):v for v in man['videos']};quality=quality_models(dev)
 selections={b:json.load(open(ROOT/'parts'/f'checkpoint_selection_{b}.json'))['selected_checkpoint'] for b in ('beta_low','beta_mid','beta_high')};wrappers={b:load_wrapper(p,dev) for b,p in selections.items()};summaries=[];frame_rows=[]
 for tg in a.tags.split(','):
  v=lookup[tg];frames=matched_frames(tg,64)
  for method in ('original','beta_low','beta_mid','beta_high'):
   wrapper=None if method=='original' else wrappers[method]
   for qp in range(4):
    bp=ROOT/'bitstreams'/'test'/tg/f'{method}_qp{qp}.bin';save=None
    if qp==1: save=ROOT/'parts'/'saved_frames'/tg/f'{method}_qp1'
    s,fr=run_stream(frames,qp,wrapper,dev,quality,float(v['fps']),bp,save)
    row={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'video_tag':tg,'method':method,'qp':qp,'num_frames':64,'fps':v['fps'],'checkpoint':'' if method=='original' else selections[method],**s};summaries.append(row)
    frame_rows += [dict(dataset=v['dataset'],video=v['name'],video_id=v['video_id'],video_tag=tg,method=method,qp=qp,**x) for x in fr]
    print(tg,method,qp,s['bytes'],flush=True)
 csv_write(ROOT/'parts'/f'final_rd_gpu{a.gpu}.csv',summaries);csv_write(ROOT/'parts'/f'final_frames_gpu{a.gpu}.csv',frame_rows)
 (ROOT/'parts'/f'eval_gpu{a.gpu}_done.json').write_text(json.dumps({'status':'PASS','tags':a.tags.split(','),'streams':len(summaries)},indent=2)+'\n')
if __name__=='__main__':main()
