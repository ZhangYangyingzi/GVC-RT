from io21 import *
import numpy as np,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
full=np.zeros(1024,dtype=np.int64);tiny=np.zeros(1024,dtype=np.int64);byMAIN=collections.defaultdict(lambda:np.zeros(1024,dtype=np.int64));byTINY=collections.defaultdict(lambda:np.zeros(1024,dtype=np.int64))
import torch
for r in load(ROOT/'plans/shared.json')['updates']:
 raw=torch.load(ROOT/'cache/raw'/f"main_{r['adaptation_step']:04d}.pt",map_location='cpu',weights_only=True)
 for y in raw['labels']:
  cc=torch.bincount(y.flatten(),minlength=1024).numpy();full+=cc;byMAIN[(r['domain'],r['external_qp'])]+=cc
for w in load(ROOT/'plans/tiny.json')['windows']:
 raw=torch.load(ROOT/'cache/raw'/f"{w['row_id']}.pt",map_location='cpu',weights_only=True)
 for pos in w['positions']:
  cc=torch.bincount(raw['labels'][pos-49].flatten(),minlength=1024).numpy();tiny+=cc;byTINY[(w['domain'],w['external_qp'])]+=cc
dump(ROOT/'audits/training_frequency_reference.json',dict(status='PASS',Laplace_alpha=1.,content_classes=1024,balanced_fake_mass=.5,MAIN_counts=full.tolist(),TINY_counts=tiny.tolist(),MAIN_frames=15000,TINY_frames=8,primary_reference_scope='training domain × external_QP',MAIN_domain_QP_counts={str(k):v.tolist() for k,v in byMAIN.items()},TINY_domain_QP_counts={str(k):v.tolist() for k,v in byTINY.items()},source='only fixed training GT labels; no validation labels used to construct predictive references'))
print('TRAINING FREQUENCY REFERENCE PASS',flush=True)
