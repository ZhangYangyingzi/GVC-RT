"""Synthetic fixtures for numerical screening, no training/evaluation data mutation."""
from common_s2 import *
def main():
 env();import selection as s
 base=ROOT/'tmp/screen_fixture';base.mkdir(parents=True,exist_ok=True);s.ROOT=base
 dump(base/'audits/H2_gate.json',dict(status='MECHANICAL_CHECK_FAILED'))
 def points(model,step,S,guard=1.):
  out=[]
  for q in [0,5,9]:
   for i in range(32):out.append(dict(sequence='SYNTHETIC_'+str(i),QP=q,Sobel_MCResidual_PtoP=S,DISTS=guard,LPIPS_Alex=guard,LPIPS_VGG=guard,FloLPIPS=guard))
  for i,r in enumerate(out):dump(base/f'evaluation/development/{model}/step_{step:04d}/points/{i}.json',dict(result=r))
  cp=base/model/f'training/checkpoints/step_{step:04d}.pt';cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(b'SYNTHETIC_NOT_MODEL')
 def reset():
  if (base/'selection.json').exists():(base/'selection.json').unlink()
 reset();points('B300',0,1)
 for g in GROUPS[:2]:points(g,50,.9);points(g,100,.8997)
 a=s.select();assert all(a['groups'][g]['selected']['step']==50 and a['groups'][g]['qualified'] for g in GROUPS[:2]);assert a['groups'][GROUPS[2]]['status']=='MECHANICAL_CHECK_FAILED'
 reset()
 for g in GROUPS[:2]:points(g,100,.8994)
 a=s.select();assert all(a['groups'][g]['selected']['step']==100 for g in GROUPS[:2])
 reset()
 for g in GROUPS[:2]:points(g,50,.8,1.006);points(g,100,.7,1.006)
 a=s.select();assert not a['any_qualified'] and all(a['groups'][g]['selected'] is None and a['groups'][g]['status']=='NOT_QUALIFIED' for g in GROUPS[:2])
 from media import ffmpeg_probe
 ffmpeg_probe()
 dump(ROOT/'audits/mechanical_screen_tests.json',dict(status='PASS',synthetic_only=True,tests=['eligible earlier if S separation<0.0005','later if improvement>=0.0005','1.006 guard fails; no step0 substitution; H2 failure excluded','FFV1 xstack, labels and source-frame start_number'],time=time.time()));print('SCREEN_AND_FFMPEG_TEST_PASS')
if __name__=='__main__':main()
