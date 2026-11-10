"""Read-only assertions on B30052-key identity and recoverable H latest states."""
from common_s2 import *
def main():
 env();import torch
 b=torch.load(B300,map_location='cpu',weights_only=True);assert len(b['trainable_state'])==52 and b['committed_updates']==300
 rows=[]
 for g in GROUPS:
  latest=ROOT/g/'training/latest.pt'
  if not latest.exists():continue
  a=torch.load(latest,map_location='cpu',weights_only=True);step=a['committed_updates'];assert a['next_cursor']==step*4 and a['B300_SHA256']==B300_SHA and a['common_step0_SHA256']==sha(ROOT/'checkpoints/common_B300_step0.pt');assert all(int(v['step'])==step for v in a['optimizer']['state'].values());assert len(a['trainable_state'])>52
  rows.append(dict(group=g,step=step,latest_SHA256=sha(latest),trainable_parameters=sum(v.numel() for v in a['trainable_state'].values()),optimizer_entries=len(a['optimizer']['state']),RNG_keys=list(a['RNG']),next_cursor=a['next_cursor']))
 dump(ROOT/'audits/checkpoint_recoverability.json',dict(status='PASS',B300_overlay_keys=52,B300_SHA256=sha(B300),checked=rows,time=time.time()));print('CHECKPOINT_RECOVERABILITY_PASS')
if __name__=='__main__':main()
