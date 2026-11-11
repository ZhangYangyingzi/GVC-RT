"""Fixed training-window reconstructions at step0/50/100, no checkpoint selection."""
from common_s2 import *


def main():
    env();import torch,numpy as np
    from PIL import Image
    from loss import load_training_context,frame,window
    from cache_h import paths
    c=load_training_context(torch.device('cuda',0));records=[]
    for group in GROUPS:
        for step in [0,50,100]:
            saved=torch.load(ROOT/group/f'training/checkpoints/step_{step:04d}.pt',map_location='cpu',weights_only=True)
            c['adapter'].load_state_dict(saved['adapter'],strict=True)
            missing_stats=not (ROOT/f'training_logs/monitor_{group}_step{step:04d}.json').exists();stats=[]
            for index,row in enumerate(load(ROOT/'manifests/monitor_windows.json')):
                p,rp=paths(row);assert sha(p)==load(rp)['SHA256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
                if missing_stats:
                    with torch.no_grad():trace,_=window(c,d,lambda_temp=saved['lambda_temp'] if group==GROUPS[1] else 0.,backward=False)
                    stats.append(dict(sample_key=row['sample_key'],QP=row['external_qp'],**trace))
                for i in [48,49,52,56,60,63]:
                    with torch.no_grad():out=frame(c,d,i)[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
                    dest=ROOT/f'training_logs/reconstructions/{group}_step{step:04d}_monitor{index}_qp{row["external_qp"]}_frame{i}.png'
                    dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(out).save(dest)
                    gt=ROOT/f'training_logs/reconstructions/GT_monitor{index}_qp{row["external_qp"]}_frame{i}.png'
                    if not gt.exists():Image.fromarray(d['GT'][i-48].numpy()).save(gt)
                    records.append(dict(group=group,step=step,sample_key=row['sample_key'],QP=row['external_qp'],frame=i,path=str(dest),SHA256=sha(dest),GT_path=str(gt),GT_SHA256=sha(gt)))
            if missing_stats:dump(ROOT/f'training_logs/monitor_{group}_step{step:04d}.json',dict(step=step,group=group,training_windows=True,records=stats,recovered_after_interruption=True))
    dump(ROOT/'training_logs/reconstruction_manifest.json',dict(status='PASS',training_windows_only=True,records=records))

if __name__=='__main__':main()
