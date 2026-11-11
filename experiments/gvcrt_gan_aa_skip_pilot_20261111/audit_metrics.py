"""Engineering execution on a training crop; not a development metric point."""
from common_s2 import *


def main():
    env()
    from cache_codec import cache_paths
    row=plan_rows()[0];_,path,receipt=cache_paths(row,row['external_qp'],True)
    output=ROOT/'cache/engineering_original_FP32.pt'
    if not output.exists():subprocess.run([PYTHON,'original_reconstruction.py',str(path),str(output)],check=True,cwd=ROOT)
    import torch
    from evaluate import Scorer
    scorer=Scorer();data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);original=torch.load(output,map_location='cpu',weights_only=True,mmap=True)
    frames=[(x.float().clamp(-1,1)+1).to('cuda:0')/2 for x in original['values'][:2]]
    gt=[data['GT'][i].permute(2,0,1)[None].to('cuda:0').float()/255 for i in range(2)]
    with torch.no_grad():
        a=float(scorer.alex(frames[1],gt[1],normalize=True).mean());v=float(scorer.vgg(frames[1],gt[1],normalize=True).mean());d=float(scorer.dists(frames[1],gt[1]).mean())
        diff=scorer.flow(gt[0],gt[1])-scorer.flow(frames[0],frames[1]);flo=float(scorer.flo(gt[0],frames[0],diff,normalize=True).mean())
    assert all(math.isfinite(x) for x in [a,v,d,flo])
    dump(ROOT/'audits/metric_execution_smoke.json',dict(status='PASS',training_crop_only=True,development_points_added=0,Original_independent=original['no_pilot_TCG_LoRA'],precision=original['precision'],LPIPS_Alex=a,LPIPS_VGG=v,DISTS=d,FloLPIPS=flo,metric_weights=scorer.hashes,original_output_SHA256=sha(output),time=time.time()))


if __name__=='__main__':main()
