"""CPU-only reporting smoke test; synthetic fixtures never enter experiment results."""
from preflight import R,sha,save as save_audit
import tempfile,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
import report
from evaluate import csvwrite

def main():
 (R/'cache').mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='report_smoke_',dir=R/'cache') as folder:
  root=Path(folder);report.R=root
  def save(name,value):
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,allow_nan=False))
  report.save=save;methods=['A_original','B_rgb_P','C_latent_P'];qs=[0,7,14,21,28,35,42,49,56,63]
  manifests={};rng=np.random.default_rng(20261015)
  for d,n in [('ulong',1),('uvg_validation',1),('mcljcv',30)]:
   manifests[d]=[]
   for v in range(n):
    inputs=root/'inputs'/d/str(v);inputs.mkdir(parents=True)
    for i in (16,32,63):Image.fromarray(np.full((8,8,3),i,dtype=np.uint8)).save(inputs/f'frame_{i:06d}.png')
    manifests[d].append(dict(name=f'videoSRC{v+1:02d}',source_frame_indices=list(range(64)),input_dir=str(inputs)))
    gt=rng.normal(size=(64,4)).astype(np.float32)
    for mi,m in enumerate(methods):
     for qi,q in enumerate(qs):
      ident=f'{d}_{v}_{m}_{q}';feature=root/'features'/f'{ident}.npz';feature.parent.mkdir(parents=True,exist_ok=True)
      np.savez(feature,real=gt,reconstruction=gt+(mi+1)*.01+(9-qi)*.005)
      frame=root/'raw'/f'{ident}.frames.csv';trans=root/'raw'/f'{ident}.transitions.csv'
      csvwrite(frame,[dict(frame=i,q=q,frame_type='I' if i==0 else 'P',LPIPS=.1,DISTS=.05) for i in range(64)])
      csvwrite(trans,[dict(from_frame=i,to_frame=i+1,FloLPIPS=.03) for i in range(63)])
      result=dict(dataset=d,video=v,name=manifests[d][v]['name'],method=m,q=q,bpp=.01+qi*.001+mi*.0001,LPIPS=.2-qi*.005+mi*.001,DISTS=.1-qi*.002+mi*.001,FloLPIPS=.07-qi*.002+mi*.001,LPIPS_P_only=.19-qi*.005+mi*.001,DISTS_P_only=.09-qi*.002+mi*.001,feature=str(feature),per_frame=str(frame),per_transitions=str(trans))
      save(f'results/points/{ident}.json',dict(status='PASS',independent_decode=True,frozen_unchanged=True,result=result,artifacts={str(p):sha(p) for p in (feature,frame,trans)}))
     if v==0:
      dest=root/'previews'/d;dest.mkdir(parents=True,exist_ok=True)
      for i in (16,32,63):Image.fromarray(np.full((8,8,3),i+mi,dtype=np.uint8)).save(dest/f'{m}_{i}.png')
  save('eval_manifest.json',manifests);save('protocol.json',dict(q_grid=qs))
  for m in methods[1:]:
   for i in range(1,1001):save(f'branches/{m}/receipts/{i:04d}.json',dict(step=i,domain='ulong' if i%2 else 'uvg',q=qs[i%10],loss=dict(LPIPS=.1,DISTS=.05,L1=.01),gradient_preclip=.5,clip_coefficient=1.,parameter_delta=.01,seconds=1.))
   save(f'branches/{m}/status.json',dict(updates=1000))
  save('audits/checkpoint_audit.json',dict(models=[]));save('audits/source_inventory.json',dict(files={}))
  report.main();integrity=json.loads((root/'final_integrity.json').read_text())
  assert integrity['status']=='PASS' and integrity['evaluation_completed']==960
  assert integrity['all_frame_metric_rows']==960*64 and integrity['transition_rows']==960*63
  assert integrity['pooled_fid_points']==90 and integrity['P_only_pooled_fid_points']==90
  assert len(list(root.glob('results/*/Rate_*.png')))==12
  assert len(list(root.glob('previews/*/comparison_*.png')))==9
  with (root/'results/mcl29_sensitivity_pooled_fid.csv').open() as f:assert len(list(csv.DictReader(f)))==30
  audit=dict(status='PASS',synthetic_only=True,formal_evaluation_points_added=0,checks=integrity,RD_png_count=12,fixed_comparison_png_count=9,MCL29_sensitivity_rows=30,report_sha256=sha(R/'report.py'),test_sha256=sha(R/'report_smoke.py'))
 save_audit('audits/report_smoke.json',dict(audit,temporary_fixtures_removed=True))
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
