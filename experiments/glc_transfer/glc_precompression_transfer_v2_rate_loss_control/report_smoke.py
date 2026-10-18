"""CPU-only synthetic report test in an auto-removed, isolated fixture tree."""
from preflight import R,sha,save as save_audit
import tempfile,json,csv,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import report,fid_pool
from evaluate import csvwrite
from bootstrap import BRANCHES
def fast_fid(a,b):
 from scipy.linalg import sqrtm
 ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False);dm=float(((a.mean(0)-b.mean(0))**2).sum());dc=float(np.trace(ca+cb)-2*np.trace(sqrtm(ca@cb).real));assert dm+dc>=-1e-7
 return dict(FID=max(0.,dm+dc),D_mu=dm,D_cov=dc,frames=len(a),ddof=1,GT_hash=hashlib.sha256(a.tobytes()).hexdigest())
def main():
 rng=np.random.default_rng(20261016);a=rng.normal(size=(8,4));b=rng.normal(size=(8,4));assert abs(fast_fid(a,b)['FID']-fid_pool.components(a,b)['FID'])<1e-8
 (R/'cache').mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='report_smoke_',dir=R/'cache') as folder:
  root=Path(folder);report.R=root;fid_pool.R=root
  def save(name,value):
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,allow_nan=False))
  report.save=save;fid_pool.save=save;fid_pool.components=fast_fid
  class FakeFrames:
   def __init__(self,row):pass
   def array(self,i):return np.full((8,8,3),i,dtype=np.uint8)
   def close(self):pass
  report.Frames=FakeFrames;methods=['A_original','v1_C_latent_P',*BRANCHES];qs=[0,7,14,21,28,35,42,49,56,63];manifest={}
  for d,n in [('ulong',1),('uvg_validation',1),('mcljcv',30)]:
   manifest[d]=[]
   for video in range(n):
    manifest[d].append(dict(name=f'videoSRC{video+1:02d}',source_frame_indices=list(range(64)),width=8,height=8,rate_accounting_fps=30))
    gt=rng.normal(size=(64,4)).astype(np.float32)
    for mi,m in enumerate(methods):
     for qi,q in enumerate(qs):
      ident=f'{d}_{video}_{m}_{q}';feature=root/'features'/f'{ident}.npz';feature.parent.mkdir(parents=True,exist_ok=True);np.savez(feature,real=gt,reconstruction=gt+.001*(mi+1)+.001*(9-qi))
      ff=root/'raw'/f'{ident}.frames.csv';tt=root/'raw'/f'{ident}.transitions.csv';csvwrite(ff,[dict(frame=i,LPIPS=.1,DISTS=.05) for i in range(64)]);csvwrite(tt,[dict(from_frame=i,to_frame=i+1,FloLPIPS=.03) for i in range(63)])
      size=1000+qi*100+mi*5;r=dict(dataset=d,video=video,name=manifest[d][video]['name'],method=m,q=q,frames=64,transitions=63,bytes=size,bpp=size*8/(64*8*8),rate_accounting_fps=30,LPIPS=.2-qi*.005+mi*.001,DISTS=.1-qi*.002+mi*.001,FloLPIPS=.07-qi*.002+mi*.001,LPIPS_P_only=.19-qi*.005+mi*.001,DISTS_P_only=.09-qi*.002+mi*.001,feature=str(feature),per_frame=str(ff),per_transitions=str(tt))
      receipt=dict(status='PASS',independent_decode=True,frozen_unchanged=True,result=r,artifacts={str(p):sha(p) for p in (feature,ff,tt)})
      if mi<2:receipt['reuse']=dict(synthetic=True)
      save(f'results/points/{ident}.json',receipt)
     if video==0:
      p=root/'previews'/d;p.mkdir(parents=True,exist_ok=True)
      for i in (16,32,63):Image.fromarray(np.full((8,8,3),i+mi,dtype=np.uint8)).save(p/f'{m}_{i}.png')
  save('eval_manifest.json',manifest);save('protocol.json',dict(q_grid=qs,evaluation_models=methods));save('audits/source_experiment.json',dict(files={}));save('audits/v1_source_inventory.json',dict(files={}));save('audits/checkpoint_audit.json',dict(models=[]));save('audits/rate_backend_decision.json',dict(backend='official_verified',changed=False));save('audits/rate_audit_index.json',dict(paired_completed=120,gradient_completed=80,artifacts={}))
  for branch in BRANCHES:
   for i in range(1,1001):save(f'branches/{branch}/receipts/{i:04d}.json',dict(branch=branch,step=i,effective_updates=i,loss=dict(LPIPS=.1,DISTS=.05),source_step=1000,adaptation_step=i))
   save(f'branches/{branch}/status.json',dict(updates=1000))
  report.main();integrity=json.loads((root/'final_integrity.json').read_text());assert integrity['status']=='PASS' and integrity['evaluation_completed']==1920 and integrity['new_evaluation_points']==1280 and integrity['reused_evaluation_points']==640 and integrity['all_frame_metric_rows']==122880 and integrity['transition_rows']==120960 and integrity['RD_png_count']==12
  with (root/'results/equal_rate_per_sequence.csv').open() as f:rows=list(csv.DictReader(f))
  assert {r['comparison_family'] for r in rows}=={'four_branch_common','all_six_common'}
  for r in rows:assert len(r['common_methods'].split(';'))==(4 if r['comparison_family']=='four_branch_common' else 6)
  audit=dict(status='PASS',synthetic_only=True,formal_points_added=0,checks=integrity,common_rate_families_verified=True,FID_synthetic_four_dimensional_covariance_shortcut='equivalence checked against formal low-rank formula on an independent small sample',report_sha256=sha(R/'report.py'),fid_pool_sha256=sha(R/'fid_pool.py'))
 save_audit('audits/report_smoke.json',dict(audit,fixtures_removed=True));print('REPORT SMOKE PASS',flush=True)
if __name__=='__main__':main()
