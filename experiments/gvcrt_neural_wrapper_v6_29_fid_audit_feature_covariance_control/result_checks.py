from io21 import *
import statistics,math
from comparison import PAIRS

def main():
 logs=[json.loads(x) for x in (ROOT/'branches/B_covstats/training_logs/train.jsonl').read_text().splitlines()];assert len(logs)==1000
 lam=load(ROOT/'calibration/final.json')['lambda_stat'];counts={}
 for r,p in zip(logs,load(ROOT/'plans/shared.json')['updates']):
  assert all(r[k]==p[k] for k in p) and r['lambda_stat']==lam and r['optimizer_update_count']==r['adaptation_step']
  assert r['supervision_frames']==15 and r['warmup_frames']==48 and r['reference_resets']==[0]
  assert r['all_finite'] and r['stat_gradient_norm']>0 and r['B1_gradient_norm']>0
  assert abs(r['stat_B1_ratio']-lam*r['stat_gradient_norm']/r['B1_gradient_norm'])<1e-12
  counts[(r['domain'],r['external_qp'])]=counts.get((r['domain'],r['external_qp']),0)+1
 assert len(counts)==20 and all(x==50 for x in counts.values())
 write(ROOT/'branches/B_covstats/stat_gradient_summary.csv',[dict(step=r['adaptation_step'],domain=r['domain'],QP=r['external_qp'],L_stat=r['stat_loss'],lambda_stat=r['lambda_stat'],weighted_L_stat=lam*r['stat_loss'],g0_norm=r['B1_gradient_norm'],gstat_unit_norm=r['stat_gradient_norm'],weighted_stat_norm=r['weighted_stat_gradient_norm'],weighted_stat_B1_ratio=r['stat_B1_ratio'],cosine=r['stat_cosine'],global_preclip_norm=r['global_preclip_norm'],clip_coefficient=r['global_clip_coefficient'],update=r['optimizer_update_count'],**{k:r[k] for k in ('bridge_B1_gradient_norm','generator_B1_gradient_norm','bridge_adv_gradient_norm','generator_adv_gradient_norm')}) for r in logs])
 frames=[]
 for r in logs:
  for f in r['frame_trace']:
   frames.append(dict(step=r['adaptation_step'],domain=r['domain'],video=r['video'],window_position=f['window_position'],actual_qp=f['actual_qp'],stat_loss=f['projected_adv'],lambda_stat=lam,weighted_stat_loss=lam*f['projected_adv'],B1_loss=f['loss'],effective_full_loss=f['loss']+lam*f['projected_adv'],source_phase='restored unit-gradient collection; effective fixed coefficient applied after15-frame average'))
 write(ROOT/'branches/B_covstats/training_logs/per_frame_covariance_loss.csv',frames)
 sameqp=[]
 iaudit=load(ROOT/'audits/native_I_payload_reconstruction.json');assert iaudit['status']=='PASS' and iaudit['points']==1840
 from native_i_audit import key as i_key
 for d in DATASETS:
  for v in sources(d):
   for q in range(10):
    rs={m:load(point(d,m,v['video_index'],q)) for m in METHODS}
    native=[iaudit['records'][i_key(d,v['video_index'],m,q)] for m in METHODS]
    for m in METHODS:
     assert iaudit['records'][i_key(d,v['video_index'],m,q)]['bitstream_sha256']==rs[m]['bitstream_sha256']
    assert len({r['native_I_sha256'] for r in native})==1 and len({r['payload_sha256'] for r in native})==1
    for m,ref in PAIRS:
     a,b=rs[m],rs[ref];sameqp.append(dict(dataset=d,sequence=v['name'],method=m,reference=ref,QP=q,bpp_method=a['bpp'],bpp_reference=b['bpp'],delta_bpp=a['bpp']-b['bpp'],delta_real_bytes=a['real_bytes']-b['real_bytes'],**{f'delta_{k}':a[k]-b[k] for k in ('LPIPS','DISTS','FloLPIPS')},comparison='same external QP diagnostic; not equal-rate'))
 write(ROOT/'results/same_QP_rate_quality_differences.csv',sameqp)
 assets={}
 for p in ROOT.rglob('*'):
  if p.is_file() and p.suffix in ('.pt','.pth','.npy','.npz','.bin','.ckpt') and '.git' not in p.parts:assets[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
 for d in DATASETS:
  for v in sources(d):
   for m in METHODS:
    for q in range(10):
     r=load(point(d,m,v['video_index'],q))
     for key,h in (('feature_path','feature_sha256'),('bitstream_path','bitstream_sha256')):
      p=Path(r[key]);assets[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=r[h],source='verified RD record')
 for b in BRANCHES:
  for cp in load(ROOT/'branches'/b/'checkpoint_index.json').values():
   p=Path(cp['path']);assets[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=cp['sha256'])
 dump(ROOT/'binary_artifacts_index.json',dict(files=list(assets.values()),binary_upload=False))
 dump(ROOT/'audits/training_counts_integrity.json',dict(status='PASS',A_updates=1000,A_verified_reuse=True,B_updates=len(logs),counts={str(k):v for k,v in counts.items()},constant_lambda_stat=lam,mean15_once=True,single_clip=True,no_GAN=True))
if __name__=='__main__':main()
