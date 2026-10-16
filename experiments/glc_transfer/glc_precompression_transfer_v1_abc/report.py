from preflight import *
from data import load
from evaluate import csvwrite
def fid_components(a,b):
 import numpy as np
 from scipy.linalg import svdvals
 assert a.shape==b.shape and len(a)>1 and np.isfinite(a).all() and np.isfinite(b).all()
 ac=a-a.mean(0);bc=b-b.mean(0)
 dm=float(((a.mean(0)-b.mean(0))**2).sum())
 dc=float((ac**2).sum()/(len(a)-1)+(bc**2).sum()/(len(b)-1)-2*svdvals(ac@bc.T/(len(a)-1)).sum())
 # The low-rank nuclear norm is the exact covariance-square-root trace.
 assert dm+dc>=-1e-7,('negative FID beyond roundoff',dm,dc)
 return dict(FID=max(0.,dm+dc),D_mu=dm,D_cov=dc,frames=len(a),ddof=1,GT_hash=hashlib.sha256(a.tobytes()).hexdigest())
def main():
 import numpy as np
 from scipy.linalg import svdvals
 from scipy.interpolate import PchipInterpolator
 os.environ['MPLCONFIGDIR']=str(R/'cache/matplotlib')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 records=[load(p)['result'] for p in sorted((R/'results/points').glob('*.json'))];csvwrite(R/'results/per_video_all_q.csv',records);pooled=[];pooled_p=[];equal=[];grids=[]
 manifests=load(R/'eval_manifest.json');methods=['A_original','B_rgb_P','C_latent_P'];qs=load(R/'protocol.json')['q_grid']
 expected_keys={(d,v,m,q) for d,vs in manifests.items() for v in range(len(vs)) for m in methods for q in qs}
 actual_keys=[(r['dataset'],r['video'],r['method'],r['q']) for r in records]
 assert len(actual_keys)==len(set(actual_keys)) and set(actual_keys)==expected_keys,'missing, duplicate or unexpected evaluation records'
 gt_identity={};frame_rows=[];transition_rows=[]
 import csv
 for r in records:
  v=manifests[r['dataset']][r['video']];identity={k:r[k] for k in ('dataset','video','name','method','q')}
  with open(r['per_frame'],newline='') as f:rr=list(csv.DictReader(f))
  with open(r['per_transitions'],newline='') as f:tt=list(csv.DictReader(f))
  assert [int(x['frame']) for x in rr]==list(range(64))
  assert [(int(x['from_frame']),int(x['to_frame'])) for x in tt]==[(i,i+1) for i in range(63)]
  frame_rows.extend({**identity,**x,'source_frame_index':v['source_frame_indices'][int(x['frame'])]} for x in rr)
  transition_rows.extend({**identity,**x} for x in tt)
  with np.load(r['feature']) as z:
   assert z['real'].shape==z['reconstruction'].shape and len(z['real'])==64
   h=hashlib.sha256(z['real'].tobytes()).hexdigest();key=(r['dataset'],r['video'])
   assert gt_identity.setdefault(key,h)==h,('GT features differ across method/q',key)
 csvwrite(R/'results/per_frame_metrics.csv',frame_rows);csvwrite(R/'results/transitions.csv',transition_rows)
 for d,vs in manifests.items():
  for m in methods:
   for q in qs:
    points=[r for r in records if r['dataset']==d and r['method']==m and r['q']==q];assert len(points)==len(vs)
    aa=[];bb=[];ap=[];bp=[]
    for r in sorted(points,key=lambda x:x['video']):
     with np.load(r['feature']) as z:
      a=z['real'].astype(np.float64);b=z['reconstruction'].astype(np.float64);aa.append(a);bb.append(b);ap.append(a[1:]);bp.append(b[1:])
    common=dict(dataset=d,method=m,q=q,bpp=float(np.mean([x['bpp'] for x in points])),rate_type='real_local_nonofficial_rans',rate_axis='equal-video mean total-stream actual bpp',videos=len(points),aggregation='fixed frame pool; not mean of video FID')
    pooled.append(dict(**common,frame_scope='all_frame',**fid_components(np.concatenate(aa),np.concatenate(bb))))
    pooled_p.append(dict(**common,frame_scope='P_only',**fid_components(np.concatenate(ap),np.concatenate(bp))))
  dest=R/'results'/d;dest.mkdir(parents=True,exist_ok=True)
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   fig,ax=plt.subplots(figsize=(6,4))
   for m in methods:
    curve=[r for r in pooled if r['dataset']==d and r['method']==m] if metric=='FID' else [dict(bpp=np.mean([r['bpp'] for r in records if r['dataset']==d and r['method']==m and r['q']==q]),**{metric:np.mean([r[metric] for r in records if r['dataset']==d and r['method']==m and r['q']==q])}) for q in qs]
    curve=sorted(curve,key=lambda x:x['bpp']);ax.plot([r['bpp'] for r in curve],[r[metric] for r in curve],marker='o',label=m)
   ax.set_xlabel('Bitrate (actual bpp; local GLC rANS)');ax.set_ylabel(metric);ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(dest/f'Rate_{metric}.png',dpi=160);plt.close(fig)
  for video in list(range(len(vs)))+['pooled']:
   for metric in (('FID','FID_P_only') if video=='pooled' else ('LPIPS','DISTS','FloLPIPS','LPIPS_P_only','DISTS_P_only')):
    source=pooled_p if metric=='FID_P_only' else pooled if video=='pooled' else records
    field='FID' if metric=='FID_P_only' else metric
    curves={m:[r for r in source if r['dataset']==d and r['method']==m and (video=='pooled' or r['video']==video)] for m in methods};lo=max(min(r['bpp'] for r in c) for c in curves.values());hi=min(max(r['bpp'] for r in c) for c in curves.values());values=None
    if hi>lo and all(len({r['bpp'] for r in c})==10 for c in curves.values()):
     grid=np.linspace(np.log(lo),np.log(hi),100);values={}
     for m,c in curves.items():
      c=sorted(c,key=lambda x:x['bpp']);values[m]=PchipInterpolator(np.log([r['bpp'] for r in c]),[r[field] for r in c],extrapolate=False)(grid)
      assert np.isfinite(values[m]).all()
      grids.extend(dict(dataset=d,video=video,metric=metric,method=m,grid_index=i,bpp=float(np.exp(x)),value=float(y),common_bpp_min=lo,common_bpp_max=hi) for i,(x,y) in enumerate(zip(grid,values[m])))
    for m,ref in [('B_rgb_P','A_original'),('C_latent_P','A_original'),('C_latent_P','B_rgb_P')]:
     row=dict(dataset=d,video=video,metric=metric,method=m,reference=ref,common_bpp_min=lo,common_bpp_max=hi,rate_type='real_local_nonofficial_rans',grid='100 uniform ln(bpp)',delta_definition='method-reference',status='available' if values is not None else 'unavailable')
     if values is not None:row.update(method_mean=float(values[m].mean()),reference_mean=float(values[ref].mean()),delta=float((values[m]-values[ref]).mean()),better_fraction=float((values[m]<values[ref]).mean()))
     equal.append(row)
 sensitivity=[]
 if 'mcljcv' in manifests:
  keep={i for i,v in enumerate(manifests['mcljcv']) if 'videoSRC17' not in v['name']}
  assert len(keep)==29
  for m in methods:
   for q in qs:
    points=sorted([r for r in records if r['dataset']=='mcljcv' and r['method']==m and r['q']==q and r['video'] in keep],key=lambda r:r['video']);aa=[];bb=[]
    for r in points:
     with np.load(r['feature']) as z:aa.append(z['real'].astype(np.float64));bb.append(z['reconstruction'].astype(np.float64))
    a=np.concatenate(aa);b=np.concatenate(bb);ac=a-a.mean(0);bc=b-b.mean(0);dm=float(((a.mean(0)-b.mean(0))**2).sum());dc=float((ac**2).sum()/(len(a)-1)+(bc**2).sum()/(len(b)-1)-2*svdvals(ac@bc.T/(len(a)-1)).sum())
    sensitivity.append(dict(dataset='mcljcv29_sensitivity',method=m,q=q,bpp=float(np.mean([p['bpp'] for p in points])),FID=dm+dc,D_mu=dm,D_cov=dc,frames=len(a),ddof=1,independent_dataset=False,additional_encodes=0))
 csvwrite(R/'results/pooled_fid.csv',pooled);csvwrite(R/'results/pooled_fid_P_only.csv',pooled_p);csvwrite(R/'results/mcl29_sensitivity_pooled_fid.csv',sensitivity);csvwrite(R/'results/equal_rate_per_sequence.csv',equal);csvwrite(R/'results/common_rate_grid.csv',grids)
 sameq=[]
 for d,vs in manifests.items():
  for m in methods:
   for q in qs:
    points=[r for r in records if r['dataset']==d and r['method']==m and r['q']==q]
    row=dict(dataset=d,method=m,q=q,videos=len(points),frames=len(points)*64,rate_axis='actual bpp; equal-video mean',rate_type='real_local_nonofficial_rans',aggregation='video metrics equal-video; FID fixed pooled frame features')
    row.update({k:float(np.mean([r[k] for r in points])) for k in ('bpp','LPIPS','DISTS','FloLPIPS','LPIPS_P_only','DISTS_P_only')})
    row['FID']=next(p['FID'] for p in pooled if p['dataset']==d and p['method']==m and p['q']==q)
    row['FID_P_only']=next(p['FID'] for p in pooled_p if p['dataset']==d and p['method']==m and p['q']==q)
    sameq.append(row)
 csvwrite(R/'results/all_q_summary.csv',sameq)
 summary=[]
 for key in {(r['dataset'],r['metric'],r['method'],r['reference']) for r in equal}:
  rs=[r for r in equal if (r['dataset'],r['metric'],r['method'],r['reference'])==key];good=[r for r in rs if r['status']=='available'];row=dict(zip(('dataset','metric','method','reference'),key));row.update(covered=len(good),expected=len(rs),aggregation='equal-video weights; FID fixed pooled',status='available' if len(good)==len(rs) else 'unavailable')
  if len(good)==len(rs):row.update(method_mean=float(np.mean([r['method_mean'] for r in rs])),reference_mean=float(np.mean([r['reference_mean'] for r in rs])),delta=float(np.mean([r['delta'] for r in rs])))
  summary.append(row)
 csvwrite(R/'results/equal_rate_summary.csv',summary)
 for d,vs in manifests.items():
  from PIL import Image
  from data import Frames
  f=Frames(vs[0]);dest=R/'previews'/d;dest.mkdir(parents=True,exist_ok=True)
  for i in (16,32,63):Image.fromarray(f.array(i)).save(dest/f'GT_{i}.png')
  f.close()
  for i in (16,32,63):
   from PIL import ImageDraw
   names=['GT']+methods;canvas=Image.new('RGB',(960,4*(540+28)),color='white');draw=ImageDraw.Draw(canvas)
   for row,name in enumerate(names):
    with Image.open(dest/f'{name}_{i}.png') as im:canvas.paste(im.resize((960,540),Image.Resampling.LANCZOS),(0,row*568+28))
    draw.text((8,row*568+8),f'{name} | frame {i} | q=28',fill='black')
   canvas.save(dest/f'comparison_{i}.png')
  csvwrite(R/'results'/d/'per_video_all_q.csv',[r for r in records if r['dataset']==d])
  csvwrite(R/'results'/d/'all_q_summary.csv',[r for r in sameq if r['dataset']==d])
  csvwrite(R/'results'/d/'pooled_fid.csv',[r for r in pooled+pooled_p if r['dataset']==d])
  csvwrite(R/'results'/d/'equal_rate_per_sequence.csv',[r for r in equal if r['dataset']==d])
  csvwrite(R/'results'/d/'equal_rate_summary.csv',[r for r in summary if r['dataset']==d])
 trainrows=[]
 for m in methods[1:]:
  rows=[load(p) for p in sorted((R/f'branches/{m}/receipts').glob('*.json'))]
  assert len(rows)==1000
  (R/f'branches/{m}/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  trainrows.extend([dict(branch=m,step=r['step'],domain=r['domain'],q=r['q'],**r['loss'],gradient_preclip=r['gradient_preclip'],clip_coefficient=r['clip_coefficient'],parameter_delta=r['parameter_delta'],seconds=r['seconds'],**{k:r[k] for k in ('beta','weighted_rate','total_loss','peak_MiB','RSS_MiB','lr') if k in r}) for r in rows])
 csvwrite(R/'results/training.csv',trainrows)
 expected=sum(len(v)*30 for v in manifests.values());assert len(records)==expected
 weights_ok=all(sha(r['path'])==r['sha256'] for r in load(R/'audits/checkpoint_audit.json')['models']);source_changes=[p for p,h in load(R/'audits/source_inventory.json')['files'].items() if sha(p)!=h]
 assert weights_ok and not source_changes,('historical source or weights changed',source_changes)
 for p in (R/'results/points').glob('*.json'):
  rr=load(p);assert rr['status']=='PASS' and rr['independent_decode'] and rr['frozen_unchanged'] and all(sha(a)==h for a,h in rr['artifacts'].items())
 attempt_audits=[]
 for pattern in ('codec_pause.json','codec_retry.json','codec_first_difference.json','early_failure_*.json','eval_failure_*.json','resource_guard_*.json','driver_handoff.json','evaluation_dispatcher_handoff*.json','evaluation_resource_*.json','publication_*.json'):
  attempt_audits.extend(dict(path=str(p.relative_to(R)),sha256=sha(p)) for p in sorted((R/'audits').glob(pattern)))
 save('final_integrity.json',dict(status='PASS',updates={m:load(R/f'branches/{m}/status.json')['updates'] for m in methods[1:]},evaluation_completed=len(records),evaluation_expected=expected,independent_points=sum(r['dataset']!='uvg_validation' for r in records),diagnostic_points=sum(r['dataset']=='uvg_validation' for r in records),per_point_frames=64,per_point_transitions=63,all_frame_metric_rows=len(frame_rows),transition_rows=len(transition_rows),pooled_fid_points=len(pooled),P_only_pooled_fid_points=len(pooled_p),GT_features_identical_across_methods_and_q=True,rate_type='real_local_nonofficial_rans',source_weights_unchanged=weights_ok,source_files_unchanged=not source_changes,attempt_and_recovery_audits=attempt_audits,failed_attempts_counted_as_completed_points=False,final_receipts_all_passed=True))
 (R/'README.md').write_text('# GLC precompression transfer v1 A/B/C\n\nTraining and evaluation complete; publication status is recorded separately on the server. B/C: 1000 effective updates each. A: unchanged original GLC. Mixed U-Long/UVG training. Fixed step1000, no checkpoint selection.\n\nSee REPORT.md, final_integrity.json and results/ for raw records, all-frame and P-only metrics, pooled FID, common-rate comparisons and RD PNGs. Rate axis: actual bpp from the pre-existing local non-official GLC rANS extension, including full stream headers and I frames. UVG validation is diagnostic; MCL29 is a sensitivity subset, not another independent dataset.\n\nFinal added sender weights are in final_P/. Original GLC weights, datasets, streams, feature arrays and full optimizer/RNG recovery checkpoints remain on the server. Restore commands are in RUNBOOK.md.\n')
 text='# Execution record\n\nB/C: 1000 effective updates each. Real bytes are from the pre-existing local non-official GLC rANS extension, including GOP and frame headers. Original author pretraining data unknown. Mixed U-Long/UVG training; independent data pools and UVG diagnostic results are separate. No checkpoint selection. All-frame and P-only FID use separate fixed frame pools. FloLPIPS covers all 63 adjacent transitions (each transition targets a P frame).\n\n'
 text+=f'Evaluation: {len(records)}/{expected}; independent points: '+str(sum(r['dataset']!='uvg_validation' for r in records))+'; diagnostic points: '+str(sum(r['dataset']=='uvg_validation' for r in records))+'.\n\n'
 text+='## Fixed q=28 numerical extract\n\n| Dataset | Method | Actual mean bpp | LPIPS | DISTS | FloLPIPS | Pooled FID |\n|---|---|---:|---:|---:|---:|---:|\n'
 for r in sameq:
  if r['q']==28:text+=f"| {r['dataset']} | {r['method']} | {r['bpp']:.9g} | {r['LPIPS']:.9g} | {r['DISTS']:.9g} | {r['FloLPIPS']:.9g} | {r['FID']:.9g} |\n"
 text+='\nAll q values and raw records: results/per_video_all_q.csv and results/all_q_summary.csv. Common-rate coverage and deltas: results/equal_rate_summary.csv; fixed 100-point grids: results/common_rate_grid.csv. Per-dataset RD PNGs are in results/<dataset>/. MCL29 rows are a sensitivity subset with zero additional encodes, not an independent test pool. Source and stream checks: final_integrity.json. Fixed frame16/32/63 comparisons are in previews/<dataset>/.\n'
 text+='\nAttempt/recovery audit references and hashes are listed in final_integrity.json. Failed attempts do not count as completed points; all final receipts passed unchanged strict independent-decode checks. The retained codec mismatch/retry record does not establish a root cause.\n'
 (R/'REPORT.md').write_text(text)
if __name__=='__main__':main()
