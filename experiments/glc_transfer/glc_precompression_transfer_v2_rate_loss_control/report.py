"""Mechanical numerical tables, fixed-pool FID, RD PNGs and integrity only."""
from preflight import *
from data import load,Frames
from bootstrap import BRANCHES,V1
from evaluate import csvwrite
def main():
 import numpy as np,csv
 from concurrent.futures import ThreadPoolExecutor
 from scipy.interpolate import PchipInterpolator
 from fid_pool import point
 os.environ['MPLCONFIGDIR']=str(R/'cache/matplotlib')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image,ImageDraw
 manifest=load(R/'eval_manifest.json');methods=load(R/'protocol.json')['evaluation_models'];qs=load(R/'protocol.json')['q_grid'];receipts=[load(p) for p in sorted((R/'results/points').glob('*.json'))];records=[r['result'] for r in receipts]
 expected={(d,i,m,q) for d,vs in manifest.items() for i in range(len(vs)) for m in methods for q in qs};actual=[(r['dataset'],r['video'],r['method'],r['q']) for r in records]
 assert len(actual)==len(set(actual)) and set(actual)==expected
 gt_identity={};frames=[];transitions=[]
 for receipt in receipts:
  assert receipt['status']=='PASS' and receipt['independent_decode'] and receipt['frozen_unchanged'] and all(Path(p).exists() and sha(p)==h for p,h in receipt['artifacts'].items())
  r=receipt['result'];v=manifest[r['dataset']][r['video']];assert r['bpp']==r['bytes']*8/(64*v['height']*v['width']) and r['rate_accounting_fps']==v['rate_accounting_fps']
  with open(r['per_frame'],newline='') as f:ff=list(csv.DictReader(f))
  with open(r['per_transitions'],newline='') as f:tt=list(csv.DictReader(f))
  assert [int(x['frame']) for x in ff]==list(range(64)) and [(int(x['from_frame']),int(x['to_frame'])) for x in tt]==[(i,i+1) for i in range(63)]
  identity={k:r[k] for k in ('dataset','video','name','method','q')};frames.extend({**identity,**x,'source_frame_index':v['source_frame_indices'][int(x['frame'])]} for x in ff);transitions.extend({**identity,**x} for x in tt)
  with np.load(r['feature']) as z:
   assert z['real'].shape==z['reconstruction'].shape and len(z['real'])==64
   h=hashlib.sha256(z['real'].tobytes()).hexdigest();key=(r['dataset'],r['video']);assert gt_identity.setdefault(key,h)==h
 csvwrite(R/'results/per_video_all_q.csv',records);csvwrite(R/'results/per_frame_metrics.csv',frames);csvwrite(R/'results/transitions.csv',transitions)
 jobs=[]
 for d,vs in manifest.items():
  for m in methods:
   for q in qs:
    rr=sorted([r for r in records if (r['dataset'],r['method'],r['q'])==(d,m,q)],key=lambda r:r['video']);assert len(rr)==len(vs)
    for scope in ('all_frame','P_only'):jobs.append((d,m,q,scope,rr))
    if d=='mcljcv':
     subset=[r for r in rr if 'videoSRC17' not in r['name']];assert len(subset)==29;jobs.append(('mcljcv29_sensitivity',m,q,'all_frame',subset))
 pooled=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for r in pool.map(point,jobs):
   pooled.append(r);save('audits/report_progress.json',dict(status='RUNNING',FID_completed=len(pooled),FID_expected=len(jobs),CPU_workers=4,additional_encodes=0))
 csvwrite(R/'results/pooled_fid.csv',[r for r in pooled if r['frame_scope']=='all_frame' and r['dataset'] in manifest]);csvwrite(R/'results/pooled_fid_P_only.csv',[r for r in pooled if r['frame_scope']=='P_only']);csvwrite(R/'results/mcl29_sensitivity_pooled_fid.csv',[dict(r,independent_dataset=False,additional_encodes=0) for r in pooled if r['dataset']=='mcljcv29_sensitivity'])
 sameq=[]
 for d,vs in manifest.items():
  for m in methods:
   for q in qs:
    rr=[r for r in records if (r['dataset'],r['method'],r['q'])==(d,m,q)];row=dict(dataset=d,method=m,q=q,videos=len(rr),frames=len(rr)*64,rate_axis='actual bpp; equal-video mean',rate_type='real_local_nonofficial_rans',aggregation='video metrics equal-video; FID fixed pooled feature pool')
    row.update({k:float(np.mean([r[k] for r in rr])) for k in ('bpp','LPIPS','DISTS','FloLPIPS','LPIPS_P_only','DISTS_P_only')})
    for scope,field in [('all_frame','FID'),('P_only','FID_P_only')]:row[field]=next(r['FID'] for r in pooled if (r['dataset'],r['method'],r['q'],r['frame_scope'])==(d,m,q,scope))
    sameq.append(row)
 csvwrite(R/'results/all_q_summary.csv',sameq)
 factorial_pairs=[('H10_rate2','H00_continue'),('H01_dists025','H00_continue'),('H11_rate2_dists025','H01_dists025'),('H11_rate2_dists025','H10_rate2'),('H11_rate2_dists025','H00_continue')]
 anchor_pairs=[(m,ref) for m in BRANCHES for ref in ('A_original','v1_C_latent_P')]+[('v1_C_latent_P','A_original')]
 families=[('four_branch_common',list(BRANCHES),factorial_pairs),('all_six_common',methods,anchor_pairs)];equal=[];grids=[];effects=[]
 for d,vs in manifest.items():
  for family,models,pairs in families:
   for video in list(range(len(vs)))+['pooled']:
    metrics=('FID','FID_P_only') if video=='pooled' else ('LPIPS','DISTS','FloLPIPS','LPIPS_P_only','DISTS_P_only')
    for metric in metrics:
     source=sameq if video=='pooled' else records
     curves={m:[r for r in source if r['dataset']==d and r['method']==m and (video=='pooled' or r['video']==video)] for m in models};lo=max(min(r['bpp'] for r in c) for c in curves.values());hi=min(max(r['bpp'] for r in c) for c in curves.values());values=None
     if hi>lo and all(len(c)==10 and len({r['bpp'] for r in c})==10 for c in curves.values()):
      grid=np.linspace(np.log(lo),np.log(hi),100);values={}
      for m,c in curves.items():
       c=sorted(c,key=lambda r:r['bpp']);values[m]=PchipInterpolator(np.log([r['bpp'] for r in c]),[r[metric] for r in c],extrapolate=False)(grid);assert np.isfinite(values[m]).all()
       grids.extend(dict(dataset=d,video=video,metric=metric,comparison_family=family,common_methods=';'.join(models),method=m,grid_index=i,bpp=float(np.exp(x)),value=float(y),common_bpp_min=lo,common_bpp_max=hi) for i,(x,y) in enumerate(zip(grid,values[m])))
     for m,ref in pairs:
      row=dict(dataset=d,video=video,metric=metric,method=m,reference=ref,comparison_family=family,common_methods=';'.join(models),common_bpp_min=lo,common_bpp_max=hi,rate_axis='actual bpp',grid='100 uniform ln(bpp), PCHIP, no extrapolation',delta_definition='method-reference',status='available' if values is not None else 'unavailable')
      if values is not None:
       a=float(values[m].mean());b=float(values[ref].mean());row.update(method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,better_fraction=float((values[m]<values[ref]).mean()))
      equal.append(row)
     if values is not None and family=='four_branch_common':
      a,b,c,h=(values[m] for m in BRANCHES);effects.append(dict(dataset=d,video=video,metric=metric,comparison_family=family,common_methods=';'.join(models),common_bpp_min=lo,common_bpp_max=hi,rate_effect_at_dists05=float((b-a).mean()),rate_effect_at_dists025=float((h-c).mean()),dists_effect_at_rate1=float((c-a).mean()),dists_effect_at_rate2=float((h-b).mean()),difference_of_differences=float(((h-c)-(b-a)).mean()),definition='same four-model grid; arithmetic differences only'))
 csvwrite(R/'results/equal_rate_per_sequence.csv',equal);csvwrite(R/'results/common_rate_grid.csv',grids);csvwrite(R/'results/factorial_numeric_differences.csv',effects)
 summaries=[]
 for key in sorted({(r['dataset'],r['metric'],r['method'],r['reference'],r['comparison_family']) for r in equal}):
  rr=[r for r in equal if (r['dataset'],r['metric'],r['method'],r['reference'],r['comparison_family'])==key];good=[r for r in rr if r['status']=='available'];s=dict(zip(('dataset','metric','method','reference','comparison_family'),key));s.update(covered=len(good),expected=len(rr),status='available' if len(good)==len(rr) else 'unavailable',aggregation='equal-video weights; pooled FID is separate fixed pool',delta_definition='method-reference',common_methods=rr[0]['common_methods'])
  if len(good)==len(rr):
   a=float(np.mean([r['method_mean'] for r in rr]));b=float(np.mean([r['reference_mean'] for r in rr]));s.update(method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,mean_video_relative_change_percent=float(np.mean([r['relative_change_percent'] for r in rr])) if all(r['relative_change_percent'] is not None for r in rr) else None)
  summaries.append(s)
 csvwrite(R/'results/equal_rate_summary.csv',summaries)
 qdiff=[]
 for d in manifest:
  for q in qs:
   rr={r['method']:r for r in sameq if (r['dataset'],r['q'])==(d,q)}
   for m,ref in factorial_pairs+anchor_pairs:qdiff.append(dict(dataset=d,q=q,method=m,reference=ref,method_bpp=rr[m]['bpp'],reference_bpp=rr[ref]['bpp'],bpp_delta=rr[m]['bpp']-rr[ref]['bpp'],bpp_change_percent=100*(rr[m]['bpp']-rr[ref]['bpp'])/rr[ref]['bpp'],comparison='same q, not equal rate'))
 csvwrite(R/'results/same_q_rate_differences.csv',qdiff)
 for d,vs in manifest.items():
  dest=R/'results'/d;dest.mkdir(parents=True,exist_ok=True)
  for name,rows in [('per_video_all_q',records),('all_q_summary',sameq),('equal_rate_per_sequence',equal),('equal_rate_summary',summaries),('pooled_fid',pooled)]:csvwrite(dest/f'{name}.csv',[r for r in rows if r['dataset']==d])
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   fig,ax=plt.subplots(figsize=(8,5))
   for m in methods:
    rr=sorted([r for r in sameq if (r['dataset'],r['method'])==(d,m)],key=lambda r:r['bpp']);ax.plot([r['bpp'] for r in rr],[r[metric] for r in rr],marker='o',label=m)
   ax.set_xlabel('Bitrate (actual bpp; local nonofficial GLC rANS)');ax.set_ylabel(metric);ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/f'Rate_{metric}.png',dpi=160);plt.close(fig)
  preview=R/'previews'/d;preview.mkdir(parents=True,exist_ok=True);f=Frames(vs[0])
  for i in (16,32,63):Image.fromarray(f.array(i)).save(preview/f'GT_{i}.png')
  f.close()
  for i in (16,32,63):
   canvas=Image.new('RGB',(960,(len(methods)+1)*568),'white');draw=ImageDraw.Draw(canvas)
   for j,m in enumerate(['GT']+methods):
    with Image.open(preview/f'{m}_{i}.png') as im:canvas.paste(im.resize((960,540),Image.Resampling.LANCZOS),(0,j*568+28))
    draw.text((8,j*568+8),f'{m} | frame {i} | q28 (not equal-rate preview)',fill='black')
   canvas.save(preview/f'comparison_{i}.png')
 trainrows=[]
 for b in BRANCHES:
  rr=[load(p) for p in sorted((R/f'branches/{b}/receipts').glob('*.json'))];assert len(rr)==1000 and [r['effective_updates'] for r in rr]==list(range(1,1001))
  (R/f'branches/{b}/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr));trainrows.extend({k:v for k,v in r.items() if k not in ('full_RGB_sha256','cropped_RGB_sha256','frame_trace')} for r in rr)
 csvwrite(R/'results/training.csv',trainrows)
 assert all(sha(path)==h for path,h in load(R/'audits/source_experiment.json')['files'].items())
 assert all(sha(path)==h for path,h in load(R/'audits/v1_source_inventory.json')['files'].items())
 assert all(sha(m['path'])==m['sha256'] for m in load(R/'audits/checkpoint_audit.json')['models'])
 audit_index=load(R/'audits/rate_audit_index.json');assert audit_index['paired_completed']==120 and audit_index['gradient_completed']==80 and all(sha(R/p)==h for p,h in audit_index['artifacts'].items())
 reused=sum('reuse' in r for r in receipts);new=len(receipts)-reused
 save('final_integrity.json',dict(status='PASS',updates={b:load(R/f'branches/{b}/status.json')['updates'] for b in BRANCHES},source_step=1000,new_adaptation_updates=1000,evaluation_completed=len(records),evaluation_expected=len(expected),new_evaluation_points=new,reused_evaluation_points=reused,independent_new_points=sum(r['dataset']!='uvg_validation' and r['method'] in BRANCHES for r in records),diagnostic_new_points=sum(r['dataset']=='uvg_validation' and r['method'] in BRANCHES for r in records),all_frame_metric_rows=len(frames),transition_rows=len(transitions),per_point_frames=64,per_point_transitions=63,GT_features_identical_across_methods_and_q=True,pooled_FID_all_and_P_only=True,RD_png_count=len(list((R/'results').glob('*/Rate_*.png'))),rate_backend=load(R/'audits/rate_backend_decision.json'),rate_type='real_local_nonofficial_rans',source_v1_and_codec_unchanged=True,all_final_receipts_passed=True,paired_rate_audits_completed=120,paired_rate_audits_expected=120,gradient_audits_completed=80,gradient_audits_expected=80))
 text='# Execution record\n\nFour fixed branches, each 1000 added updates from the same v1 C step1000 sender. Only latent P trained. Original GLC frozen. No checkpoint selection. Evaluation uses unchanged local nonofficial real rANS bytes including complete headers.\n\n'
 text+=f'Completed {len(records)}/{len(expected)} evaluation points; new {new}, reused {reused}. Rate backend: '+load(R/'audits/rate_backend_decision.json')['backend']+'.\n\n'
 text+='Common-rate comparison families are explicitly recorded: four-branch factorial common interval and all-six-model anchor interval. All use 100 ln(bpp) points and no extrapolation. Video metrics use equal-video weights; FID uses fixed pooled features. MCL29 is sensitivity only; UVG validation is diagnostic only.\n\n'
 text+='Paired training-only rate facts and separately identifiable header costs: audits/paired_rate_summary.csv. Unisolated rANS flush/alignment remains combined. Gradient and local response records: audits/component_gradients/. Backend evidence and unchanged/changed beta: audits/rate_backend_decision.json and audits/rate_freeze.json. No unsupported rate-error cause is asserted.\n\n'
 decision=load(R/'audits/rate_backend_decision.json');text+=('The fixed audit reproduced the probability-table discrepancy specified in its mechanical decision record.\n\n' if decision['changed'] else 'A structural probability-table mismatch was not demonstrated under the recorded mechanical audit rule; original beta and official Gaussian rate estimator retained.\n\n');text+='All four branches use the v2-only categorical forward alignment documented in audits/reference_alignment_fix.json. H00 is on this common repaired numerical forward base, not an unchanged v1 numerical-training-path continuation. Evaluation codec and v1 source weights are unchanged.\n\n'
 text+='See results/ for all raw numbers, same-q actual-rate differences, equal-rate tables, pooled FID, factorial arithmetic differences and RD PNGs. Fixed q28 previews are not equal-rate previews. Full recovery checkpoints, original weights, videos, bitstreams and feature arrays remain on the server.\n'
 (R/'REPORT.md').write_text(text);(R/'README.md').write_text('# GLC precompression transfer v2 rate/loss control\n\nTraining and evaluation complete; publication is separately verified. See REPORT.md, final_integrity.json, results/ and RUNBOOK.md. Four branches: 1000 added updates each. Evaluation: '+str(len(records))+' points, including reused A/v1 C baselines.\n')
 save('audits/report_progress.json',dict(status='PASS',FID_completed=len(jobs),FID_expected=len(jobs),formal_points_added=0));print('REPORT PASS',len(records),flush=True)
if __name__=='__main__':main()
