"""Mechanical six-receiver union and separate A/B common-rate summaries."""
import csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500'
sys.path.insert(0,str(OLD))
from aggregate37 import pool
from intervene41 import METHODS,atomic_csv,atomic_json,sha

SPATIAL=('LPIPS','DISTS','FloLPIPS')
ALL=(*SPATIAL,'FID','D_mu','D_cov')
BNAMES={'A00':'O_original','G_fullLoss_Gonly_500':'G_fullLoss_Gonly_500','F_BG500':'F_BG500'}
BCOMP=(('Gonly_vs_O','G_fullLoss_Gonly_500','A00'),('BG_vs_O','F_BG500','A00'),
       ('Gonly_vs_BG','G_fullLoss_Gonly_500','F_BG500'))

def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def tagged_point(p,method,source):
 z=dict(p);z['method']=method;z['method_full']=METHODS[method][0] if method in METHODS else BNAMES[method]
 z['source_result']=source;z['actual_bpp']=float(z['actual_bpp']);z['QP']=int(z['QP']);z['video']=int(z['video'])
 for k in SPATIAL:z[k]=float(z[k])
 z['reuse_bitstream']=True;z['new_real_encodes']=0
 return z
def interval(curves,rate):
 lo=max(min(float(p[rate]) for p in c) for c in curves.values())
 hi=min(max(float(p[rate]) for p in c) for c in curves.values())
 return lo,hi
def curve(c,rate,metric,grid):
 q=sorted((float(p[rate]),float(p[metric])) for p in c)
 assert len(q)==3 and q[0][0]<q[1][0]<q[2][0]
 return np.interp(grid,np.log([x for x,y in q]),[y for x,y in q])
def diffrow(level,ds,video,qp,metric,label,m,ref,a,b,lo=None,hi=None):
 return dict(level=level,dataset=ds,video=video,QP=qp,metric=metric,contrast=label,method=m,
  method_full=BNAMES.get(m,METHODS.get(m,(m,))[0]),reference=ref,
  reference_full=BNAMES.get(ref,METHODS.get(ref,(ref,))[0]),method_value=float(a),reference_value=float(b),
  delta=float(a-b),relative_percent=100*float(a-b)/float(b) if b else None,
  lower_is_better=True,common_bpp_min=lo,common_bpp_max=hi)

def main():
 manifest=json.loads((OLD/'eval_manifest.json').read_text())
 tasks=json.loads((HERE/'evaluation/task_manifest.json').read_text())
 a=read(HERE/'results/A_all_rd_points.csv');assert len(a)==156
 points=[tagged_point(x,x['method'],'v6.41 task A verified endpoint or cross inference') for x in a]
 receipts=[];missing=[]
 for entry in tasks:
  t=entry['task'];key=f"{t['dataset']}_{t['video']}_q{t['qp']}"
  p=HERE/'tasks'/f'gonly_{key}.json'
  if not p.exists():missing.append(key);continue
  r=json.loads(p.read_text());assert r['status']=='PASS' and r['task']==t
  assert r['new_optimizer_updates']==r['new_real_encodes']==0 and r['adagn_path_checks']==315
  assert sha(r['result']['feature_path'])==r['result']['feature_sha256']
  receipts.append(r);points.append(tagged_point(r['result'],'G_fullLoss_Gonly_500',str(p)))
  old=json.loads((OLD/'tasks'/f'replay_{key}.json').read_text())
  assert old['status']=='PASS' and old['result']['F_fullLoss_noP_500']['bitstream_sha256']==r['result']['bitstream_sha256']
  points.append(tagged_point(old['result']['F_fullLoss_noP_500'],'F_BG500',str(OLD/'tasks'/f'replay_{key}.json')))
 assert len({(p['dataset'],p['video'],p['QP'],p['method']) for p in points})==len(points)
 for p in points:assert sha(p['feature_path'])==p['feature_sha256'] and sha(p['bitstream_path'])==p['bitstream_sha256']
 assert all(len({p['bitstream_sha256'] for p in points if (p['dataset'],p['video'],p['QP'])==(t['task']['dataset'],t['task']['video'],t['task']['qp'])})==1 for t in tasks)
 atomic_csv(HERE/'results/all_rd_points.csv',points)
 pooled=[]
 for ds,videos in manifest.items():
  for method in (*METHODS,'G_fullLoss_Gonly_500','F_BG500'):
   for qp in (0,5,9):
    rows=[p for p in points if (p['dataset'],p['method'],p['QP'])==(ds,method,qp)]
    if len(rows)==len(videos):
     fid=pool(rows);assert abs(fid['FID']-fid['D_mu']-fid['D_cov'])<=1e-4
     assert int(fid['samples'])==64*len(videos)
     pooled.append(dict(dataset=ds,method=method,method_full=BNAMES.get(method,METHODS.get(method,(method,))[0]),QP=qp,**fid))
 assert all(len({p['GT_pool_sha256'] for p in pooled if p['dataset']==ds})<=1 for ds in manifest)
 atomic_csv(HERE/'results/pooled_fid.csv',pooled)
 idx={(p['dataset'],p['video'],p['QP'],p['method']):p for p in points}
 pair=[]
 for ds,videos in manifest.items():
  for v in videos:
   for qp in (0,5,9):
    rr={m:idx.get((ds,v['video_index'],qp,m)) for m in BNAMES}
    if not all(rr.values()):continue
    for metric in SPATIAL:
     for label,m,ref in BCOMP:pair.append(dict(diffrow('video_QP',ds,v['video_index'],qp,metric,label,m,ref,rr[m][metric],rr[ref][metric]),panel='B'))
  for qp in (0,5,9):
   rr={m:next((x for x in pooled if (x['dataset'],x['method'],x['QP'])==(ds,m,qp)),None) for m in BNAMES}
   if all(rr.values()):
    for metric in ('FID','D_mu','D_cov'):
     for label,m,ref in BCOMP:
      pair.append(dict(diffrow('dataset_QP_pooled',ds,'ALL',qp,metric,label,m,ref,rr[m][metric],rr[ref][metric]),panel='B'))
 a_pairs=[dict(x,panel='A',contrast=x['effect']) for x in read(HERE/'results/A_same_stream_pairs.csv')]
 atomic_csv(HERE/'results/same_stream_pairs.csv',pair+a_pairs)
 pervideo=[];summary=[];coverage=[]
 for panel,methods in [('A',tuple(METHODS)),('B',tuple(BNAMES))]:
  for ds,videos in manifest.items():
   sub=[]
   for v in videos:
    curves={m:[p for p in points if (p['dataset'],p['video'],p['method'])==(ds,v['video_index'],m)] for m in methods}
    if any(len(c)!=3 for c in curves.values()):continue
    lo,hi=interval(curves,'actual_bpp');coverage.append(dict(panel=panel,dataset=ds,video=v['video_index'],lo=lo,hi=hi,available=hi>lo))
    if hi<=lo:continue
    grid=np.linspace(math.log(lo),math.log(hi),20)
    for metric in SPATIAL:
     ys={m:curve(c,'actual_bpp',metric,grid) for m,c in curves.items()}
     contrasts=BCOMP if panel=='B' else ((name,x,y) for name,x,y in
       [('mod_old_body','A01','A00'),('mod_new_body','A11','A10'),('body_old_mod','A10','A00'),
        ('body_new_mod','A11','A01'),('full_generator','A11','A00')])
     for label,m,ref in contrasts:
      z=diffrow('video_equal_rate',ds,v['video_index'],'ALL',metric,label,m,ref,float(ys[m].mean()),float(ys[ref].mean()),lo,hi)
      z['panel']=panel;pervideo.append(z);sub.append(z)
   for metric in SPATIAL:
    for label in sorted({r['contrast'] for r in sub if r['metric']==metric}):
     rr=[r for r in sub if r['metric']==metric and r['contrast']==label]
     if rr:
      m,ref=rr[0]['method'],rr[0]['reference'];z=diffrow('dataset_video_equal_weight',ds,'ALL','ALL',metric,label,m,ref,
       float(np.mean([r['method_value'] for r in rr])),float(np.mean([r['reference_value'] for r in rr])))
      z['panel']=panel;z['videos']=len(rr);z['grid_points_per_video']=20;summary.append(z)
   curves={m:[p for p in pooled if (p['dataset'],p['method'])==(ds,m)] for m in methods}
   if all(len(c)==3 for c in curves.values()):
    lo,hi=interval(curves,'mean_actual_bpp')
    if hi>lo:
     grid=np.linspace(math.log(lo),math.log(hi),20)
     for metric in ('FID','D_mu','D_cov'):
      ys={m:curve(c,'mean_actual_bpp',metric,grid) for m,c in curves.items()}
      contrasts=BCOMP if panel=='B' else ((name,x,y) for name,x,y in
       [('mod_old_body','A01','A00'),('mod_new_body','A11','A10'),('body_old_mod','A10','A00'),
        ('body_new_mod','A11','A01'),('full_generator','A11','A00')])
      for label,m,ref in contrasts:
       z=diffrow('dataset_pooled_equal_rate',ds,'ALL','ALL',metric,label,m,ref,float(ys[m].mean()),float(ys[ref].mean()),lo,hi)
       z['panel']=panel;z['videos']=len(videos);z['samples']=64*len(videos);summary.append(z)
 atomic_csv(HERE/'results/equal_rate_per_video.csv',pervideo)
 atomic_csv(HERE/'results/equal_rate_summary.csv',summary)
 atomic_csv(HERE/'results/B_Gonly_vs_BG_summary.csv',[r for r in summary if r['panel']=='B'])
 wins=[]
 for ds in manifest:
  for metric in SPATIAL:
   for label,_,_ in BCOMP:
    q=[x for x in pervideo if (x['panel'],x['dataset'],x['metric'],x['contrast'])==('B',ds,metric,label)]
    wins.append(dict(dataset=ds,metric=metric,contrast=label,videos=len(q),
     method_better=sum(x['delta']<0 for x in q),reference_better=sum(x['delta']>0 for x in q),ties=sum(x['delta']==0 for x in q)))
 atomic_csv(HERE/'results/B_video_win_counts.csv',wins)
 atomic_csv(HERE/'results/rate_coverage.csv',coverage)
 trainpath=HERE/'branches/G_fullLoss_Gonly_500/training_integrity.json';train=json.loads(trainpath.read_text()) if trainpath.exists() else {}
 A=json.loads((HERE/'A_integrity.json').read_text())
 full=(len(points)==234 and len(pooled)==54 and len(receipts)==39 and not missing and
       train.get('status')=='PASS' and train.get('committed_updates')==500 and A.get('status')=='PASS')
 integrity=dict(status='PASS' if full else 'PARTIAL',A_RD_completed=A['RD_completed'],A_RD_planned=156,
  A_FID_completed=A['FID_completed'],A_FID_planned=36,B_RD_completed=sum(p['method'] in BNAMES for p in points),B_RD_planned=117,
  B_FID_completed=sum(p['method'] in BNAMES for p in pooled),B_FID_planned=27,
  unique_receiver_configs=6,RD_completed=len(points),RD_planned=234,FID_completed=len(pooled),FID_planned=54,
  reused_RANS_streams=39,new_real_encodes=0,G_only_committed_updates=train.get('committed_updates',0),
  A_endpoint_bitwise=A['endpoint_bitwise'],A_path_checks=A['AdaGN_path_checks'],
  G_only_adagn_path_checks=315*len(receipts),FID_sample_counts={ds:64*len(v) for ds,v in manifest.items()},
  GT_pool_consistent=all(len({p['GT_pool_sha256'] for p in pooled if p['dataset']==ds})<=1 for ds in manifest),
  missing_G_only_tasks=missing)
 atomic_json(HERE/'final_integrity.json',integrity)
 lines=['# v6.41 generator parameter and G-only matched control','',
  f"Execution **{integrity['status']}**. A {A['RD_completed']}/156 RD, {A['FID_completed']}/36 pooled FID; B {integrity['B_RD_completed']}/117 RD, {integrity['B_FID_completed']}/27 pooled FID. Global unique {len(points)}/234 RD and {len(pooled)}/54 FID. Reused streams 39; new encodes 0. G-only committed updates {integrity['G_only_committed_updates']}/500.",'',
  'A00 is original bridge+G0; A11 is original bridge+BG-trained generator. F_BG500 uses the BG-trained bridge and generator. All A inputs use the same z0. Negative quality delta is better at identical actual bitrate.','',
  '## Prespecified equal-rate contrasts','',
  '|Panel|Dataset|Contrast|Metric|Method|Reference|Absolute delta|Relative %|','|---|---|---|---|---|---|---:|---:|']
 for x in summary:
  if x['metric'] in (*SPATIAL,'FID'):
   lines.append(f"|{x['panel']}|{x['dataset']}|{x['contrast']}|{x['metric']}|{x['method']}|{x['reference']}|{x['delta']:.6f}|{x['relative_percent']:.3f}|")
 lines+=['','See `results/all_rd_points.csv` for measured QP anchors, `results/pooled_fid.csv` for pooled Inception FID and mean/covariance terms, and `results/rate_coverage.csv` for common intervals. The 20 interpolation points are diagnostic, not independent encoded samples. A mixed-weight degradation can reflect incompatibility. G-only vs BG includes the difference in trainable set and optimizer trajectory under the same rule; this single seed, 13-video, three-QP comparison does not establish general behavior or convergence.','']
 (HERE/'REPORT.md').write_text('\n'.join(lines))
 brief=lines[:10]
 for x in summary:
  if x['metric'] in (*SPATIAL,'FID') and x['contrast'] in ('Gonly_vs_BG','full_generator'):
   brief.append(f"|{x['panel']}|{x['dataset']}|{x['contrast']}|{x['metric']}|{x['method']}|{x['reference']}|{x['delta']:.6f}|{x['relative_percent']:.3f}|")
 brief.append('')
 (HERE/'RESULTS_SUMMARY.md').write_text('\n'.join(brief)+'\n')
 print('AGGREGATE41',integrity['status'],len(points),len(pooled),flush=True)

if __name__=='__main__':main()
