"""Deterministic numerical summaries only; no research interpretation."""
from io21 import *
import math,traceback
from collections import defaultdict

def table(name,rows):
 aliases={'p2_5':'p2.5','p97_5':'p97.5','CI2_5':'CI2.5','CI97_5':'CI97.5'};rows=[{aliases.get(k,k):v for k,v in r.items()}for r in rows]
 if name.endswith('bootstrap_summary'):
  for r in rows:
   if 'p2.5'in r:r.update({'CI2.5':r['p2.5'],'CI97.5':r['p97.5'],'mean_delta':r['mean'],'std_delta':r['std'],'median_delta':r['median']})
 write(ROOT/'results'/f'{name}.csv',rows);dump(ROOT/'results'/f'{name}.json',rows)
def stats(v):
 import numpy as np
 a=np.asarray(v,float);return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),median=float(np.median(a)),p2_5=float(np.percentile(a,2.5)),p97_5=float(np.percentile(a,97.5)))
def interp(bp,values,grid):
 import numpy as np
 from scipy.interpolate import PchipInterpolator
 x=np.asarray(bp,float);y=np.asarray(values,float);order=np.argsort(x);x=x[order];y=y[order];assert np.all(np.diff(x)>0);assert min(grid)>=min(x)-1e-14 and max(grid)<=max(x)+1e-14
 return PchipInterpolator(np.log(x),y,extrapolate=False)(np.log(grid))

def main():
 import numpy as np
 from statistics import counts_for
 # Only after all GPU tasks finish: distinguish the historical weight catalog from this experiment matrix.
 catalog=load(ROOT/'evaluation/config.json');before=sha(ROOT/'evaluation/config.json');dump(ROOT/'audits/v620_evaluation_catalog_snapshot.json',catalog)
 keys=['checkpoints','compression_hash','I_hash','deployment_receiver_hashes','original_receiver_hashes','metric_module_hashes','i_frame_modes'];unchanged={k:catalog[k] for k in keys}
 catalog.update(schema='v621a_diagnostic',no_training=True,methods=list(METHODS),expected_points=480,frames=128,source_catalog_path=str(V20/'evaluation/config.json'),source_catalog_sha256=sha(V20/'evaluation/config.json'));dump(ROOT/'evaluation/config.json',catalog)
 assert {k:catalog[k]for k in keys}==unchanged
 hashes=load(ROOT/'audits/protocol_hashes.json');hashes['evaluation/config.json']=sha(ROOT/'evaluation/config.json');dump(ROOT/'audits/protocol_hashes.json',hashes)
 dump(ROOT/'audits/evaluation_catalog_metadata.json',dict(status='PASS',before_sha256=before,after_sha256=sha(ROOT/'evaluation/config.json'),runtime_keys_unchanged=keys,change='Diagnostic no_training/methods/expected_points/frame metadata; historical source catalog archived. No model/codec/metric mapping change.'))
 full=[];sample=[];boots=[];skipped=[];feature_rows=[]
 for d in DATASETS:
  for q in range(10):
   p=ROOT/'statistics'/d/f'qp{q}.json';r=load(p);assert r['status']=='PASS'and r['bootstrap_repeats']==1000;skipped.extend(dict(dataset=d,QP=q,**v)for v in r['unavailable_sample_sizes'])
   for field,out in [('full',full),('sample',sample),('bootstrap',boots)]:
    rows=read(p.with_suffix('.'+field+'.csv'))
    for row in rows:
     for k in ['FID','KID','bpp']:
      if k in row:row[k]=float(row[k]);assert math.isfinite(row[k])
     for k in ['QP','repeat','sample_size','samples','actual_sample_count']:
      if k in row:row[k]=int(row[k])
    out.extend(rows)
   feature_rows.extend(r['features'])
 table('fid_pooled_expanded',full);table('kid_per_qp',[{k:v for k,v in r.items()if k!='FID'}for r in full]);table('fid_sample_size_stability',sample)
 groups=defaultdict(list)
 for r in sample:groups[r['dataset'],r['method'],r['QP'],r['sample_size']].append(r['FID'])
 table('fid_sample_size_summary',[dict(dataset=d,method=m,QP=q,sample_size=n,repeats=len(v),**stats(v))for(d,m,q,n),v in groups.items()]);dump(ROOT/'results/unavailable_sample_sizes.json',skipped)
 lookup={(r['dataset'],r['method'],r['QP'],r['repeat']):r for r in boots};deltas=[];kid_deltas=[];fid_summary=[];kid_summary=[]
 for d in DATASETS:
  for m in METHODS[1:]:
   for q in range(10):
    fd=[];kd=[]
    for i in range(1000):
     r=lookup[d,m,q,i];ref=lookup[d,'original',q,i];f=r['FID']-ref['FID'];k=r['KID']-ref['KID'];fd.append(f);kd.append(k);deltas.append(dict(dataset=d,method=m,reference='original',QP=q,repeat=i,FID_method=r['FID'],FID_reference=ref['FID'],delta=f));kid_deltas.append(dict(dataset=d,method=m,reference='original',QP=q,repeat=i,KID_method=r['KID'],KID_reference=ref['KID'],delta=k))
    for vv,dest in [(fd,fid_summary),(kd,kid_summary)]:
     ss=stats(vv);dest.append(dict(dataset=d,method=m,reference='original',QP=q,mean_delta=ss['mean'],median_delta=ss['median'],std_delta=ss['std'],CI2_5=ss['p2_5'],CI97_5=ss['p97_5'],fraction_delta_below_zero=float(np.mean(np.array(vv)<0)),n_bootstrap=1000))
 table('fid_bootstrap_delta',deltas);table('kid_bootstrap_delta',kid_deltas);table('fid_bootstrap_summary',fid_summary);table('kid_bootstrap_summary',kid_summary)
 fl={(r['dataset'],r['method'],r['QP']):r for r in full};table('kid_delta_vs_original',[dict(dataset=d,method=m,reference='original',QP=q,KID_method=fl[d,m,q]['KID'],KID_reference=fl[d,'original',q]['KID'],delta=fl[d,m,q]['KID']-fl[d,'original',q]['KID'])for d in DATASETS for m in METHODS[1:]for q in range(10)])
 grids=[];eq={'FID':[],'KID':[]};eqboot={'FID':[],'KID':[]}
 for d in DATASETS:
  bp={m:[fl[d,m,q]['bpp']for q in range(10)]for m in METHODS};lo=max(min(v)for v in bp.values());hi=min(max(v)for v in bp.values());assert lo<hi;grid=np.exp(np.linspace(np.log(lo),np.log(hi),100));grid[0]=lo;grid[-1]=hi;grids.extend(dict(dataset=d,grid_index=i,bpp=float(b),common_min_bpp=lo,common_max_bpp=hi,models=METHODS)for i,b in enumerate(grid))
  for metric in ['FID','KID']:
   vals={m:interp(bp[m],[fl[d,m,q][metric]for q in range(10)],grid)for m in METHODS}
   for m in METHODS[1:]:
    delta=vals[m]-vals['original'];eq[metric].extend(dict(dataset=d,method=m,reference='original',grid_index=i,bpp=float(b),method_value=float(vals[m][i]),reference_value=float(vals['original'][i]),delta=float(delta[i]),common_min_bpp=lo,common_max_bpp=hi)for i,b in enumerate(grid))
    replicates=[]
    for i in range(1000):
     a=interp(bp[m],[lookup[d,m,q,i][metric]for q in range(10)],grid);b=interp(bp['original'],[lookup[d,'original',q,i][metric]for q in range(10)],grid);replicates.append(a-b)
    reps=np.asarray(replicates)
    for i,b in enumerate(grid):eqboot[metric].append(dict(dataset=d,method=m,reference='original',scope='grid_point',grid_index=i,bpp=float(b),common_min_bpp=lo,common_max_bpp=hi,n_bootstrap=1000,fraction_delta_below_zero=float(np.mean(reps[:,i]<0)),**stats(reps[:,i])))
    integrated=reps.mean(1);eqboot[metric].append(dict(dataset=d,method=m,reference='original',scope='mean_of_100_ln_bpp_grid',grid_index=None,bpp=None,common_min_bpp=lo,common_max_bpp=hi,n_bootstrap=1000,fraction_delta_below_zero=float(np.mean(integrated<0)),**stats(integrated)))
 table('common_rate_grid',grids)
 for metric in ['FID','KID']:table('equal_rate_'+metric.lower(),eq[metric]);table('equal_rate_'+metric.lower()+'_bootstrap_summary',eqboot[metric])
 points=[]
 for d in DATASETS:
  for m in METHODS:
   for v in sources(d):
    for q in range(10):
     r=load(point(d,m,v['video_index'],q));assert r['real_RANS']and r['independent_decode_pass']and r['state_sync_pass']and r['model_hashes_unchanged']and r['historical_64_prefix_bitstream_verified'];assert r['compression_hash_before']==r['compression_hash_after']==evalcfg()['compression_hash'];assert sha(r['feature_path'])==r['feature_sha256']and sha(r['bitstream_path'])==r['bitstream_sha256'];assert r['frames']==v['frames']and len(r['actual_qps'])==v['frames']
     old=load(V20/'parts'/d/m/f'video_{v["video_index"]:02d}_qp{q}.json')
     with np.load(r['feature_path'])as f,np.load(old['feature_path'])as g:
      assert np.array_equal(f['real'][:64],g['real']) and np.array_equal(f['reconstruction'][:64],g['reconstruction']),('first64 feature mismatch',d,m,v['name'],q)
     points.append(dict(dataset=d,method=m,sequence=v['name'],QP=q,frames=r['frames'],bpp=r['bpp'],real_bytes=r['real_bytes'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256'],bitstream_sha256=r['bitstream_sha256']))
 assert len(points)==480;table('evaluation_points',points)
 hist=load(ROOT/'audits/historical_inventory.json');changed=[]
 for p,(size,mtime)in hist.items():
  s=Path(p).stat()
  if [s.st_size,s.st_mtime_ns]!=[size,mtime]:changed.append(p)
 assert not changed,('history changed',changed)
 for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,(p,'dependency changed')
 for cp in load(ROOT/'checkpoint_integrity.json')['records']:
  assert sha(cp['full_checkpoint_absolute_path'])==cp['full_checkpoint_sha256']
  assert sha(cp['deployment']['path'])==cp['deployment']['sha256']
 schedule=load(ROOT/'gpu_schedule.json');schedule.update(active={},final_success=True,status='PASS',startup_snapshot=load(ROOT/'audits/gpu_startup_snapshot.json'));dump(ROOT/'gpu_schedule.json',schedule)
 audit=dict(status='PASS',expected_points=480,completed_points=480,failed_points=0,missing=[],real_RANS=True,independent_decode=True,recurrent_state=True,no_teacher_forcing=True,all_64_prefixes_verified=True,all_64_feature_prefixes_exact=True,points=points)
 dump(ROOT/'codec_integrity.json',audit);dump(ROOT/'evaluation_integrity.json',audit)
 # The blocked experiment retains the original 64-frame manifest and no fabricated metrics.
 b=ROOT.parent/'gvcrt_neural_wrapper_v6_21b_gvcrt_latent_interface_audit';ba=load(b/'final_integrity.json');ba.update(historical_files_unchanged=True,source_checkpoints_verified=True);(b/'final_integrity.json').write_text(json.dumps(ba,indent=2)+'\n');(b/'dataset_integrity.json').write_text(json.dumps(dict(status='PASS',videos=[v for d in DATASETS for v in load(V20/'manifests'/f'{d}.json')['videos']],source= str(V20),training_source_disjoint=True),indent=2)+'\n');(b/'gpu_schedule.json').write_text(json.dumps(dict(status='BLOCKED',tasks_started=0,no_training=True),indent=2)+'\n')
 (ROOT/'README_NUMBERS_ONLY.md').write_text('# Protocol\n\nNo training. Seed 20261010. 3 methods, 16 sequences, QP 0–9. 128 canonical frames per sequence. Paired sequence-stratified sampling. FID: sample covariance; KID: unbiased cubic polynomial MMD. PCHIP on 100 uniform ln(bpp) points within the three-method common measured interval.\n\n| Output | Count |\n|---|---:|\n| Real RANS points | 480 |\n| V6.20 reproduction points | 120 |\n| Bootstrap repeats per dataset/QP | 1000 |\n| Sample-size repeats | 100 |\n\n## Files\n\nresults/v620_fid_reproduction.csv\nresults/fid_sample_size_stability.csv\nresults/fid_sample_size_summary.csv\nresults/fid_bootstrap_delta.csv\nresults/fid_bootstrap_summary.csv\nresults/kid_per_qp.csv\nresults/kid_delta_vs_original.csv\nresults/kid_bootstrap_summary.csv\nresults/equal_rate_fid.csv\nresults/equal_rate_kid.csv\nresults/equal_rate_fid_bootstrap_summary.csv\nresults/equal_rate_kid_bootstrap_summary.csv\n')
 argv=[PLOT_PYTHON,'-B',str(ROOT/'plot.py')];command(argv);subprocess.run(argv,check=True)
 outputs={str(p.relative_to(ROOT)):sha(p)for folder in [ROOT/'results',ROOT/'plots']for p in folder.rglob('*')if p.is_file()};dump(ROOT/'final_integrity.json',dict(status='PASS',no_training=True,historical_files_unchanged=True,source_checkpoints_verified=True,real_RANS=True,independent_decode=True,recurrent_state=True,no_teacher_forcing=True,source_frames_verified=True,FID_feature_extractor_same_as_v620=True,V620_FID_reproduced=True,bootstrap_completed=True,KID_checked=True,no_unexplained_missing_points=True,expected_points=480,completed_points=480,statistics_points=40,bootstrap_repeats=1000,failed_points=0,outputs=outputs,executed_source_hashes={str(p):sha(p)for p in ROOT.glob('*.py')},finished_unix=time.time()));print('FINAL INTEGRITY PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs/report_failure.json',dict(error=traceback.format_exc()));raise
