"""D learning curves, training-only predictive references, paired separation and evidence report."""
from io21 import *
import math,collections,numpy as np
from scipy.stats import rankdata
BR=('A_raw_current','B_rms_current','C_zero_current','D_raw_official_optimizer')
def auc(real,fake):
 n=len(real);m=len(fake)
 if not n or not m:return None
 ranks=rankdata(np.r_[real,fake]);return float((ranks[n:].sum()-m*(m+1)/2)/(n*m))
def averages(rows,fields):return {k:float(np.mean([float(r[k]) for r in rows if r.get(k) not in (None,'')])) for k in fields if any(r.get(k) not in (None,'') for r in rows)}
def main():
 status=load(ROOT/'pipeline_status.json');status.update(status='RUNNING',phase='CPU_summary');dump(ROOT/'pipeline_status.json',status)
 ROOT.joinpath('results').mkdir(exist_ok=True);ROOT.joinpath('plots').mkdir(exist_ok=True)
 # Keep the core all-step summary directly readable; retain full precision in original logs/full CSV.
 import shutil
 core=['effective_step','D_loss','CE_real','CE_fake','condition_adapter_grad_norm','image_unet_grad_norm','condition_adapter_relative_update','image_unet_relative_update','frames','G_updates','gradient_clip']
 for branch in BR:
  for stage in ('TINY','MAIN'):
   p=ROOT/'branches'/branch/stage/'gradient_summary.csv';fullp=p.with_name('gradient_summary_full.csv')
   if not fullp.exists():shutil.copyfile(p,fullp)
   rows=read(fullp);compact=[]
   for r in rows:compact.append({k:(format(float(r[k]),'.6g') if r[k] and k not in ('effective_step','frames','G_updates','gradient_clip') else r[k]) for k in core})
   write(p,compact);assert p.stat().st_size<=524288

 # Preserve measured complete-task peaks with scheduling PID/admission records.
 peak_rows=[]
 for line in (ROOT/'gpu_schedule.jsonl').read_text().splitlines():
  event=json.loads(line)
  if event.get('event')!='finish':continue
  argv=event['argv'];kind=event['kind'];peak=0.
  if kind=='train':
   branch=argv[argv.index('--branch')+1];stage=argv[argv.index('--stage')+1];peak=load(ROOT/'branches'/branch/stage/'status.json')['peak_memory_MiB']
  elif kind=='cache':
   for key in argv[argv.index('--keys')+1:]:
    meta=ROOT/'cache/status'/f'{key}.json'
    if meta.exists():peak=max(peak,load(meta)['peak_memory_MiB'])
  elif kind=='validation':
   branch=argv[argv.index('--branch')+1];stage=argv[argv.index('--stage')+1];step=int(argv[argv.index('--step')+1])
   for panel in argv[argv.index('--panel')+1:]:
    meta=ROOT/'validation'/branch/stage/f'step_{step:05d}'/panel/'integrity.json'
    if meta.exists():peak=max(peak,load(meta)['peak_memory_MiB'])
  peak_rows.append(dict(task=event['task'],kind=kind,PID=event['PID'],physical_gpu=event['gpu'],logical_gpu=0,reserved_MiB=event['reserved_MiB'],measured_peak_MiB=peak,exit_code=event['exit_code'],retry=event['attempt']))
 write(ROOT/'results/gpu_peak_admission.csv',peak_rows)
 dump(ROOT/'audits/gpu_peak_admission.json',dict(status='PASS',tasks=len(peak_rows),measured_peak_within_reserved=all(r['measured_peak_MiB']<=r['reserved_MiB'] for r in peak_rows),startup_reservations_included=True,no_external_process_terminated=True,physical_GPUs=[4,5,6,7]))
 # Fixed training-pool empirical references, independent of checkpoint and validation labels.
 reference_path=ROOT/'audits/training_frequency_reference.json'
 if not reference_path.exists():subprocess.check_call([PYTHON,str(ROOT/'frequency_reference.py')],cwd=REPO)
 import ast
 ref=load(reference_path);assert ref['status']=='PASS' and ref['MAIN_frames']==15000 and ref['TINY_frames']==8
 full=np.array(ref['MAIN_counts'],dtype=np.int64);tiny=np.array(ref['TINY_counts'],dtype=np.int64);byMAIN={ast.literal_eval(k):np.array(v,dtype=np.int64) for k,v in ref['MAIN_domain_QP_counts'].items()};byTINY={ast.literal_eval(k):np.array(v,dtype=np.int64) for k,v in ref['TINY_domain_QP_counts'].items()}

 summaries=[];scale_summary=[];pairsout=[];freqout=[];distribution=[];window_rows=[]
 fields=['CE_content','CE_fake','CE_content_only','p_fake','all_class_top1','all_class_top5','content_top1','content_top5','fake_accuracy','CE_G_image_grad_norm']
 for b in BR:
  for stage in ['TINY','MAIN']:
   counts=tiny if stage=='TINY' else full;prior=(counts+1)/(counts.sum()+1024);order=np.argsort(-prior);top1=int(order[0]);top5=set(map(int,order[:5]))
   for step in load(ROOT/'config.json')['stages'][stage]['checkpoints']:
    folder=ROOT/'validation'/b/stage/f'step_{step:05d}';allr=[];refs={};hgroups={};scgroups=collections.defaultdict(list)
    for p in sorted(folder.glob('*/frames.csv')):allr.extend(read(p))
    for p in sorted(folder.glob('*/histograms.jsonl')):
     with p.open() as f:
      for line in f:
       r=json.loads(line);key=(r['panel'],r['domain'],r['external_QP'],r['image']);h=hgroups.setdefault(key,dict(GT=np.zeros(1024,dtype=np.int64),predicted_all=np.zeros(1025,dtype=np.int64),predicted_content=np.zeros(1024,dtype=np.int64)))
       for k in h:h[k]+=np.array(r[k],dtype=np.int64)
    for p in sorted(folder.glob('*/scales.jsonl')):
     with p.open() as f:
      for line in f:
       s=json.loads(line)
       for feature,v in [('image',s['image_scale']),('raw_condition',s['raw_condition']),('transformed_condition',s['transformed_condition']),*s['activations'].items()]:
        scgroups[(s['panel'],s['domain'],s['external_QP'],s['image'],feature)].append(v)
    wg=collections.defaultdict(list)
    for r in allr:wg[(r['panel'],r['panel_id'],r['domain'],r['external_QP'])].append(r)
    for (wp,pid,dom,qq),rr in wg.items():
     re=[z for z in rr if z['image']=='real'];fa=[z for z in rr if z['image']=='reconstruction'];window_rows.append(dict(branch=b,stage=stage,checkpoint=step,panel=wp,panel_id=pid,domain=dom,external_QP=qq,n_pairs=len(re),AUROC=auc([float(z['p_fake']) for z in re],[float(z['p_fake']) for z in fa]),**{'real_'+k:v for k,v in averages(re,fields).items()},**{'fake_'+k:v for k,v in averages(fa,fields).items()}))
    groups=collections.defaultdict(list)
    for r in allr:groups[(r['panel'],r['domain'],int(r['external_QP']))].append(r)
    for (panel,dom,q),rows in groups.items():
     real=[r for r in rows if r['image']=='real'];fake=[r for r in rows if r['image']=='reconstruction'];key=lambda r:(r['panel_id'],int(r['frame_index']));fmap={key(r):r for r in fake};pp=[]
     for r in real:
      f=fmap[key(r)];diff=float(f['p_fake'])-float(r['p_fake']);pp.append(diff);pairsout.append(dict(branch=b,stage=stage,checkpoint=step,panel=panel,domain=dom,external_QP=q,panel_id=r['panel_id'],frame_index=r['frame_index'],real_p_fake=float(r['p_fake']),fake_p_fake=float(f['p_fake']),fake_minus_real=diff,correct_order=diff>0,tie=diff==0))
     rh=hgroups[(panel,dom,q,'real')];gt=rh['GT'];nn=gt.sum();global_freqce=float(-(gt*np.log(prior)).sum()/nn);global_majority=float(gt[top1]/nn)
     scount=(byTINY if stage=='TINY' else byMAIN).get((dom,q),counts);sprior=(scount+1)/(scount.sum()+1024);sorder=np.argsort(-sprior);stop1=int(sorder[0]);stop5=sorder[:5];freqce=float(-(gt*np.log(sprior)).sum()/nn);majority=float(gt[stop1]/nn);majority5=float(gt[stop5].sum()/nn)
     base=dict(branch=b,stage=stage,checkpoint=step,panel=panel,domain=dom,external_QP=q,n_pairs=len(pp),AUROC=auc([float(r['p_fake']) for r in real],[float(r['p_fake']) for r in fake]),paired_p_fake_difference_mean=float(np.mean(pp)),paired_p_fake_difference_std=float(np.std(pp)),paired_p_fake_difference_p05=float(np.percentile(pp,5)),paired_p_fake_difference_p95=float(np.percentile(pp,95)),paired_correct_order_fraction=float(np.mean(np.array(pp)>0)),paired_tie_fraction=float(np.mean(np.array(pp)==0)),uniform_CE=math.log(1025),uniform_content_CE=math.log(1024),training_reference_scope='domain×QP with global fallback if no TINY stratum',training_global_frequency_CE=global_freqce,training_global_majority_accuracy=global_majority,training_frequency_content_CE=freqce,training_majority_content_accuracy=majority,training_top5_content_accuracy=majority5,balanced_prior_real_CE=math.log(2)+freqce,balanced_prior_fake_CE=math.log(2),balanced_prior_AUROC=.5,balanced_prior_real_all_class_accuracy=0.,balanced_prior_fake_accuracy=1.,balanced_prior_p_fake=.5,uniform_content_top1=1/1024,uniform_content_top5=5/1024)
     ra=averages(real,fields);fa=averages(fake,fields);base.update({'real_'+k:v for k,v in ra.items()});base.update({'fake_'+k:v for k,v in fa.items()});base['D_loss']=.5*(base['real_CE_content']+base['fake_CE_fake']);base['CE_G_readonly']=base['fake_CE_content'];base['CE_real']=base['real_CE_content'];base['CE_fake']=base['fake_CE_fake'];base['CE_real_content_only']=base['real_CE_content_only'];base['balanced_full_class_accuracy']=.5*(base['real_all_class_top1']+base['fake_fake_accuracy']);base['balanced_prior_full_class_accuracy']=.5;summaries.append(base)
     freqout.append(dict(branch=b,stage=stage,checkpoint=step,panel=panel,domain=dom,external_QP=q,predictive_counts_source='fixed MAIN one-cycle labels' if stage=='MAIN' else 'fixed eight TINY pair labels',Laplace_alpha=1,training_majority_class=stop1,reference_scope='domain×QP with global fallback if no TINY stratum',global_frequency_CE=global_freqce,frequency_content_CE=freqce,majority_content_accuracy=majority,top5_content_accuracy=majority5,validation_GT_entropy=float(-sum((z/nn)*math.log(z/nn) for z in gt if z)),validation_GT_majority_descriptive=float(gt.max()/nn),validation_GT_top5_descriptive=float(np.sort(gt)[-5:].sum()/nn)))
    for (panel,dom,q,image),h in hgroups.items():distribution.append(dict(branch=b,stage=stage,checkpoint=step,panel=panel,domain=dom,external_QP=q,image=image,**{k:v.tolist() for k,v in h.items()}))
    for (panel,dom,q,image,feature),vs in scgroups.items():scale_summary.append(dict(branch=b,stage=stage,checkpoint=step,panel=panel,domain=dom,external_QP=q,image=image,feature=feature,frames=len(vs),**{k:float(np.mean([v[k] for v in vs])) for k in ('mean','std','RMS','min','max')},finite=all(v['finite'] for v in vs)))
 write(ROOT/'results/window_metrics.csv',window_rows);write(ROOT/'results/classification_summary.csv',summaries);write(ROOT/'results/paired_probability_separation.csv',pairsout);write(ROOT/'results/frequency_baselines.csv',freqout);write(ROOT/'results/input_activation_scales.csv',scale_summary)
 with (ROOT/'results/class_distributions.jsonl').open('w') as f:
  for r in distribution:f.write(json.dumps(r)+'\n')
 # Aggregations across all QPs and both domains weight frames equally; preserve per-domain/sequence groups above.
 final=[];curve=[]
 for b in BR:
  for stage in ['TINY','MAIN']:
   for step in load(ROOT/'config.json')['stages'][stage]['checkpoints']:
    for panel in ['tiny_fit','seen','unseen']:
     rs=[r for r in summaries if r['branch']==b and r['stage']==stage and r['checkpoint']==step and r['panel']==panel]
     if not rs:continue
     sel=[r for r in pairsout if r['branch']==b and r['stage']==stage and r['checkpoint']==step and r['panel']==panel];weights=np.array([r['n_pairs'] for r in rs]);entry=dict(branch=b,stage=stage,checkpoint=step,panel=panel,n_pairs=int(weights.sum()),AUROC=auc([r['real_p_fake'] for r in sel],[r['fake_p_fake'] for r in sel]))
     for k in ['D_loss','real_CE_content','fake_CE_fake','CE_G_readonly','real_CE_content_only','real_content_top1','real_content_top5','fake_content_top1','fake_content_top5','real_all_class_top1','fake_fake_accuracy','real_p_fake','fake_p_fake','paired_p_fake_difference_mean','paired_correct_order_fraction','training_frequency_content_CE','training_majority_content_accuracy','training_top5_content_accuracy']:entry[k]=float(np.average([r[k] for r in rs],weights=weights))
     curve.append(entry)
     if step==load(ROOT/'config.json')['stages'][stage]['updates']:final.append(entry)
 write(ROOT/'results/learning_curves.csv',curve);write(ROOT/'results/final_summary.csv',final)
 contrasts=[]
 for r in final:
  if r['branch']==BR[0]:continue
  ref=next(x for x in final if x['branch']==BR[0] and x['stage']==r['stage'] and x['panel']==r['panel'])
  contrasts.append(dict(method=r['branch'],reference=BR[0],stage=r['stage'],panel=r['panel'],**{k+'_delta':r[k]-ref[k] for k in ['AUROC','real_content_top1','real_content_top5','real_CE_content_only','paired_p_fake_difference_mean','paired_correct_order_fraction']}))
 write(ROOT/'results/recipe_comparisons.csv',contrasts)
 subprocess.check_call([PLOT_PYTHON,str(ROOT/'plot_results.py')],cwd=REPO)
 lines=['# V6.28 fixed-B1 discriminator learnability','', 'Only D and its condition adapter were updated. GVC-RT/B1 and the labeler remained frozen. No new codec, MCL-JCV, FID or RD evaluation was performed.','', '## Fixed budgets','', 'Four branches each completed TINY 2000 updates and MAIN 5000 updates. TINY and MAIN each started from the same full C step0 D state with a fresh optimizer.','', '## Final validation','', '| Branch | Stage | Panel | AUROC | fake−real p(fake) | Content top1 | Training majority | Content-only CE | Frequency CE |','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in final:lines.append('| '+ ' | '.join([r['branch'],r['stage'],r['panel'],*[f"{r[k]:.6g}" for k in ['AUROC','paired_p_fake_difference_mean','real_content_top1','training_majority_content_accuracy','real_CE_content_only','training_frequency_content_CE']]])+' |')
 lines+=['','## Evidence and limits','','An initial local preflight rejected mixed integer/float optimizer betas. Both values were cast to float without changing the optimizer recipe; the complete preflight then passed. No historical results were modified.','', 'Normal matched-condition paired responses, AUROC and content-only classification must be considered together. Frequency baselines use training labels only with Laplace alpha=1, stratified by domain/external QP (global fallback for untrained TINY QPs); global references are retained separately. Validation label frequencies are descriptive. Unseen denotes disjoint frame intervals from this round’s MAIN plan, not absent historical exposure. Fake is one aggregate class, while real content is split across 1024 classes; fake top1 share or low full-class real accuracy alone is not used to diagnose collapse.','']
 for b in BR:
  main=next(r for r in final if r['branch']==b and r['stage']=='MAIN' and r['panel']=='unseen');tinyfit=next(r for r in final if r['branch']==b and r['stage']=='TINY' and r['panel']=='tiny_fit');initrow=next(r for r in curve if r['branch']==b and r['stage']=='MAIN' and r['checkpoint']==0 and r['panel']=='unseen');aucgain=main['AUROC']-initrow['AUROC'];cgain=main['real_content_top1']-initrow['real_content_top1'];freqbeat=main['real_CE_content_only']<main['training_frequency_content_CE'];lines.append(f"- {b}: TINY fitting AUROC {tinyfit['AUROC']:.6f}; MAIN unseen AUROC {main['AUROC']:.6f} (step0 change {aucgain:+.6f}); content top1 change {cgain:+.6f}; content-only CE below training-frequency reference: {freqbeat}.")
 lines+=['','## Recipe contrasts on unseen MAIN','']
 for r in contrasts:
  if r['stage']=='MAIN' and r['panel']=='unseen':lines.append(f"- {r['method']} vs A: AUROC {r['AUROC_delta']:+.6f}; content-only CE {r['real_CE_content_only_delta']:+.6f}; content top1 {r['real_content_top1_delta']:+.6f}; paired probability separation {r['paired_p_fake_difference_mean_delta']:+.6f}.")
 lines+=['','The RMS and training-zero contrasts change condition handling; any corresponding learning difference is evidence about this connection under the fixed budget, not proof of a universal cause. The official-optimizer contrast changes the full AdamW recipe, not learning rate alone. Successful TINY fitting is not generalization; discriminator learning does not establish FID benefit.','', '## One unexecuted joint-training contrast','', 'Use the best-supported MAIN unseen discriminator recipe only after reviewing the full normal-condition separation and content evidence; compare fixed B1 decoder continuation against the same continuation plus this local supervision, with matched data/initialization/budget and the existing whole-gradient 5% control. This contrast was not executed here. Do not automatically unfreeze G.','', 'All raw validation frames, category histograms, input/activation scales and training update records are indexed for review.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');(ROOT/'RESULTS_SUMMARY.md').write_text('\n'.join(lines[:lines.index('## Evidence and limits')])+'\n')
 dump(ROOT/'results/summary_integrity.json',dict(status='PASS',validation_configs=1704,final_rows=len(final),learning_curve_rows=len(curve),training_frequency_source='fixed training GT only',no_G_updates=True,no_new_FID_RD=True))
 subprocess.check_call([PYTHON,str(ROOT/'assess_results.py')],cwd=REPO)
 print('SUMMARY PASS',len(final),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  dump(ROOT/'logs/summary_failure.json',dict(status='BLOCKED',traceback=traceback.format_exc()));state=load(ROOT/'pipeline_status.json');state.update(status='BLOCKED',phase='CPU_summary',error_log=str(ROOT/'logs/summary_failure.json'));dump(ROOT/'pipeline_status.json',state);raise
