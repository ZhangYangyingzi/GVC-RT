"""Correct offline native-image validity only; preserve training, outputs and other metrics."""
from util import *
import argparse,csv,statistics,math,torch,gc,collections
from evaluate import tag,proofpath,blur,dominates
AUDIT=ROOT/'audits/native_validity_correction'
TARGET_IDS=[r['id'] for r in devrows() if (ROOT/'cache/evaluation_supervision'/r['id']).exists()]

def prepare():
 AUDIT.mkdir(exist_ok=True,parents=True)
 dest=AUDIT/'before';assert not dest.exists(),'Do not overwrite initial metric evidence';dest.mkdir()
 jobs=load(ROOT/'receipts/evaluation_queue.complete.json')['jobs'];affected=[j for j in jobs if j['id'] in TARGET_IDS]
 for j in affected:
  for p in [proofpath(j),ROOT/'metrics/frames'/f'{tag(j)}.csv']:
   q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 for rel in ['evaluation/milestones','metrics/per_video_metrics.csv']:
  p=ROOT/rel;q=dest/rel
  if p.is_dir():shutil.copytree(p,q)
  else:q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 # If four unchanged metrics already reject domination, changing only MC-Sobel cannot trigger the rule.
 checks=[]
 for group in ['A_zero_base','B_flow_guided']:
  vals=[load(ROOT/'evaluation/milestones'/group/f'{s:04d}.json') for s in [0,50,100,200,300]]
  for k in range(2,len(vals)):
   for early in vals[:k-1]:
    ok=[]
    for late in vals[k-1:k+1]:
     flags=[]
     for key in ['PSNR','LPIPS','DISTS','FloLPIPS']:
      a,b=early['metrics'][key],late['metrics'][key];t=1e-6*max(abs(a),abs(b),1e-12);flags.append(a>=b-t if key=='PSNR' else a<=b+t)
     ok.append(all(flags))
    checks.append(dict(group=group,early=early['step'],previous=vals[k-1]['step'],current=vals[k]['step'],could_dominate_both_in_four_unchanged_metrics=all(ok)))
 assert not any(r['could_dominate_both_in_four_unchanged_metrics'] for r in checks)
 dump(AUDIT/'plan.json',dict(reason='Padded GT flow validity was cropped without intersecting native sampling domain; only MC-RGB/MC-Sobel offline metrics corrected',targets=TARGET_IDS,jobs=affected,points=len(affected),training_unchanged=True,inference_unchanged=True,stop_rule_invariant_from_four_unchanged_metrics=checks,stream_implementation_sha256=sha(ROOT/'stream.py'),time=stamp()))

def run(shard):
 setup();from restore import restore_b300
 from reference import coexist_original
 from model import DeformableHistory,bridge,decode
 from stream import raw_frames,stream_file,decode_frames,gt_supervision
 from temporal import temporal_errors
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);rt,im,pm=coexist_original();st=stats()
 jobs=load(AUDIT/'plan.json')['jobs']
 for j in jobs[shard::4]:
  done=AUDIT/'points'/f'{tag(j)}.json'
  if done.exists():continue
  proof=load(AUDIT/'before/evaluation/points'/f'{tag(j)}.json');oldrows=list(csv.DictReader((AUDIT/'before/metrics/frames'/f'{tag(j)}.csv').open()));rows=[dict(r) for r in oldrows];row=next(r for r in devrows() if r['id']==j['id']);arrays=raw_frames(row);stream,meta=stream_file(rt,im,pm,row,j['qp'],arrays);assert sha(stream)==proof['stream_sha256'];g=c['models']['generator'];g.reset_temporal_buffer();m=None
  if j['label'] in ['A_zero_base','B_flow_guided']:
   cp=ROOT/'checkpoints'/j['label']/f"step_{j['step']:04d}.pt";ck=readtensor(cp);assert sha(cp)==proof['checkpoint_sha256'];m=DeformableHistory(st,j['label']).cuda();m.load_state_dict(ck['model']);m.eval();m.reset()
  h,w=row['height'],row['width'];prev_y=prev_gt=prev_ref=None;changes=[];mse_proofs=[];new_rgb=[];new_sobel=[];removed=[]
  with torch.no_grad():
   for i,q,f,ref,sps,_ in decode_frames(rt,im,pm,stream,meta):
    heartbeat('fix_metrics'+str(shard),'CORRECTING_NATIVE_VALIDITY',job=j,frame=i)
    if i==0 or j['label']=='Original':y=ref[...,:h,:w]
    else:
     l=bridge(c,f.float());lp=l
     if m is not None:
      side=npzload(ROOT/'cache/eval_decoded_refs'/f"{j['id']}_q{j['qp']}"/f'{i:03d}.npz');back=torch.from_numpy(side['decoded_ref_backward']).cuda();e=torch.from_numpy(side['decoded_ref_evidence']).cuda();lp,_=m(l,back,e,q,j['id'],i,j['beta'])
     y=decode(c,lp,q)[...,:h,:w]
    if j.get('sigma'):y=blur(y,j['sigma'])
    gt=torch.from_numpy(arrays[i]).permute(2,0,1)[None].cuda().float()/255
    if i in [1,32,63]:
     mse=float((y-gt).square().mean());error=abs(mse-float(oldrows[i]['RGB_MSE']));assert error<=1e-12,('OUTPUT_CHANGED',j,i,error);mse_proofs.append(dict(frame=i,MSE_difference=error))
    if i>0:
     flow,valid=gt_supervision(row,i,arrays,None,None);raw=npzload(ROOT/'cache/evaluation_supervision'/row['id']/f'{i:03d}.npz');removed.append(int(raw['GT_valid'].sum())-int(valid.sum()));a,b,_=temporal_errors(y,prev_y,gt,prev_gt,flow,valid);assert a is not None and b is not None;a,b=float(a),float(b);assert math.isfinite(a+b);rows[i].update(MC_RGB=a,MC_Sobel=b);new_rgb.append(a);new_sobel.append(b);changes.append(dict(frame=i,removed_pixels=removed[-1],old_MC_RGB=float(oldrows[i]['MC_RGB']),new_MC_RGB=a,old_MC_Sobel=float(oldrows[i]['MC_Sobel']),new_MC_Sobel=b))
    prev_y=y.detach();prev_gt=gt.detach();prev_ref=ref.detach()
  summary=dict(proof['summary']);summary['MC_RGB']=statistics.mean(new_rgb);summary['MC_Sobel']=statistics.mean(new_sobel)
  correction=dict(reason='native RGB sampling-domain intersection of fixed GT-valid mask',unchanged_fixed_flow=True,unchanged_model_outputs=True,unchanged_other_metrics=True,MSE_proofs=mse_proofs,removed_valid_pixels_across_63_pairs=sum(removed),implementation_sha256=sha(ROOT/'stream.py'),corrector_sha256=sha(ROOT/'fix_native_metrics.py'))
  proof['summary']=summary;proof['native_validity_correction']=correction
  # Other metric fields retain original strings; no precision conversion.
  csvwrite(ROOT/'metrics/frames'/f'{tag(j)}.csv',rows);dump(proofpath(j),proof);dump(done,dict(status='PASS',job=j,correction=correction,per_frame_changes=changes));del m,arrays;gc.collect();torch.cuda.empty_cache()
 heartbeat('fix_metrics'+str(shard),'COMPLETE')

def finish():
 jobs=load(AUDIT/'plan.json')['jobs'];assert all((AUDIT/'points'/f'{tag(j)}.json').exists() for j in jobs)
 for j in jobs:
  name=tag(j);before=list(csv.DictReader((AUDIT/'before/metrics/frames'/f'{name}.csv').open()));after=list(csv.DictReader((ROOT/'metrics/frames'/f'{name}.csv').open()));assert len(before)==len(after)==64
  for b,a in zip(before,after):assert {k:v for k,v in b.items() if k not in ['MC_RGB','MC_Sobel']}=={k:v for k,v in a.items() if k not in ['MC_RGB','MC_Sobel']},('OTHER_FRAME_METRIC_CHANGED',name)
  b=load(AUDIT/'before/evaluation/points'/f'{name}.json');a=load(proofpath(j));assert {k:v for k,v in b['summary'].items() if k not in ['MC_RGB','MC_Sobel']}=={k:v for k,v in a['summary'].items() if k not in ['MC_RGB','MC_Sobel']};assert b['stream_sha256']==a['stream_sha256'] and b['codec_feature_hashes']==a['codec_feature_hashes'] and b['checkpoint_sha256']==a['checkpoint_sha256']
 original=load(ROOT/'receipts/evaluation_queue.complete.json')['jobs'];csvwrite(ROOT/'metrics/per_video_metrics.csv',[load(proofpath(j))['summary'] for j in original]);decisions=[]
 for group in ['A_zero_base','B_flow_guided']:
  values=[]
  for step in [0,50,100,200,300]:
   js=[dict(label=group,step=step,id=r['id'],qp=5,beta=1.) for r in devrows()[:4]];ss=[load(proofpath(j))['summary'] for j in js];metrics={k:statistics.mean(s[k] for s in ss) for k in load(ROOT/'PROTOCOL.json')['stop_rule']['metrics']};values.append((step,metrics));stop=any(dominates(a,values[-2][1]) and dominates(a,metrics) for _,a in values[:-2]) if len(values)>=3 else False;p=ROOT/'evaluation/milestones'/group/f'{step:04d}.json';d=load(p);assert stop==d['stop']==False;d.update(metrics=metrics,native_validity_corrected=True,original_decision_preserved=True);dump(p,d);decisions.append(dict(group=group,step=step,stop=stop))
 dump(AUDIT/'complete.json',dict(status='PASS',points=len(jobs),original_metrics_preserved_under=str(AUDIT/'before'),stop_decisions=decisions,training_unchanged=True,other_metrics_unchanged=True,all_sparse_output_MSE_match_exact=True,datetime=stamp()))
 event('NATIVE_VALIDITY_METRICS_CORRECTED',points=len(jobs))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--shard',type=int);a.add_argument('--finish',action='store_true');p=a.parse_args()
 if p.prepare:prepare()
 elif p.finish:finish()
 else:
  with lock('fix_metrics'+str(p.shard)+'.lock'):run(p.shard)
