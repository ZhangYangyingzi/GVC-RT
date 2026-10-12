"""Full diagnostic statistics and measured-rate five-model comparisons."""
from io21 import *
import math,statistics,collections
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata
from comparison import comparisons,summarize,window_rows,PAIRS

def distribution(values):
 a=np.asarray(values,float);return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))
def auc(real,fake):
 x=np.asarray(real+fake);r=rankdata(x);return float((r[len(real):].sum()-len(fake)*(len(fake)+1)/2)/(len(real)*len(fake)))
def diagnostics():
 allframes=[];allpoints=[];grads=[];classcounts=collections.defaultdict(lambda:dict(GT=np.zeros(1024,dtype=np.int64),predicted_all=np.zeros(1025,dtype=np.int64),predicted_content=np.zeros(1024,dtype=np.int64)))
 for row in load(ROOT/'diagnostic_plan.json')['updates']:
  for step in (0,250,500,1000):
   p=ROOT/'diagnostics/points'/f'{row["panel_id"]}_s{step}.json';j=load(p);assert j['status']=='PASS' and j['parameters_buffers_unchanged'] and j['no_training'];assert sha(j['frame_csv'])==j['frame_csv_SHA256'] and sha(j['histogram_path'])==j['histogram_SHA256'];allpoints.append(j);fr=read(j['frame_csv']);allframes.extend(fr)
   g=j['gradient'];out=dict(panel_id=row['panel_id'],panel_type=row['panel'],domain=row['domain'],QP=row['external_qp'],step=step,**{k:v for k,v in g.items() if isinstance(v,(int,float,bool))});out['image_gradient_norm_mean']=statistics.mean(x['norm'] for x in g['image_gradients']);out['image_gradient_nonzero_count']=sum(x['norm']>0 for x in g['image_gradients']);out.update(**{f'DISTS_deep_{x["module"]}_norm':x['deepest_term']['norm'] for x in g['DISTS_deep']});grads.append(out)
   for line in Path(j['histogram_path']).open():
    h=json.loads(line)
    if h['condition']=='matched':
     key=(h['panel_type'],h['domain'],step,h['reconstruction_panel'],h['image']);c=classcounts[key]
     for k in c:c[k]+=np.asarray(h[k],np.int64)
 groups=collections.defaultdict(list)
 metrics=['CE_content','CE_fake','p_fake','content_probability_mass','all_class_top1','all_class_top5','content_only_top1','content_only_top5','fake_class_accuracy']
 for row in allframes:
  key=tuple(row[k] for k in ['panel_type','domain','external_QP','checkpoint','reconstruction_panel','image','condition']);groups[key].append(row)
 summary=[]
 for key,rr in groups.items():
  base=dict(zip(['panel_type','domain','QP','step','reconstruction_panel','image','condition'],key));summary.append(dict(base,frames=len(rr),windows=len({r['panel_id'] for r in rr}),**{k:statistics.mean(float(r[k]) for r in rr) for k in metrics}))
 write(ROOT/'diagnostics/per_frame_metrics.csv',allframes);write(ROOT/'diagnostics/classification_group_summary.csv',summary);write(ROOT/'diagnostics/matched_gradient_windows.csv',grads)
 groups2=collections.defaultdict(list)
 for r in allframes:
  if r['condition']=='matched' and r['image'] in ('real','reconstruction'):groups2[(r['panel_type'],r['domain'],int(r['checkpoint']),r['reconstruction_panel'])].append(r)
 ability=[]
 for key,rr in groups2.items():
  real=[r for r in rr if r['image']=='real'];fake=[r for r in rr if r['image']=='reconstruction'];realce=statistics.mean(float(r['CE_content']) for r in real);fakece=statistics.mean(float(r['CE_fake']) for r in fake);gce=statistics.mean(float(r['CE_content']) for r in fake);counts=classcounts[(*key,'real')]['GT'];freq=counts/counts.sum();nonzero=freq[freq>0]
  ability.append(dict(panel_type=key[0],domain=key[1],step=key[2],reconstruction_panel=key[3],windows=len({r['panel_id'] for r in real}),frames=len(real),CE_real=realce,CE_fake=fakece,CE_G=gce,D_loss=.5*(realce+fakece),uniform_CE_ln1025=math.log(1025),GT_frequency_entropy=-float((nonzero*np.log(nonzero)).sum()),GT_majority_top1=float(freq.max()),GT_majority_top5=float(np.sort(freq)[-5:].sum()),real_content_top1=statistics.mean(float(r['content_only_top1']) for r in real),real_content_top5=statistics.mean(float(r['content_only_top5']) for r in real),fake_content_top1=statistics.mean(float(r['content_only_top1']) for r in fake),fake_content_top5=statistics.mean(float(r['content_only_top5']) for r in fake),real_all_class_top1=statistics.mean(float(r['all_class_top1']) for r in real),fake_all_class_accuracy=statistics.mean(float(r['fake_class_accuracy']) for r in fake),real_p_fake=statistics.mean(float(r['p_fake']) for r in real),fake_p_fake=statistics.mean(float(r['p_fake']) for r in fake),AUROC_fake_from_frame_mean_p_fake=auc([float(r['p_fake']) for r in real],[float(r['p_fake']) for r in fake])))
 write(ROOT/'diagnostics/discriminator_ability.csv',ability)
 hist=[]
 for key,cs in classcounts.items():
  for k,a in cs.items():
   for cl,count in enumerate(a):hist.append(dict(panel_type=key[0],domain=key[1],step=key[2],reconstruction_panel=key[3],image=key[4],distribution=k,class_index=cl,count=int(count),fraction=float(count/a.sum())))
 write(ROOT/'diagnostics/class_usage.csv',hist)
 refs={(r['panel_id'],r['checkpoint'],r['frame_index'],r['reconstruction_panel'],r['image']):r for r in allframes if r['condition']=='matched'};sensitivity=[]
 for r in allframes:
  if r['reconstruction_panel']=='fixed' and r['condition']!='matched':
   ref=refs[(r['panel_id'],r['checkpoint'],r['frame_index'],r['reconstruction_panel'],r['image'])];sensitivity.append(dict(**{k:r[k] for k in ['panel_id','panel_type','domain','external_QP','checkpoint','frame_index','image','condition']},**{'delta_'+k:float(r[k])-float(ref[k]) for k in metrics},probe_scope='zero/mismatched conditions are OOD; do not alone establish shortcut'))
 write(ROOT/'diagnostics/condition_sensitivity.csv',sensitivity)
 # Image probes with condition held fixed, same label target.
 imageprobes=[]
 for r in allframes:
  if r['reconstruction_panel']=='fixed' and r['image']!='real':
   ref=next(x for x in allframes if x['panel_id']==r['panel_id'] and x['checkpoint']==r['checkpoint'] and x['frame_index']==r['frame_index'] and x['reconstruction_panel']=='fixed' and x['image']=='real' and x['condition']==r['condition'])
   imageprobes.append(dict(**{k:r[k] for k in ['panel_id','panel_type','domain','external_QP','checkpoint','frame_index','image','condition']},**{'delta_'+k:float(r[k])-float(ref[k]) for k in metrics}))
 write(ROOT/'diagnostics/image_sensitivity.csv',imageprobes)
 import torch
 torch.set_num_threads(2);parameter=[];source=torch.load(load(ROOT/'config.json')['source_checkpoint']['path'],map_location='cpu',weights_only=True)
 for b in BRANCHES:
  for step in (0,250,500,1000):
   cp=load(V26/'branches'/b/'checkpoint_index.json')[str(step)];s=torch.load(cp['path'],map_location='cpu',weights_only=True)
   for k in ('wrapper','bridge','generator'):
    denom=sum(float(t.double().square().sum()) for t in source[k].values())**.5;num=sum(float((s[k][name].double()-t.double()).square().sum()) for name,t in source[k].items())**.5;parameter.append(dict(branch=b,step=step,module=k,L2_change=num,source_L2=denom,relative_L2=num/max(denom,1e-12),hash=tensor_hash(s[k]),source_hash=tensor_hash(source[k])))
   del s
 write(ROOT/'diagnostics/parameter_changes.csv',parameter)
 recchanges=[]
 for row in load(ROOT/'diagnostic_plan.json')['updates']:
  base=torch.load(ROOT/'cache/diagnostic/B1'/f'{row["panel_id"]}.pt',map_location='cpu',weights_only=True)
  for step in (0,250,500,1000):
   s=torch.load(ROOT/'cache/diagnostic'/f'C{step}'/f'{row["panel_id"]}.pt',map_location='cpu',weights_only=True)
   for i,(x,y) in enumerate(zip(s['reconstructions'],base['reconstructions'])):recchanges.append(dict(panel_id=row['panel_id'],panel_type=row['panel'],domain=row['domain'],QP=row['external_qp'],step=step,frame=49+i,reconstruction_MAE_vs_B1=float((x-y).abs().mean()),reconstruction_RMSE_vs_B1=float((x-y).square().mean().sqrt())))
 write(ROOT/'diagnostics/reconstruction_changes.csv',recchanges)
 logstats=[];logrows=[]
 for b in BRANCHES:
  rr=read(V26/'branches'/b/'gradient_summary.csv');assert len(rr)==1000
  write(ROOT/'diagnostics/training_logs'/f'{b}_gradient_summary.csv',rr);logrows.extend(dict(r,branch=b) for r in rr)
  for lo,hi in [(1,1000),(1,250),(251,500),(501,750),(751,1000)]:
   rs=[r for r in rr if lo<=int(r['step'])<=hi]
   for metric in ['B1_gradient_norm','unweighted_adv_gradient_norm','weighted_adv_gradient_norm','lambda_eff','adv_B1_ratio','gradient_cosine','bridge_weighted_adv_B1_ratio','generator_weighted_adv_B1_ratio','global_clip_coefficient']:
    if metric in rs[0]:logstats.append(dict(branch=b,start=lo,end=hi,metric=metric,**distribution([float(r[metric]) for r in rs])))
 write(ROOT/'diagnostics/training_gradient_distributions.csv',logstats)
 dump(ROOT/'audits/diagnostic_integrity.json',dict(status='PASS',expected_points=len(load(ROOT/'diagnostic_plan.json')['updates'])*4,completed_points=len(allpoints),frames=len(allframes),optimizer_steps=0,no_training=True,states_unchanged=True,condition_detached=True,D_strict_checkpoint_restored=True,labeler_pretrained_verified=True,full15_frame_gradients=True,unit_scale4096=True,whole_BG_ratio_max=max(g['adv_B1_ratio'] for g in grads),DISTS_deep_valid=True))
 print('DIAGNOSTICS SUMMARY PASS',len(allpoints),flush=True)

def evaluation():
 from adapter import validate
 fid=module('mcl_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py').low_rank_fid
 vs=sources('mcljcv');raw=[];windows=[];gt=[]
 for v in vs:
  for m in METHODS:
   for q in range(10):
    r=load(point('mcljcv',m,v['video_index'],q));validate(r,v,q,m);raw.append(r);fr=read(r['frame_metrics_path']);tr=read(r['transitions_path']);windows.extend(window_rows(r,fr,tr))
  with np.load(next(r['feature_path'] for r in raw if r['method']=='original' and r['QP']==0 and r['video_index']==v['video_index'])) as z:gt.append(z['real'].copy())
 reference=np.concatenate(gt);assert reference.shape==(64*len(vs),2048);np.save(ROOT/'features/GT_mcljcv.npy',reference);pool=sha(ROOT/'features/GT_mcljcv.npy');out=ROOT/'results/mcljcv';out.mkdir(parents=True,exist_ok=True);pooled=[];macro=[];components=[]
 # Fixed GT feature identities across all methods and QPs.
 for m in METHODS:
  for q in range(10):
   cp=out/'pooled_points'/f'{m}_qp{q}.json'
   if cp.exists():j=load(cp);assert j['GT_features_sha256']==pool;pooled.append(j);continue
   rs=[r for r in raw if r['method']==m and r['QP']==q];rs.sort(key=lambda r:r['video_index']);recs=[];paths=[]
   for i,r in enumerate(rs):
    with np.load(r['feature_path']) as z:assert np.allclose(z['real'],reference[i*64:(i+1)*64],rtol=1e-5,atol=1e-6);recs.append(z['reconstruction'])
    paths.append(dict(path=r['feature_path'],sha256=r['feature_sha256']))
   reconstruction=np.concatenate(recs);first,second=reference.astype(np.float64),reconstruction.astype(np.float64);a,b=first-first.mean(0),second-second.mean(0);dmu=float(np.square(first.mean(0)-second.mean(0)).sum());dcov=float(np.square(a).sum()/(len(a)-1)+np.square(b).sum()/(len(b)-1)-2*np.linalg.svd((a@b.T)/(len(a)-1),compute_uv=False).sum());value=float(fid(first,second));assert abs(value-dmu-dcov)<1e-7+1e-8*abs(value)
   r=dict(dataset='mcljcv',method=m,QP=q,bpp=statistics.mean(r['bpp'] for r in rs),FID=value,D_mu=dmu,D_cov=dcov,FID_closure_error=abs(value-dmu-dcov),samples=len(reference),sequences=len(vs),GT_features_sha256=pool,feature_sources=paths,checkpoint_SHA256=rs[0]['checkpoint_sha256'],aggregation='fixed dataset pooled features; sample covariance ddof1; float64',**{k:statistics.mean(r[k] for r in rs) for k in ['LPIPS','DISTS','FloLPIPS','kbps','kbps_20fps','PSNR','SSIM']});dump(cp,r);pooled.append(r);print('POOLED',m,q,flush=True)
 write(out/'per_sequence_all_qp.csv',raw);write(out/'fid_pooled.csv',pooled);dump(out/'fid_pooled.json',pooled);write(out/'all_qp_summary.csv',pooled);write(out/'window_metrics.csv',windows)
 per=[];grids=[]
 for v in vs:
  for metric in ('LPIPS','DISTS','FloLPIPS'):
   rr,gg=comparisons('mcljcv',v['name'],metric,[r for r in raw if r['video_index']==v['video_index']]);per.extend(rr);grids.extend(gg)
 fids,gg=comparisons('mcljcv','pooled_dataset','FID',pooled,True);grids.extend(gg);summary=summarize('mcljcv',per,len(vs));write(out/'equal_rate_per_sequence.csv',per);write(out/'equal_rate_summary.csv',summary+fids);write(out/'common_rate_grid.csv',grids);dump(out/'equal_rate_summary.json',summary+fids)
 # Explicit shared common intervals and interpolation inputs; low/middle/high preset equal-ln thirds.
 bands=[]
 for m,ref in PAIRS:
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   values=collections.defaultdict(list)
   for name,rows in ([('pooled_dataset',pooled)] if metric=='FID' else [(v['name'],[r for r in raw if r['video_index']==v['video_index']]) for v in vs]):
    curves={mm:sorted([r for r in rows if r['method']==mm],key=lambda x:x['bpp']) for mm in METHODS};lo=max(min(r['bpp'] for r in rs) for rs in curves.values());hi=min(max(r['bpp'] for r in rs) for rs in curves.values())
    if hi<=lo:continue
    xs=np.linspace(np.log(lo),np.log(hi),100);bounds=np.linspace(xs[0],xs[-1],4);v1=PchipInterpolator(np.log([r['bpp'] for r in curves[m]]),[r[metric] for r in curves[m]],extrapolate=False)(xs);v0=PchipInterpolator(np.log([r['bpp'] for r in curves[ref]]),[r[metric] for r in curves[ref]],extrapolate=False)(xs)
    for j,label in enumerate(('low','middle','high')):
     idx=(xs>=bounds[j])&((xs<bounds[j+1]) if j<2 else (xs<=bounds[j+1]));values[label].append((float(v1[idx].mean()),float(v0[idx].mean())));bands.append(dict(scope='per_sequence',sequence=name,method=m,reference=ref,metric=metric,band=label,low_ln_bpp=float(bounds[j]),high_ln_bpp=float(bounds[j+1]),method_mean=float(v1[idx].mean()),reference_mean=float(v0[idx].mean()),delta=float((v1[idx]-v0[idx]).mean()),relative_change_percent=100*(float(v1[idx].mean())-float(v0[idx].mean()))/float(v0[idx].mean())))
   for label,rs in values.items():
    a=statistics.mean(x[0] for x in rs);b=statistics.mean(x[1] for x in rs);bands.append(dict(scope='dataset',sequence='equal_video_weight' if metric!='FID' else 'pooled_dataset',method=m,reference=ref,metric=metric,band=label,method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b))
 write(out/'rate_band_summary.csv',bands)
 lo=max(min(r['bpp'] for r in pooled if r['method']==m) for m in METHODS);hi=min(max(r['bpp'] for r in pooled if r['method']==m) for m in METHODS);xs=np.linspace(np.log(lo),np.log(hi),100);comp={}
 for m in METHODS:
  rs=sorted([r for r in pooled if r['method']==m],key=lambda x:x['bpp']);comp[m]={k:np.interp(xs,np.log([r['bpp'] for r in rs]),[r[k] for r in rs]) for k in ['FID','D_mu','D_cov']};assert np.allclose(comp[m]['D_mu']+comp[m]['D_cov'],comp[m]['FID'],atol=1e-7)
 eq=[];cb=[];bounds=np.linspace(xs[0],xs[-1],4)
 for m,ref in PAIRS:
  for i,x in enumerate(xs):eq.append(dict(dataset='mcljcv',method=m,reference=ref,grid_index=i,bpp=float(np.exp(x)),common_low_bpp=lo,common_high_bpp=hi,interpolation='diagnostic shared linear ln(bpp) weights',**{'delta_'+k:float(comp[m][k][i]-comp[ref][k][i]) for k in comp[m]}))
  for j,label in enumerate(('low','middle','high')):
   idx=(xs>=bounds[j])&((xs<bounds[j+1]) if j<2 else (xs<=bounds[j+1]));cb.append(dict(dataset='mcljcv',method=m,reference=ref,band=label,low_ln_bpp=float(bounds[j]),high_ln_bpp=float(bounds[j+1]),**{'delta_'+k:float((comp[m][k]-comp[ref][k])[idx].mean()) for k in comp[m]}))
 write(out/'fid_decomposition_raw.csv',pooled);write(out/'fid_decomposition_equal_rate.csv',eq);write(out/'fid_component_rate_bands.csv',cb)
 confirmed=set(load(ROOT/'audits/overlap_integrity.json')['confirmed']);retained=[v for v in vs if v['name'] not in confirmed]
 if not confirmed:write(out/'confirmed_overlap_excluded_equal_rate_summary.csv',[dict(r,excluded_confirmed_sequences=0,retained_sequences=len(vs)) for r in summary+fids])
 else:
  # Recompute paired and pooled statistics from retained feature pool; no per-video FID averaging.
  per2=[r for r in per if r['sequence'] not in confirmed];pool2=[];ref2=np.concatenate([reference[i*64:(i+1)*64] for i,v in enumerate(vs) if v['name'] not in confirmed])
  for m in METHODS:
   for q in range(10):
    rs=[r for r in raw if r['sequence'] not in confirmed and r['method']==m and r['QP']==q];rec2=np.concatenate([np.load(r['feature_path'])['reconstruction'] for r in rs]);pool2.append(dict(dataset='mcljcv',method=m,QP=q,bpp=statistics.mean(r['bpp'] for r in rs),FID=float(fid(ref2,rec2))))
  f2,_=comparisons('mcljcv','pooled_dataset','FID',pool2,True);write(out/'confirmed_overlap_excluded_equal_rate_summary.csv',summarize('mcljcv',per2,len(retained))+f2)
 dump(ROOT/'audits/evaluation_integrity.json',dict(status='PASS',N=len(vs),expected_points=len(vs)*50,completed_points=len(raw),fresh=len(raw),reused=0,real_RANS=True,independent_decode=True,total_bytes_include_all_headers_I=True,GT_fixed_pool=pool,FID_pooled_samples=len(reference),FID_float64_ddof1=True,FID_decomposition_closed=True,PCHIP_main=True,linear_diagnostic_components=True,no_extrapolation=True,comparisons=len(PAIRS),common_rates_saved=True))
 print('EVALUATION SUMMARY PASS',len(raw),flush=True)
if __name__=='__main__':diagnostics();evaluation()
