"""Quality-blind actual-rate matching and video-cluster statistical analysis."""
import itertools
import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from audit_io import *

LATENTS=('cosine','L1','normalized_L2')
QUALITY=('LPIPS','DISTS','FloLPIPS')


def match_rates(points):
    """Only rates and QPs decide pair inclusion and primary selection."""
    rows=[];primary=[]
    for (dataset,video),group in points.groupby(['dataset','video'],sort=True):
        original=group[group.method=='Original']
        for method in ('B1','B_covstats'):
            eligible=[]
            for (_,a),(_,b) in itertools.product(original.iterrows(),group[group.method==method].iterrows()):
                avg=(a.actual_bpp+b.actual_bpp)/2
                gap=abs(a.actual_bpp-b.actual_bpp)/avg
                if gap<=.02:
                    eligible.append(dict(dataset=dataset,video=video,method=method,
                         original_QP=int(a.QP),method_QP=int(b.QP),original_bpp=float(a.actual_bpp),
                         method_bpp=float(b.actual_bpp),mean_bpp=float(avg),rate_gap=float(gap),
                         rate_match_level='1pct' if gap<=.01 else '2pct'))
            rows.extend(eligible)
            exact=[r for r in eligible if r['rate_match_level']=='1pct']
            if exact:primary.append(min(exact,key=lambda r:(r['rate_gap'],r['mean_bpp'],r['original_QP'],r['method_QP'])))
    fields=['dataset','video','method','original_QP','method_QP','original_bpp','method_bpp','mean_bpp','rate_gap','rate_match_level']
    return pd.DataFrame(rows,columns=fields),pd.DataFrame(primary,columns=fields)


def correlate(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);mask=np.isfinite(x)&np.isfinite(y);x=x[mask];y=y[mask]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return dict(Pearson=None,Spearman=None,n=len(x))
    return dict(Pearson=float(pearsonr(x,y).statistic),Spearman=float(spearmanr(x,y).statistic),n=len(x))


def cluster_ci(frame,x,y,reps=2000):
    ids=frame.video.unique()
    if len(ids)<2:return dict(Pearson_low=None,Pearson_high=None,Spearman_low=None,Spearman_high=None,video_clusters=len(ids))
    # Rank once for pooled Spearman. Re-rank inside each resample for exact bootstrap statistic.
    rng=np.random.default_rng(20261014);clusters={v:g for v,g in frame.groupby('video')};vals=[]
    for _ in range(reps):
        sample=pd.concat([clusters[v] for v in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
        c=correlate(sample[x],sample[y]);vals.append([c['Pearson'],c['Spearman']])
    result={'video_clusters':len(ids)}
    for i,name in enumerate(('Pearson','Spearman')):
        a=[v[i] for v in vals if v[i] is not None]
        result.update({name+'_low':float(np.quantile(a,.025)) if a else None,name+'_high':float(np.quantile(a,.975)) if a else None})
    return result


def residuals(group,x,y):
    # Rate plus video/method fixed effects; refitted for bootstrap below.
    rate=np.log(group.actual_bpp.to_numpy(float))
    design=np.column_stack([np.ones(len(group)),rate,
          pd.get_dummies(group.video,drop_first=True,dtype=float).to_numpy(),
          pd.get_dummies(group.method,drop_first=True,dtype=float).to_numpy()])
    a=group[x].to_numpy(float);b=group[y].to_numpy(float)
    return a-design@np.linalg.lstsq(design,a,rcond=None)[0],b-design@np.linalg.lstsq(design,b,rcond=None)[0]


def regression_ci(group,x,y):
    ids=group.video.unique();rng=np.random.default_rng(20261014);values=[]
    if len(ids)<2:return dict(Pearson_low=None,Pearson_high=None,Spearman_low=None,Spearman_high=None,video_clusters=len(ids))
    for _ in range(2000):
        pieces=[]
        for j,v in enumerate(rng.choice(ids,len(ids),replace=True)):
            piece=group[group.video==v].copy();piece['video']=f'cluster_{j}';pieces.append(piece)
        g=pd.concat(pieces,ignore_index=True);a,b=residuals(g,x,y);c=correlate(a,b)
        values.append([c['Pearson'],c['Spearman']])
    out={'video_clusters':len(ids)}
    for i,name in enumerate(('Pearson','Spearman')):
        vs=[v[i] for v in values if v[i] is not None]
        out.update({name+'_low':float(np.quantile(vs,.025)) if vs else None,name+'_high':float(np.quantile(vs,.975)) if vs else None})
    return out


def enrich(pairs,points):
    rows=[]
    index=points.set_index(['dataset','video','method','QP'])
    for r in pairs.to_dict('records'):
        a=index.loc[(r['dataset'],r['video'],'Original',r['original_QP'])]
        b=index.loc[(r['dataset'],r['video'],r['method'],r['method_QP'])]
        row=dict(r)
        for scope in ('P','all','I'):
            for metric in (*LATENTS,*QUALITY):
                k=scope+'_'+metric
                row['original_'+k]=a[k];row['method_'+k]=b[k];row['delta_'+k]=b[k]-a[k]
        rows.append(row)
    fields=list(pairs.columns)+[prefix+scope+'_'+metric for scope in ('P','all','I') for metric in (*LATENTS,*QUALITY) for prefix in ('original_','method_','delta_')]
    return pd.DataFrame(rows,columns=fields)


def main():
    from worker import key,valid
    tasks=load(ROOT/'task_manifest.json')['tasks'];records=[];latent_parts=[];summaries=[];wrapper_parts=[]
    noise=[]
    for task in tasks:
        assert valid(task),('invalid or missing point',task)
        folder=ROOT/'parts'/key(task);r=load(folder/'point.json');records.append(r)
        latent=pd.read_csv(folder/'latent.csv');trans=pd.read_csv(folder/'features.transitions.csv')
        latent['FloLPIPS']=latent.frame.map(dict(zip(trans.to_frame,trans.FloLPIPS)))
        latent_parts.append(latent)
        if (folder/'wrapper.csv').exists():
            wr=pd.read_csv(folder/'wrapper.csv')
            for k,v in [('dataset',task['dataset']),('video',r['video']),('method',task['method']),('QP',task['qp'])]:wr[k]=v
            wrapper_parts.append(wr)
        summary={k:r[k] for k in ('dataset','video','method','QP','actual_bpp','kbps_20fps','real_bytes','bitstream_sha256','feature_path','feature_sha256','teacher_hash')}
        for scope,mask in [('P',latent.frame>0),('I',latent.frame==0),('all',latent.frame>=0)]:
            chunk=latent[mask]
            for metric in (*LATENTS,*QUALITY,'RMSE','channel_NRMSE','channel_Pearson','mean_difference','covariance_difference','PSNR','MS_SSIM',
                           *('quant_'+m for m in LATENTS)):
                summary[scope+'_'+metric]=chunk[metric].mean()
            summaries.append(dict(dataset=r['dataset'],video=r['video'],method=r['method'],QP=r['QP'],frame_scope=scope,
                                  actual_bpp=r['actual_bpp'],**{k[len(scope)+1:]:val for k,val in summary.items() if k.startswith(scope+'_')}))
        r['summary']=summary
        if r['noise_repeat']:noise.append(r['noise_repeat'])
    assert len(noise)==10 and all(n['status']=='PASS' and n['independent_repeats']==2 for n in noise)
    tolerance={m:max(n['max_absolute_metric_noise'][m] for n in noise) for m in (*LATENTS,'RMSE')}
    dump('numerical_noise_floor.json',dict(points=10,repeats_per_point=2,tolerance=tolerance,selection='pre-registered SHA256 ordering',evidence=noise))
    latents=pd.concat(latent_parts,ignore_index=True)
    latents.to_csv(ROOT/'latent_fidelity_per_frame.csv',index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'latent_fidelity_per_video_qp.csv',index=False)
    points=pd.DataFrame([r['summary'] for r in records]);points.to_csv(ROOT/'all_rd_points.csv',index=False)
    if wrapper_parts:pd.concat(wrapper_parts,ignore_index=True).to_csv(ROOT/'wrapper_diagnostics.csv',index=False)
    allpairs,primary=match_rates(points);allpairs=enrich(allpairs,points);primary=enrich(primary,points)
    allpairs.to_csv(ROOT/'all_matched_pairs.csv',index=False);primary.to_csv(ROOT/'primary_matched_pairs.csv',index=False)
    allpairs[allpairs.rate_match_level=='2pct'].to_csv(ROOT/'secondary_2pct_matched_pairs.csv',index=False)
    inv=[];sums=[];boot=[]
    for population,pairs in [('primary_1pct',primary),('all_eligible_1pct',allpairs[allpairs.rate_match_level=='1pct']),('secondary_2pct',allpairs[allpairs.rate_match_level=='2pct'])]:
        pairs=pairs.copy()
        if len(pairs):pairs['rate_band']=pd.qcut(pairs.mean_bpp.rank(method='first'),3,labels=['low','mid','high'])
        else:pairs['rate_band']=pd.Series(dtype=str)
        for scope in ('P','all'):
            for lm,qm in itertools.product(LATENTS,QUALITY):
                pairs['is_inversion']=(pairs['delta_'+scope+'_'+lm]>tolerance[lm])&(pairs['delta_'+scope+'_'+qm]<0)
                pairs['is_same_latent_gain']=(pairs['delta_'+scope+'_'+lm].abs()<=tolerance[lm])&(pairs['delta_'+scope+'_'+qm]<0)
                if population=='primary_1pct':
                    for r in pairs[pairs.is_inversion|pairs.is_same_latent_gain].to_dict('records'):
                        inv.append(dict(r,latent_metric=lm,quality_metric=qm,frame_scope=scope,population=population))
                groups=[('all','all',pairs)]+[(column,str(k),g) for column in ('dataset','method','rate_band','original_QP','method_QP') for k,g in pairs.groupby(column,observed=True)]
                for groupby,label,g in groups:
                    base=dict(population=population,frame_scope=scope,latent_metric=lm,quality_metric=qm,groupby=groupby,group=label,
                              matched_pair_count=len(g),inversion_count=int(g.is_inversion.sum()),same_latent_gain_count=int(g.is_same_latent_gain.sum()),
                              inversion_fraction=float(g.is_inversion.mean()) if len(g) else None)
                    sums.append(base)
                    if groupby in ('all','dataset','method') and len(g):
                        ids=g.video.unique();rng=np.random.default_rng(20261014)
                        means=g.groupby('video')[['is_inversion','is_same_latent_gain']].agg(['sum','count'])
                        values=[]
                        if len(ids)>1:
                            for _ in range(2000):
                                sample=means.loc[rng.choice(ids,len(ids),replace=True)]
                                values.append(sample[('is_inversion','sum')].sum()/sample[('is_inversion','count')].sum())
                        boot.append(dict(base,statistic='inversion_fraction',video_clusters=len(ids),
                              ci_low=float(np.quantile(values,.025)) if values else None,ci_high=float(np.quantile(values,.975)) if values else None))
    invfields=list(primary.columns)+['rate_band','is_inversion','is_same_latent_gain','latent_metric','quality_metric','frame_scope','population']
    pd.DataFrame(inv,columns=invfields).to_csv(ROOT/'inversion_pairs.csv',index=False)
    pd.DataFrame(sums).to_csv(ROOT/'inversion_summary.csv',index=False)
    # Within-video correlations: QP variation is explicitly labeled rate-confounded.
    corr=[];ratecorr=[]
    for scope in ('P','all'):
        for lm,qm in itertools.product(LATENTS,QUALITY):
            x,y=scope+'_'+lm,scope+'_'+qm
            for (dataset,video,method),g in points.groupby(['dataset','video','method']):
                corr.append(dict(dataset=dataset,video=video,method=method,frame_scope=scope,latent_metric=lm,quality_metric=qm,
                                 rate_controlled=False,**correlate(g[x],g[y]),ci_status='single video: cluster CI undefined'))
            for dataset,g in points.groupby('dataset'):
                # Pooled methods with fixed effects; refit rate model in each video resample.
                a,b=residuals(g,x,y)
                rec=dict(analysis='log_bpp_residuals_video_method_fixed_effects',dataset=dataset,frame_scope=scope,latent_metric=lm,quality_metric=qm,
                         **correlate(a,b),**regression_ci(g,x,y))
                ratecorr.append(rec);boot.append(dict(rec,statistic='residual_correlation'))
            for (dataset,method),g in allpairs.groupby(['dataset','method']):
                rec=dict(analysis='matched_local_delta_correlation_max2pct',dataset=dataset,method=method,frame_scope=scope,
                         latent_metric=lm,quality_metric=qm,**correlate(g['delta_'+x],g['delta_'+y]),
                         **cluster_ci(g,'delta_'+x,'delta_'+y))
                ratecorr.append(rec);boot.append(dict(rec,statistic='local_correlation'))
            # Aggregate within-video Spearman using video as the independent unit.
            temp=pd.DataFrame([r for r in corr if r['frame_scope']==scope and r['latent_metric']==lm and r['quality_metric']==qm])
            for (dataset,method),g in temp.groupby(['dataset','method']):
                vals=g.Spearman.dropna().to_numpy(float);rng=np.random.default_rng(20261014)
                draws=np.mean(rng.choice(vals,(2000,len(vals)),replace=True),axis=1) if len(vals)>1 else []
                boot.append(dict(statistic='mean_within_video_Spearman',dataset=dataset,method=method,frame_scope=scope,
                                 latent_metric=lm,quality_metric=qm,n=len(vals),video_clusters=len(vals),
                                 estimate=float(vals.mean()) if len(vals) else None,
                                 ci_low=float(np.quantile(draws,.025)) if len(draws) else None,ci_high=float(np.quantile(draws,.975)) if len(draws) else None))
    pd.DataFrame(corr).to_csv(ROOT/'correlation_by_video.csv',index=False)
    pd.DataFrame(ratecorr).to_csv(ROOT/'rate_controlled_correlation.csv',index=False)
    pd.DataFrame(boot).to_csv(ROOT/'bootstrap_summary.csv',index=False)
    # Fixed subset MCL29 is reaggregated, not counted as an independent dataset.
    subset=load(ROOT/'dataset_manifest.json')['mcl29']['names']
    p29=primary[(primary.dataset=='mcljcv')&primary.video.isin(subset)]
    p29.to_csv(ROOT/'mcl29_primary_matched_pairs.csv',index=False)
    m29=[]
    for lm,qm in itertools.product(LATENTS,QUALITY):
        m29.append(dict(dataset='mcljcv_exclude_overlap29',not_independent_dataset=True,latent_metric=lm,quality_metric=qm,
                        matched_pair_count=len(p29),inversion_count=int(((p29['delta_P_'+lm]>tolerance[lm])&(p29['delta_P_'+qm]<0)).sum())))
    pd.DataFrame(m29).to_csv(ROOT/'mcl29_inversion_summary.csv',index=False)
    # FID uses official Inception features, pooled across source videos.
    fidmod=module('v631a_fid_math',V29/'fid_statistics.py');fids=[]
    for dataset,group in points.groupby('dataset'):
        variants=[(dataset,group)]
        if dataset=='mcljcv':variants.append(('mcljcv_exclude_overlap29',group[group.video.isin(subset)]))
        for label,g in variants:
            for (method,qp),rows in g.groupby(['method','QP']):
                real=[];rec=[]
                for p in rows.feature_path:
                    with np.load(p) as features:real.append(features['real']);rec.append(features['reconstruction'])
                real=np.concatenate(real);rec=np.concatenate(rec)
                # Small populations use exact low-rank identity; larger pooled sets explicit covariance.
                value=fidmod.lowrank_unclamped(real,rec) if len(real)<2048 else fidmod.explicit_fid(real,rec)['FID']
                fids.append(dict(dataset=label,method=method,QP=qp,FID=value,samples=len(real),videos=len(rows),
                                 bpp=rows.actual_bpp.mean(),evidence_level='pooled_distribution_secondary',features='official pytorch-fid InceptionV3'))
    pd.DataFrame(fids).to_csv(ROOT/'fid_summary.csv',index=False)
    from plots_report import finish
    finish(points,primary,pd.DataFrame(inv,columns=invfields),pd.DataFrame(sums),tolerance)
    print('AGGREGATION_AND_REPORT_PASS',flush=True)


if __name__=='__main__':main()
