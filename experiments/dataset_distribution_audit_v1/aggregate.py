"""Raw tables, descriptive associations and training-only fitted coverage diagnostics."""
import itertools
import math
import collections
import numpy as np
from audit_io import *

BASE_FEATURES=['temporal_RGB_L1','temporal_RGB_MSE','sobel_mean','edge_density','laplacian_variance','high_frequency_energy',
    'flow_magnitude_mean','flow_spatial_variance','global_translation_normalized','global_rotation_magnitude','global_scale_change',
    'compensated_residual_L1','compensated_residual_MSE']
CODEC_FEATURES=[f'{m}_q{q}' for m in ('bpp','LPIPS','DISTS','FloLPIPS') for q in (0,4,9)]
FEATURES=BASE_FEATURES+CODEC_FEATURES
CANDIDATES=('ulong_train1024','ulong_unused1024','vimeo_train2048')

def finite(value):return value is not None and isinstance(value,(int,float,np.number)) and math.isfinite(float(value))
def summarize(rows,keys,groups):
    result=[]
    for group,own in groups(rows):
        for key in keys:
            vals=np.array([r[key] for r in own if finite(r.get(key))],float)
            if not len(vals):continue
            d=dict(**group,feature=key,N=len(vals),mean=float(vals.mean()),std=float(vals.std()),min=float(vals.min()),max=float(vals.max()))
            d.update({name:float(np.quantile(vals,q)) for name,q in [('p10',.1),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95)]});result.append(d)
    return result

def collect_sources():
    videos=load(ROOT/'manifests/all_sources.json')['videos'];combined=[];motions=[];embeddings=[];embedding_manifest=[]
    for v in videos:
        sid=v['sample_id'];source=load(ROOT/'parts/source'/f'{sid}.json');motion=load(ROOT/'parts/motion'/f'{sid}.json')
        assert source['status']==motion['status']=='PASS'
        for s in source['views']:
            m=next(x for x in motion['views'] if x['view']==s['view']);motions.append(m)
            combined.append(dict(dataset=v['dataset'],sample_id=sid,view=s['view'],frame_count=s['frame_count'],width=s['width'],height=s['height'],
                effective_fps=s['effective_fps'],**s['spatial'],**s['temporal'],**s['camera'],
                **{k:x for k,x in m.items() if k.startswith('flow_')}))
        if motion['embedding_present']:
            p=ROOT/'parts/embeddings'/f'{sid}.npy';e=np.load(p);assert e.shape==(2048,) and np.isfinite(e).all()
            embedding_manifest.append(dict(row=len(embeddings),sample_id=sid,dataset=v['dataset'],source_embedding_sha256=sha(p)));embeddings.append(e)
    write(ROOT/'motion_statistics.csv',motions);write(ROOT/'complete_source_statistics.csv',combined)
    motion_audit=load(ROOT/'motion_method_audit.json');motion_audit.update(status='PASS',videos=len(videos),views=2,statistic_aggregation='Frame-pair means, including mean of per-pair magnitude percentiles');dump(ROOT/'motion_method_audit.json',motion_audit)
    if embeddings:
        with (ROOT/'visual_embeddings.npy').open('wb') as f:np.save(f,np.stack(embeddings))
        write(ROOT/'visual_embedding_manifest.csv',embedding_manifest)
        audit=load(ROOT/'visual_embedding_audit.json');audit.update(status='PASS',videos=len(embeddings),output_sha256=sha(ROOT/'visual_embeddings.npy'));dump(ROOT/'visual_embedding_audit.json',audit)
    return combined

def collect_codec():
    candidates=load(ROOT/'manifests/codec_candidates.json')['videos'];expected={(v['sample_id'],q) for v in candidates for q in (0,4,9)}
    expected|={(r['sample_id'],r['QP']) for r in load(ROOT/'codec_reference_reuse_audit.json')['points']}
    rows=[];raw=[]
    for sid,q in sorted(expected):
        r=load(ROOT/'parts/codec'/sid/f'qp{q}.json');assert r['status']=='PASS' and r['original_only'] and r['force_zero_thres']==.12
        assert r['real_RANS'] and r['independent_decode_pass'] and Path(r['bitstream_path']).stat().st_size==r['real_bytes']==r['bytes_consumed']
        for key in ('bitstream','feature','frame_metrics'):assert sha(r[key+'_path'])==r[key+'_sha256']
        assert all(math.isfinite(float(r[m])) for m in ('bpp','kbps','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'))
        assert math.isclose(r['bpp'],r['real_bytes']*8/(r['frames']*r['width']*r['height']),rel_tol=1e-12)
        assert math.isclose(r['kbps'],r['bits_per_frame']*r['comparison_fps']/1000,rel_tol=1e-12)
        if r['reused']:assert sha(r['reused_point_path'])==r['reused_point_sha256']
        row={k:r[k] for k in ('dataset','sample_id','external_qp','real_bytes','bits_per_frame','bpp','kbps','comparison_fps','frames','width','height','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM','reused')}
        row['content_sample_id']=r.get('content_sample_id',sid);row['QP']=q;row['raw_point_path']=str(ROOT/'parts/codec'/sid/f'qp{q}.json');rows.append(row);raw.append(r)
    assert len({r['compression_hash_before'] for r in raw})==1
    write(ROOT/'codec_profile_per_video.csv',rows)
    def groups(rs):
        for d,q in sorted({(r['dataset'],r['QP']) for r in rs}):yield dict(dataset=d,QP=q),[r for r in rs if r['dataset']==d and r['QP']==q]
    write(ROOT/'codec_profile_summary.csv',summarize(rows,['bpp','kbps','bits_per_frame','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'],groups))
    features=[];proxies=[]
    for sid in sorted({r['sample_id'] for r in rows}):
        own={r['QP']:r for r in rows if r['sample_id']==sid};assert set(own)=={0,4,9}
        f=dict(dataset=own[0]['dataset'],sample_id=sid,content_sample_id=own[0]['content_sample_id'],codec_frame_count=own[0]['frames'])
        for q,r in own.items():
            for key in ('bpp','kbps','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'):f[f'{key}_q{q}']=r[key]
            f[f'R_q{q}']=r['bpp']
        f['rate_slope_q0_q9']=(f['bpp_q9']-f['bpp_q0'])/9;f['quality_slope_q0_q9']=(f['LPIPS_q9']-f['LPIPS_q0'])/9
        for m in ('LPIPS','DISTS','FloLPIPS','PSNR'):f[f'{m}_slope_q0_q9']=(f[f'{m}_q9']-f[f'{m}_q0'])/9
        features.append(f);proxies.append({k:f[k] for k in ('dataset','sample_id','content_sample_id','codec_frame_count',*[f'{m}_q{q}' for m in ('LPIPS','DISTS','FloLPIPS') for q in (0,4,9)])})
    write(ROOT/'codec_difficulty_features.csv',features);write(ROOT/'reconstruction_difficulty_proxy.csv',proxies)
    dump(ROOT/'codec_profile_audit.json',dict(status='PASS',points=len(rows),fresh_points=sum(not r['reused'] for r in rows),reused_points=sum(r['reused'] for r in rows),
        original_only=True,real_RANS=True,independent_decode=True,compression_hash=raw[0]['compression_hash_before'],
        R_definition='bpp',slope_definition='(value_q9-value_q0)/9; generic quality slope uses LPIPS',
        reconstruction_difficulty_proxy_not_ground_truth=True,Vimeo7frames_user_approved=True,frame_length_difference_preserved=True))
    return features

def failure_association(combined,codec):
    sys.path.insert(0,str(V52));from rd_analysis import compare
    from scipy.stats import pearsonr,spearmanr,rankdata
    output=[];eq=[]
    source={r['sample_id']:r for r in combined if r['view']=='standardized_content_view'}
    frozen=load(V62B/'source_manifest.json')['videos'];table={d:read(V62B/'results'/d/'per_sequence_all_qp.csv') for d in ('uvg','virat720','virat480')}
    for c in [r for r in codec if r['dataset'] not in CANDIDATES]:
        sr=source[c['content_sample_id']];d='uvg' if c['dataset']=='uvg_reference_test' else c['dataset']
        origin=next(v for v in frozen if v['dataset']==d and ('uvg_'+v['name']==sr['sample_id'] if d=='uvg' else 'virat_'+str(v['video_index'])==sr['sample_id']))
        rows=[r for r in table[d] if int(r['video_index'])==origin['video_index']]
        row=dict(sr,**{k:v for k,v in c.items() if k not in ('dataset','sample_id')});row.update(dataset=c['dataset'],sample_id=c['sample_id'],sequence=origin['name'])
        for m in ('LPIPS','DISTS','FloLPIPS'):
            e,_=compare([r for r in rows if r['method']=='original'],[r for r in rows if r['method']=='v62'],m,'original','v62')
            value=e['mean_equal_rate_delta'] if e['status']=='valid' else None
            row['equal_rate_'+m+'_delta']=value;row['equal_rate_'+m+'_status']=e['status'];eq.append(dict(dataset=d,sequence=origin['name'],**e))
        output.append(row)
    write(ROOT/'test_failure_association.csv',output);write(ROOT/'test_per_sequence_equal_rate.csv',eq)
    correlations=[];ranks=[]
    for d in ('uvg_reference_test','virat720','virat480'):
        own=[r for r in output if r['dataset']==d]
        for target in ('equal_rate_LPIPS_delta','equal_rate_DISTS_delta','equal_rate_FloLPIPS_delta'):
            for key in FEATURES:
                valid=[r for r in own if finite(r.get(key)) and finite(r.get(target))];x=np.array([r[key] for r in valid]);y=np.array([r[target] for r in valid])
                usable=len(valid)>=3 and x.std()>0 and y.std()>0
                correlations.append(dict(dataset=d,feature=key,target=target,N=len(valid),Pearson=float(pearsonr(x,y).statistic) if usable else None,
                    Spearman=float(spearmanr(x,y).statistic) if usable else None,status='descriptive_only' if usable else 'undefined_constant_or_insufficient',significance_claim=False))
                for r,rx,ry in zip(valid,rankdata(x),rankdata(y)):ranks.append(dict(dataset=d,sequence=r['sequence'],feature=key,target=target,feature_rank=float(rx),target_rank=float(ry),N=len(valid)))
    write(ROOT/'test_failure_correlations.csv',correlations);write(ROOT/'test_failure_ranks.csv',ranks)
    return output

def coverage(combined,codec):
    from scipy.spatial.distance import cdist
    from scipy.stats import wasserstein_distance
    source={r['sample_id']:r for r in combined if r['view']=='standardized_content_view'};merged=[];excluded=[]
    for c in codec:
        # VIRAT counted once at content level, with720 codec descriptor.480 retained as sensitivity projection, not doubled in source aggregates.
        if c['dataset']=='virat480':continue
        s=source[c['content_sample_id']];r=dict(s,**{k:v for k,v in c.items() if k not in ('dataset','sample_id')})
        missing=[f for f in FEATURES if not finite(r.get(f))]
        if missing:excluded.append(dict(sample_id=r['sample_id'],dataset=r['dataset'],missing_features=missing));continue
        merged.append(r)
    fit=[r for r in merged if r['dataset'] in CANDIDATES];assert fit
    x=np.asarray([[r[f] for f in FEATURES] for r in fit]);mu=x.mean(0);sd=x.std(0);keep=sd>1e-12
    cols=[f for f,b in zip(FEATURES,keep) if b];z=(x[:,keep]-mu[keep])/sd[keep]
    allx=np.asarray([[r[f] for f in cols] for r in merged]);allz=(allx-mu[keep])/sd[keep]
    dump(ROOT/'feature_scaler.json',dict(status='PASS',features=cols,mean=mu[keep].tolist(),std=sd[keep].tolist(),ddof=0,
        fit_sample_ids=[r['sample_id'] for r in fit],fit_datasets=sorted({r['dataset'] for r in fit}),fit_count=len(fit),
        dropped_constant_features=[f for f,b in zip(FEATURES,keep) if not b],reference_fit=False,missing_imputation=False,
        covariance_condition_threshold=1e8,primary_view='standardized_content_view',codec_support_uses_profiled_subsets_not_all1024_or2048=True))
    dump(ROOT/'complete_feature_cohort_audit.json',dict(status='PASS',complete_count=len(merged),excluded_missing=excluded,
        complete_counts=dict(collections.Counter(r['dataset'] for r in merged)),VIRAT_content_deduplication='720p descriptor only;480 codec results retained separately',
        differing_codec_frame_counts=dict(collections.Counter(r['codec_frame_count'] for r in merged))))
    wide=[dict(r,**{'z_'+f:float(row[i]) for i,f in enumerate(cols)}) for r,row in zip(merged,allz)];write(ROOT/'complete_feature_cohort.csv',wide)
    center=z.mean(0);_,singular,vt=np.linalg.svd(z-center,full_matrices=False);variance=singular**2/(len(z)-1);components=vt[:2].copy()
    for i in range(len(components)):
        if components[i,np.argmax(np.abs(components[i]))]<0:components[i]*=-1
    coords=(allz-center)@components.T
    write(ROOT/'pca_coordinates.csv',[dict(dataset=r['dataset'],sample_id=r['sample_id'],role='candidate_training' if r['dataset'] in CANDIDATES else 'reference',PC1=float(c[0]),PC2=float(c[1])) for r,c in zip(merged,coords)])
    write(ROOT/'pca_explained_variance.csv',[dict(component=i+1,variance=float(v),explained_variance_ratio=float(v/variance.sum())) for i,v in enumerate(variance)])
    dump(ROOT/'pca_fit_audit.json',dict(status='PASS',fit_sample_ids=[r['sample_id'] for r in fit],reference_fit=False,components=components.tolist(),center=center.tolist()))
    masks={label:np.array([r['dataset'] in ds for r in merged]) for label,ds in {'ulong_train1024':{'ulong_train1024'},'vimeo_train2048':{'vimeo_train2048'},'ulong_plus_vimeo':{'ulong_train1024','vimeo_train2048'}}.items()}
    distances=[];percentiles=[];near_ulong=[];near_vimeo=[]
    refs=[i for i,r in enumerate(merged) if r['dataset'] not in CANDIDATES]
    for label,mask in masks.items():
        support=allz[mask];ids=np.flatnonzero(mask);assert len(support)>=20
        covariance=np.cov(support,rowvar=False);condition=float(np.linalg.cond(covariance));valid_cov=math.isfinite(condition) and condition<1e8 and len(support)>len(cols)
        inv=np.linalg.inv(covariance) if valid_cov else None
        for ri in refs:
            r=merged[ri];ds=np.linalg.norm(support-allz[ri],axis=1);order=np.argsort(ds,kind='stable');delta=allz[ri]-support.mean(0)
            distances.append(dict(reference=r['sample_id'],dataset=r['dataset'],support=label,support_complete_N=len(support),nearest_distance=float(ds[order[0]]),
                mean_knn5=float(ds[order[:5]].mean()),mean_knn20=float(ds[order[:20]].mean()),Mahalanobis=float(np.sqrt(max(0,delta@inv@delta))) if valid_cov else None,
                covariance_condition_number=condition if math.isfinite(condition) else None,Mahalanobis_status='valid' if valid_cov else 'invalid_covariance_condition_or_sample_count'))
            for j,f in enumerate(cols):
                vals=support[:,j];pct=100*(np.sum(vals<allz[ri,j])+.5*np.sum(vals==allz[ri,j]))/len(vals)
                percentiles.append(dict(reference=r['sample_id'],dataset=r['dataset'],support=label,feature=f,percentile_position=float(pct),support_N=len(vals)))
            if r['dataset']=='uvg_reference_test' and label in ('ulong_train1024','vimeo_train2048'):
                dest=near_ulong if label=='ulong_train1024' else near_vimeo
                for rank,k in enumerate(order[:10],1):
                    nr=merged[ids[k]];row=dict(sequence=r['sample_id'],neighbor=nr['sample_id'],rank=rank,distance=float(ds[k]),support=label)
                    row.update({f'difference_{f}':float(r[f]-nr[f]) for f in cols});row.update({f'standardized_difference_{f}':float(allz[ri,j]-support[k,j]) for j,f in enumerate(cols)});dest.append(row)
    write(ROOT/'reference_to_training_distance.csv',distances);write(ROOT/'reference_feature_percentiles.csv',percentiles)
    write(ROOT/'uvg_nearest_ulong.csv',near_ulong);write(ROOT/'uvg_nearest_vimeo.csv',near_vimeo)
    pairwise=[]
    for a,b in [('ulong_train1024','vimeo_train2048'),('ulong_train1024','uvg_reference_test'),('ulong_train1024','virat_reference'),('vimeo_train2048','uvg_reference_test'),('vimeo_train2048','virat_reference')]:
        aa=allz[[r['dataset']==a for r in merged]];bb=allz[[r['dataset']==b for r in merged]];assert len(aa) and len(bb)
        norm=float(np.linalg.norm(aa.mean(0)-bb.mean(0)))
        for j,f in enumerate(cols):pairwise.append(dict(dataset_a=a,dataset_b=b,N_a=len(aa),N_b=len(bb),feature=f,standardized_mean_distance=norm,
            standardized_feature_mean_difference=float(bb[:,j].mean()-aa[:,j].mean()),Wasserstein1_standardized=float(wasserstein_distance(aa[:,j],bb[:,j])),cohort='complete_codec_profiled_subsets'))
    write(ROOT/'dataset_pairwise_distance.csv',pairwise)
    audit=load(ROOT/'data_leakage_audit.json');audit.update(status='PASS',scaler_fitted_reference=False,PCA_fitted_reference=False,
        fit_sample_ids=[r['sample_id'] for r in fit],nearest_neighbors_not_added_to_training=True,missing_fields_not_zero_imputed=True);dump(ROOT/'data_leakage_audit.json',audit)

def main():
    combined=collect_sources();codec=collect_codec();failure_association(combined,codec);coverage(combined,codec)
    keys=sorted({k for r in combined for k,v in r.items() if finite(v)}-{'frame_count','width','height'})
    def groups(rows):
        for d,view in sorted({(r['dataset'],r['view']) for r in rows}):yield dict(dataset=d,view=view),[r for r in rows if r['dataset']==d and r['view']==view]
    write(ROOT/'dataset_statistics_summary.csv',summarize(combined,keys,groups))
    dump(ROOT/'aggregation_audit.json',dict(status='PASS',source_view_rows=len(combined),codec_samples=len(codec),no_scientific_conclusions=True))
    print('AGGREGATION PASS',flush=True)
if __name__=='__main__':main()
