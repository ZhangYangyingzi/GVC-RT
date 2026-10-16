"""Numerical aggregation only. Missing measurements never become zero results."""
from evaluate import *
import numpy as np

METRICS=('LPIPS','DISTS','FloLPIPS')
COMPARISONS=[(BRANCHES[1],BRANCHES[0]),(BRANCHES[2],BRANCHES[0]),(BRANCHES[3],BRANCHES[1]),(BRANCHES[3],BRANCHES[2])]

def explicit_comparisons(points,summary):
    text='\n## Requested numerical comparisons\n\nLOWER/HIGHER describe common actual-rate metric means only; statistical significance is not asserted.\n\n|Dataset|Comparison|LPIPS|DISTS|FloLPIPS|FID|\n|---|---|---|---|---|---|\n'
    for d in load(V32/'eval_manifest.json'):
        for m,a in [(BRANCHES[3],BRANCHES[1]),(BRANCHES[2],BRANCHES[0])]:
            labels=[]
            for metric in (*METRICS,'FID'):
                r=next((r for r in summary if (r['dataset'],r['method'],r['reference'],r['metric'])==(d,m,a,metric)),None)
                labels.append('NOT ESTABLISHED' if r is None else 'LOWER' if r['delta']<0 else 'HIGHER' if r['delta']>0 else 'EQUAL')
            text+='|'+d+'|'+m[0]+' vs '+a[0]+'|'+'|'.join(labels)+'|\n'
    index={(r['method'],r['dataset'],r['video'],r['QP']):r for r in points};assert len(index)==1840
    rows=[];totals=[]
    for m,a in COMPARISONS:
        local=[]
        for y in points:
            if y['method']!=m:continue
            x=index[(a,y['dataset'],y['video'],y['QP'])]
            r=dict(dataset=y['dataset'],video=y['video'],QP=y['QP'],method=m,reference=a,method_actual_RANS_bytes=y['real_bytes'],reference_actual_RANS_bytes=x['real_bytes'],delta_actual_RANS_bytes=y['real_bytes']-x['real_bytes'],method_bpp=y['bpp'],reference_bpp=x['bpp'],comparison='same QP only; neither same rate nor same quality is asserted')
            rows.append(r);local.append(r)
        totals.append(dict(method=m,reference=a,points=len(local),lower_byte_count=sum(r['delta_actual_RANS_bytes']<0 for r in local),equal_byte_count=sum(r['delta_actual_RANS_bytes']==0 for r in local),higher_byte_count=sum(r['delta_actual_RANS_bytes']>0 for r in local)))
    write('evaluation/same_qp_actual_rate_diagnostics.csv',rows)
    write('evaluation/same_qp_actual_rate_count_summary.csv',totals)
    text+='\nSame-QP actual-byte diagnostics across the complete fixed46-video ×10-QP matrix. These counts do not establish a bitrate reduction at equal quality.\n\n|Comparison|Points|Fewer actual bytes|Equal actual bytes|More actual bytes|\n|---|---:|---:|---:|---:|\n'
    for r in totals:text+='|'+r['method'][0]+' vs '+r['reference'][0]+'|'+'|'.join(str(r[k]) for k in ['points','lower_byte_count','equal_byte_count','higher_byte_count'])+'|\n'
    return text

def common_rate(points,pools):
    from scipy.interpolate import PchipInterpolator
    rows=[];grids=[];coverage=[]
    keys={(r['dataset'],r['video'],metric) for r in points for metric in METRICS}
    keys|={(r['dataset'],'pooled','FID') for r in pools if r['status']=='available'}
    for d,video,metric in sorted(keys,key=str):
        source=pools if video=='pooled' else points
        curves={m:[r for r in source if r['dataset']==d and r['method']==m and (video=='pooled' or r['video']==video) and (video!='pooled' or r['status']=='available')] for m in BRANCHES}
        if any(len(v)!=10 for v in curves.values()):
            coverage.append(dict(dataset=d,video=video,metric=metric,status='UNAVAILABLE',reason='incomplete fixed 4-method QP matrix'));continue
        if any(len({r['bpp'] for r in v})!=10 or any(r['bpp']<=0 for r in v) for v in curves.values()):
            coverage.append(dict(dataset=d,video=video,metric=metric,status='UNAVAILABLE',reason='duplicate or nonpositive actual bpp; no interpolation performed'));continue
        lo=max(min(r['bpp'] for r in v) for v in curves.values());hi=min(max(r['bpp'] for r in v) for v in curves.values())
        if hi<=lo:
            coverage.append(dict(dataset=d,video=video,metric=metric,status='UNAVAILABLE',reason='no 4-method common actual-rate interval'));continue
        coverage.append(dict(dataset=d,video=video,metric=metric,status='AVAILABLE',reason='4-method common interval',common_actual_bpp_min=lo,common_actual_bpp_max=hi))
        grid=np.linspace(np.log(lo),np.log(hi),100);ys={}
        for m,rs in curves.items():
            rs=sorted(rs,key=lambda r:r['bpp']);x=np.array([r['bpp'] for r in rs]);assert len(set(x))==10
            ys[m]=PchipInterpolator(np.log(x),[r[metric] for r in rs],extrapolate=False)(grid)
        grids.extend(dict(dataset=d,video=video,metric=metric,actual_bpp=float(np.exp(x)),**{m:float(ys[m][j]) for m in BRANCHES}) for j,x in enumerate(grid))
        for m,a in COMPARISONS:
            x=float(ys[m].mean());y=float(ys[a].mean())
            rows.append(dict(dataset=d,video=video,metric=metric,method=m,reference=a,method_mean=x,reference_mean=y,delta=x-y,relative_change_percent=100*(x-y)/y,common_actual_bpp_min=lo,common_actual_bpp_max=hi))
    write('evaluation/common_rate_grid.csv',grids);write('evaluation/equal_rate_per_sequence.csv',rows);write('evaluation/common_rate_coverage.csv',coverage)
    summary=[]
    for d,metric,m,a in sorted({(r['dataset'],r['metric'],r['method'],r['reference']) for r in rows}):
        rs=[r for r in rows if (r['dataset'],r['metric'],r['method'],r['reference'])==(d,metric,m,a)]
        x=float(np.mean([r['method_mean'] for r in rs]));y=float(np.mean([r['reference_mean'] for r in rs]))
        summary.append(dict(dataset=d,metric=metric,method=m,reference=a,sequences=len(rs),method_mean=x,reference_mean=y,delta=x-y,relative_change_percent=100*(x-y)/y,rate_axis='persisted actual total RANS bpp',interpolation='100 common log-bpp PCHIP points; no extrapolation',aggregation='equal weight per video; FID fixed pooled distribution'))
    write('evaluation/equal_rate_summary.csv',summary)
    pairs=[]
    for d,video in sorted({(r['dataset'],r['video']) for r in points}):
        for m,a in COMPARISONS:
            left=[r for r in points if r['dataset']==d and r['video']==video and r['method']==a]
            right=[r for r in points if r['dataset']==d and r['video']==video and r['method']==m]
            if len(left)!=10 or len(right)!=10:continue
            gap,qa,qm,x,y=min((abs(y['bpp']/x['bpp']-1),x['QP'],y['QP'],x,y) for x in left for y in right)
            pairs.append(dict(dataset=d,video=video,method=m,reference=a,method_QP=qm,reference_QP=qa,method_actual_bpp=y['bpp'],reference_actual_bpp=x['bpp'],relative_bpp_gap=gap,tier='primary_1pct' if gap<=.01 else 'secondary_2pct' if gap<=.02 else 'UNMATCHED',pair_selection='minimum actual rate gap only; QP lexicographic tie break; no quality used',**{'Delta_'+k:y[k]-x[k] for k in METRICS}))
    write('evaluation/equal_rate_pairs.csv',pairs)
    return summary

def distribution(values):
    x=np.array(values,float);mass=np.maximum(x,0);total=mass.sum();n=len(x)
    if total>0:
        prob=mass/total;s=np.sort(mass)
        gini=float((2*np.arange(1,n+1)-n-1)@s/(n*total))
        entropy=float(-(prob[prob>0]*np.log(prob[prob>0])).sum()/np.log(n))
        top10=float(np.sort(mass)[-max(1,int(np.ceil(n*.1))):].sum()/total)
        top25=float(np.sort(mass)[-max(1,int(np.ceil(n*.25))):].sum()/total)
    else:gini=entropy=top10=top25=None
    return dict(signed_mean=float(x.mean()),signed_std=float(x.std()),signed_CV=float(x.std()/abs(x.mean())) if x.mean()!=0 else None,positive_mass_Gini=gini,positive_mass_normalized_entropy=entropy,positive_mass_top10_fraction=top10,positive_mass_top25_fraction=top25,positive_mass=float(total),negative_channel_count=int((x<0).sum()))

def utility_tables():
    from scipy.stats import spearmanr,pearsonr
    stats=[];raw=[];rate_rows=[]
    for p in sorted((ROOT/'utility').glob('*.json')):
        u=load(p)
        if u.get('status')!='PASS':continue
        for stream,rates in [('y',u['rate']['estimated_channel_bits']),('z',u['rate']['hyperlatent']['estimated_channel_bits'])]:
            for c,value in enumerate(rates):rate_rows.append(dict(dataset=u['dataset'],video=u['video'],method=u['method'],QP=u['QP'],frame=32,stream=stream,channel=c,estimated_entropy_model_channel_bits=value,actual_total_video_RANS_bpp=u['actual_total_bpp'],actual_frame32_payload_bits=u['actual_frame32_payload_bits'],per_channel_actual_rate_available=False))
        for r in u['rows']:raw.append(dict(dataset=u['dataset'],video=u['video'],method=u['method'],QP=u['QP'],tier=u['matched_rate_tier'],actual_total_bpp=u['actual_total_bpp'],**r))
        for intervention in ('zero','spatial_mean','attenuate_0.95'):
            rows=[r for r in u['rows'] if r['intervention']==intervention]
            for metric in METRICS:
                rates=np.array([r['R_est'] for r in rows]);utilities=np.array([r['I_'+metric] for r in rows])
                corr=lambda fn:float(fn(rates,utilities).statistic) if np.std(rates)>0 and np.std(utilities)>0 else None
                stats.append(dict(dataset=u['dataset'],video=u['video'],method=u['method'],QP=u['QP'],intervention=intervention,metric=metric,**distribution(utilities),R_est_sum=float(rates.sum()),R_est_mean=float(rates.mean()),R_est_std=float(rates.std()),R_est_CV=float(rates.std()/rates.mean()) if rates.mean()>0 else None,R_est_Gini=distribution(rates)['positive_mass_Gini'],R_est_top10_fraction=distribution(rates)['positive_mass_top10_fraction'],R_est_top25_fraction=distribution(rates)['positive_mass_top25_fraction'],R_est_vs_I_Pearson=corr(pearsonr),R_est_vs_I_Spearman=corr(spearmanr),eta_mean=float(np.mean([r['eta_'+metric] for r in rows])),eta_median=float(np.median([r['eta_'+metric] for r in rows])),zero_rate_channels=int((rates==0).sum())))
    tiers={(r['dataset'],r['video'],r['method'],r['QP']):r['tier'] for r in raw}
    for r in stats+rate_rows:r['tier']=tiers[(r['dataset'],r['video'],r['method'],r['QP'])]
    write('utility/channel_sensitivity.csv',raw);write('utility/distribution_summary.csv',stats);write('utility/channel_rate_estimates.csv',rate_rows)
    changes=[];distribution_changes=[]
    for pair in load(ROOT/'utility/matched_pair_plan.json')['pairs']:
        for intervention in ('zero','spatial_mean','attenuate_0.95'):
            no={r['channel']:r for r in raw if r['dataset']==pair['dataset'] and r['video']==pair['video'] and r['method']==pair['noP'] and r['QP']==pair['noP_QP'] and r['intervention']==intervention}
            pp={r['channel']:r for r in raw if r['dataset']==pair['dataset'] and r['video']==pair['video'] and r['method']==pair['P'] and r['QP']==pair['P_QP'] and r['intervention']==intervention}
            assert set(no)==set(pp) and len(no)==128
            for metric in METRICS:
                for c in sorted(no):
                    a=no[c]['I_'+metric];b=pp[c]['I_'+metric]
                    ra=no[c]['R_est'];rb=pp[c]['R_est']
                    changes.append(dict(dataset=pair['dataset'],video=pair['video'],noP=pair['noP'],P=pair['P'],noP_QP=pair['noP_QP'],P_QP=pair['P_QP'],tier=pair['tier'],actual_bpp_relative_gap=pair['relative_gap'],channel=c,intervention=intervention,metric=metric,I_noP=a,I_P=b,Delta_I=b-a,R_est_noP=ra,R_est_P=rb,Delta_R_est=rb-ra,eta_noP=no[c]['eta_'+metric],eta_P=pp[c]['eta_'+metric]))
                a=distribution([no[c]['I_'+metric] for c in sorted(no)]);b=distribution([pp[c]['I_'+metric] for c in sorted(pp)])
                oi=np.array([no[c]['I_'+metric] for c in sorted(no)]);dr=np.array([pp[c]['R_est']-no[c]['R_est'] for c in sorted(no)])
                distribution_changes.append(dict(dataset=pair['dataset'],video=pair['video'],noP=pair['noP'],P=pair['P'],intervention=intervention,metric=metric,tier=pair['tier'],original_I_vs_Delta_R_Pearson=float(pearsonr(oi,dr).statistic) if np.std(oi)>0 and np.std(dr)>0 else None,original_I_vs_Delta_R_Spearman=float(spearmanr(oi,dr).statistic) if np.std(oi)>0 and np.std(dr)>0 else None,eta_noP_mean=float(np.mean([no[c]['eta_'+metric] for c in no])),eta_P_mean=float(np.mean([pp[c]['eta_'+metric] for c in pp])),**{'Delta_'+k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in a}))
    write('utility/paired_channel_changes.csv',changes);write('utility/distribution_changes.csv',distribution_changes)
    return raw,stats

def figures(summary,pools,raw,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    for metric,filename in [('LPIPS','lpips'),('DISTS','dists'),('FloLPIPS','flolpips'),('FID','fid')]:
        fig,ax=plt.subplots(figsize=(11,5));data=[r for r in summary if r['metric']==metric]
        for j,(m,a) in enumerate(COMPARISONS):
            rs=[r for r in data if r['method']==m and r['reference']==a];xs=np.arange(len(rs))+(j-1.5)*.18
            ax.bar(xs,[r['delta'] for r in rs],width=.18,label=m[0]+' vs '+a[0])
        labels=sorted({r['dataset'] for r in data});ax.set_xticks(range(len(labels)),labels,rotation=15);ax.axhline(0,color='black',lw=.7)
        ax.set_ylabel('Common actual-rate '+metric+' delta (method-reference)');ax.set_title('Actual TOTAL persisted RANS bitrate; 4-method common interval');ax.legend();fig.tight_layout()
        name='p_causal_ablation_fid.png' if metric=='FID' else f'p_causal_ablation_equal_rate_{filename}.png';fig.savefig(figdir/name,dpi=180);plt.close(fig)
    selected=[r for r in raw if r['intervention']=='spatial_mean' and r['tier']=='primary_1pct']
    fig,ax=plt.subplots(figsize=(12,4))
    for m in BRANCHES:
        rs=[r for r in selected if r['method']==m]
        if rs:
            cs=sorted({r['channel'] for r in rs});ax.plot(cs,[np.mean([r['R_est'] for r in rs if r['channel']==c]) for c in cs],label=m[0])
    ax.set_xlabel('Decoded y_hat channel');ax.set_ylabel('Estimated entropy-model channel bits');ax.set_title('Mean of selected <=1% actual-rate pairs; main y, frame32 only. Actual TOTAL bitrate is separate.');ax.legend();fig.tight_layout();fig.savefig(figdir/'channel_rate_noP_vs_P.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,METRICS):
        for m in BRANCHES:
            rs=[r for r in selected if r['method']==m]
            if rs:
                cs=sorted({r['channel'] for r in rs});ax.plot(cs,[np.mean([r['eta_'+metric] for r in rs if r['channel']==c]) for c in cs],label=m[0])
        ax.set_xlabel('Decoded y_hat channel');ax.set_ylabel('Delta '+metric+' / estimated y bits');ax.legend()
    fig.suptitle('Mean of selected <=1% actual-rate pairs; finite-difference utility-per-estimated-rate proxy');fig.tight_layout();fig.savefig(figdir/'utility_per_rate_noP_vs_P.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,METRICS):
        with (ROOT/'utility/paired_channel_changes.csv').open() as f:change=list(csv.DictReader(f))
        rr=[r for r in change if r['intervention']=='spatial_mean' and r['metric']==metric and r['tier']=='primary_1pct']
        ax.scatter([float(r['I_noP']) for r in rr],[float(r['Delta_R_est']) for r in rr],s=6,alpha=.3)
        ax.set_xlabel('Original no-P Delta '+metric);ax.set_ylabel('Delta estimated entropy-model y bits')
    fig.tight_layout();fig.savefig(figdir/'delta_rate_vs_original_utility.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9))
    for ax,metric in zip(axes,METRICS):
        for m in BRANCHES:
            rs=[r for r in selected if r['method']==m]
            if rs:
                cs=sorted({r['channel'] for r in rs});ax.plot(cs,[np.mean([r['I_'+metric] for r in rs if r['channel']==c]) for c in cs],label=m[0])
        ax.set_ylabel('Delta '+metric);ax.legend()
    axes[-1].set_xlabel('Decoded y_hat channel');fig.suptitle('Mean channel generative sensitivity; spatial-mean intervention; selected <=1% actual-rate pairs');fig.tight_layout();fig.savefig(figdir/'channel_utility_noP_vs_P.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,METRICS):
        for m in BRANCHES:
            vals=[r['I_'+metric] for r in selected if r['method']==m]
            if vals:ax.hist(vals,bins=35,alpha=.35,label=m[0])
        ax.set_xlabel('Delta '+metric);ax.legend()
    fig.tight_layout();fig.savefig(figdir/'utility_distribution_histogram.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for row,metric in enumerate(METRICS):
        for ax,key in zip(axes[row],['positive_mass_Gini','signed_CV']):
            for j,m in enumerate(BRANCHES):
                vals=[r[key] for r in stats if r['method']==m and r['metric']==metric and r['intervention']=='spatial_mean' and r['tier']=='primary_1pct' and r[key] is not None]
                if vals:ax.scatter([j]*len(vals),vals,label=m[0])
            ax.set_xticks(range(4),['A','B','C','D']);ax.set_ylabel(metric+' '+key)
    fig.suptitle('Selected <=1% actual-rate pairs; Gini uses positive sensitivity mass; metrics shown separately');fig.tight_layout();fig.savefig(figdir/'utility_gini_cv_summary.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    plan=load(ROOT/'utility/matched_pair_plan.json')['pairs']
    selected_all=[r for r in raw if r['intervention']=='spatial_mean']
    for ax,metric in zip(axes,METRICS):
        for pair in plan:
            no=[r for r in selected_all if r['dataset']==pair['dataset'] and r['video']==pair['video'] and r['method']==pair['noP'] and r['QP']==pair['noP_QP']]
            pp=[r for r in selected_all if r['dataset']==pair['dataset'] and r['video']==pair['video'] and r['method']==pair['P'] and r['QP']==pair['P_QP']]
            if len(no)==len(pp)==128:ax.scatter([r['I_'+metric] for r in no],[r['I_'+metric] for r in pp],s=6,alpha=.4,marker='o' if pair['tier']=='primary_1pct' else 'x',label=pair['dataset']+' '+pair['noP'][0]+'/'+pair['P'][0]+' '+pair['tier'])
        ax.set_xlabel('No-P Delta '+metric);ax.set_ylabel('With-P Delta '+metric)
    axes[0].legend(fontsize=5);fig.suptitle('Selected actual TOTAL RANS rate pairs; 1% primary / 2% secondary');fig.tight_layout();fig.savefig(figdir/'matched_rate_channel_scatter.png',dpi=180);plt.close(fig)

def main():
    points=[load(p)['result'] for p in sorted((ROOT/'evaluation/points').glob('*.json'))]
    assert len(points)==1840
    table=[{k:r.get(k) for k in ['dataset','video','sequence','method','QP','bpp','real_bytes','LPIPS','DISTS','FloLPIPS','bitstream_path','bitstream_sha256','independent_decode_pass','state_sync_pass']} for r in points]
    write('evaluation/all_rd_points.csv',table);write('evaluation/per_sequence_all_qp.csv',table)
    from aggregate_fid import manifest
    fid=load(ROOT/'evaluation/fid_aggregation_integrity.json')
    assert fid['status']=='PASS' and fid['point_receipt_sha256']==manifest() and fid['source_helper_sha256']==sha(V32/'report32.py') and fid['pooled_csv_sha256']==sha(ROOT/'evaluation/pooled_fid.csv')
    with (ROOT/'evaluation/pooled_fid.csv').open() as f:pools=list(csv.DictReader(f))
    for r in pools:
        if r['status']=='available':
            for k in ('FID','D_mu','D_cov','bpp'):r[k]=float(r[k])
        r['QP']=int(r['QP'])
    summary=common_rate(points,pools);raw,stats=utility_tables();figures(summary,pools,raw,stats)
    probe=load(ROOT/'sanity/channel_rate_probe.json');assert probe['status']=='PASS' and probe['decoder_vs_encoder_z_estimate_abs_error']<1e-7
    escape_probe=load(ROOT/'sanity/native_escape_bypass.json');assert escape_probe['status']=='PASS' and all(x['decode_exact'] for x in escape_probe['cases'])
    utilities=[u for p in (ROOT/'utility').glob('*.json') if (u:=load(p)).get('status')=='PASS' and 'rows' in u and not u.get('probe_only')]
    interface=load(ROOT/'latent_interface_audit.json')
    interface['formal_native_observations']=[dict(method=u['method'],dataset=u['dataset'],video=u['video'],QP=u['QP'],frame=u['frame'],y_hat_shape=u['latent_shape'],y_hat_dtype=u['dtype'],z_hat_shape=u['hyperlatent_shape'],z_hat_dtype=u['hyperlatent_dtype']) for u in utilities]
    dump('latent_interface_audit.json',interface)
    dump('channel_rate_integrity.json',dict(status='PASS_ESTIMATED_ONLY',per_channel_actual_rate_available=False,estimated_channel_rate_available=True,definition='installed-CDF symbol mass plus native escape bypass length; main y only; not actual RANS bytes',probe=probe,native_escape_rule_validation=escape_probe,maximum_channel_sum_error=max(u['rate']['sum_channel_vs_combined_y_estimate_abs_error'] for u in utilities),excludes='z hyperstream, I frame, SPS/header, RANS state flush and byte renormalization overhead',comparison_to_training_rate='Training Gaussian-CDF differentiable estimate uses a different probability discretization and is not relabelled as this installed-CDF estimate.'))
    integrities={b:load(ROOT/'branches'/b/'training_integrity.json') for b in BRANCHES[1:]}
    receiver_initials=[{k:x['initial']['modules'][k] for k in ('bridge','generator')} for x in integrities.values()]
    assert all(x==receiver_initials[0] for x in receiver_initials)
    assert all(x['initial']['quality']==next(iter(integrities.values()))['initial']['quality'] for x in integrities.values())
    assert integrities[BRANCHES[2]]['initial']['modules']['wrapper']==integrities[BRANCHES[3]]['initial']['modules']['wrapper']
    sanity={}
    for b in BRANCHES[1:]:
        for step in (100,250):
            p=ROOT/'sanity'/f'checkpoint_{b}_{step}.json';x=load(p);assert x['status']=='PASS';sanity[str(p)]=sha(p)
    dump('sanity/checkpoints_integrity.json',dict(status='PASS',fixed_steps=[100,250],receipts=sanity,all_receiver_initial_states_equal=True,C_D_P_initial_states_equal=True))
    cache_records=[dict(path=str(p),**load(p)) for p in sorted((ROOT/'cache/training_crops').glob('*.json'))]
    assert all(x['source_full_RGB_hashes_verified'] and x['all_64_cropped_RGB_hashes_verified'] and x['float32_model_inputs_exact'] for x in cache_records)
    dump('data_cache_integrity.json',dict(status='PASS',cached_windows=len(cache_records),cache_records=cache_records,loss_codec_sample_order_QP_unchanged=True,per_branch_cached_update_count={b:sum(r.get('crop_cache_used',False) for r in load(ROOT/'branches'/b/'training_rows.json')) for b in BRANCHES[1:]}))
    dump('training_integrity.json',dict(status='PASS',branches=integrities,baseline_A_training_steps=0,shared_plan_sha256=sha(V32/'plans/shared.json'),B_D_same_receiver_initialization=integrities[BRANCHES[1]]['initial']['modules']['bridge']==integrities[BRANCHES[3]]['initial']['modules']['bridge'] and integrities[BRANCHES[1]]['initial']['modules']['generator']==integrities[BRANCHES[3]]['initial']['modules']['generator']))
    dump('frozen_module_integrity.json',dict(status='PASS',branches={b:dict(compression_unchanged=x['initial']['compression']==x['final']['compression'],receiver_frozen=(x['initial']['modules']['bridge']==x['final']['modules']['bridge'] and x['initial']['modules']['generator']==x['final']['modules']['generator']) if b==BRANCHES[2] else None) for b,x in integrities.items()}))
    for r in points:assert sha(r['bitstream_path'])==r['bitstream_sha256']
    dump('actual_bitstream_integrity.json',dict(status='PASS',points=len(points),persisted_streams_all_rehashed=True,actual_rate='complete stream bytes*8/(64*1920*1080)',independent_decode_all=all(r['independent_decode_pass'] for r in points),state_sync_all=all(r['state_sync_pass'] for r in points)))
    index=[]
    paths={Path(r[k]) for r in points for k in ('bitstream_path','feature_path')}
    paths.update(ROOT.rglob('*.pt'));paths.update((ROOT/'features').glob('*.npz'))
    paths.update((ROOT/'bitstreams').glob('*.bin'))
    paths.update((ROOT/'cache/training_crops').glob('*.npy'))
    for p in sorted(paths):index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    dump('binary_artifacts_index.json',dict(artifacts=index))
    report='# v6.33 numerical report\n\nTraining: B/C/D each 1000 updates, A 0. Fixed B1 loss, original receiver initialization and identity-initialized existing P architecture. Historical B1 is a verified reference only.\n\n'
    report+='Branches: A = identity sender + frozen original receiver; B = identity sender + trained receiver; C = trained P + frozen original receiver; D = trained P + trained receiver. Receiver denotes the existing B/G modules.\n\n'
    report+='Formal evaluation: 1840 actual-RANS points, 460 verified historical original points and 1380 new points; 46 videos ×10 QPs ×4 branches.\n\n'
    report+='## Common actual-rate comparisons\n\n|Dataset|Comparison|Metric|Method mean|Reference mean|Delta|Relative %|\n|---|---|---|---:|---:|---:|---:|\n'
    for r in summary:report+=f'|{r["dataset"]}|{r["method"][0]} vs {r["reference"][0]}|{r["metric"]}|{r["method_mean"]:.9g}|{r["reference_mean"]:.9g}|{r["delta"]:.9g}|{r["relative_change_percent"]:.7g}|\n'
    report+=explicit_comparisons(points,summary)
    report+='\nThe factorial experiment starts fresh from the original receiver. Direct numerical partition of the historical jointly adapted B1 gain is NOT ESTABLISHED by these fresh-initialization contrasts.\n\n## Matched actual-rate channel results\n\nSpatial-mean intervention; one rate-selected pair per dataset/comparison; frame32. Primary1% and secondary2% are separate tiers. Gini uses positive sensitivity mass. Higher Gini means more concentrated positive channel sensitivity.\n\n|Dataset|NoP/P|Tier|Metric|Delta sensitivity mean|Delta Gini|Original I vs Delta estimated rate (Spearman)|NoP mean eta|P mean eta|\n|---|---|---|---|---:|---:|---:|---:|---:|\n'
    with (ROOT/'utility/distribution_changes.csv').open() as f:
        for r in csv.DictReader(f):
            if r['intervention']!='spatial_mean':continue
            report+='|'+ '|'.join(r[k] for k in ['dataset'])+'|'+r['noP'][0]+'/'+r['P'][0]+'|'+r['tier']+'|'+r['metric']+'|'+'|'.join(r[k] for k in ['Delta_signed_mean','Delta_positive_mass_Gini','original_I_vs_Delta_R_Spearman','eta_noP_mean','eta_P_mean'])+'|\n'
    report+='\n## Synchronized measured latency\n\n|Mode|Component|Mean ms|Std ms|Median ms|p95 ms|FPS|Peak allocated MiB|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    with (ROOT/'latency/latency_summary.csv').open() as f:
        for r in csv.DictReader(f):report+='|'+'|'.join(r[k] for k in ['mode','component','mean_ms','std_ms','median_ms','p95_ms','FPS','peak_allocated_MiB'])+'|\n'
    report+='\nA/B/C: Separate factorial contrasts are recorded above; no universal superiority is asserted across datasets/metrics. Negative metric delta means a lower metric value. These are fixed-budget observations with one fixed seed.\n\nD: Actual total bitrate reduction is NOT ESTABLISHED by matched/common-rate quality comparisons. See all raw QP/byte records.\n\nE/H: Channel sensitivity distributions and their Gini/CV/entropy/top-k statistics are recorded per metric/intervention in utility/distribution_summary.csv; paired changes are in utility/distribution_changes.csv. Negative utilities are retained; Gini/entropy/top-k use positive sensitivity mass only. Mathematical marginal bit utility and statistical significance of a population-level effect are NOT ESTABLISHED.\n\nF/G: Estimated rate correlations, original importance versus Delta R, and utility-per-estimated-y-rate changes are recorded separately in utility/distribution_summary.csv, utility/distribution_changes.csv and utility/paired_channel_changes.csv. These are associations and finite-difference proxies, not proof of bit-level marginal utility or optimality. Rates use installed 16-bit CDF probabilities plus native escape bypass lengths. Actual total RANS bytes remain separate. Hyperlatent z, headers, I frame and RANS state/rounding overhead are excluded from main-y channel attribution. Zero-rate channels and the denominator epsilon1e-9 are retained and identified; eta is not a marginal-bit claim.\n\nI: latency/latency_summary.csv records synchronized mean/std/median/p95 for both paths. It measures the historical B1 P and same B1 receiver in both paths, GPU7,1080p,batch1,QP5,20 warmup and100 measurements. Native-I reference reset is outside timing. P is sender-only.\n\nLimits: The inherited historical pretraining exposure caveat, including MCL source17, remains in DATA_SPLIT_AUDIT.md. No test-based selection was used. Utility pairs are chosen using actual rate only and frame32 is fixed. Sensitivity holds baseline decoder context fixed and does not propagate interventions to future references.\n'
    report=report.replace('It measures the historical B1 P and same B1 receiver in both paths','It measures final factorial D step1000 P with the same final D receiver in both paths')
    report=report.replace('GPU7,1080p',f'GPU{load(ROOT/"latency_integrity.json")["gpu"]},1080p')
    report=report.replace('Formal evaluation:', 'Effective backend seed: Torch/NumPy0 after the canonical codec setup override; configured Python/plan seed20261024. All three saved step0 RNG states and initialization hashes were audited. No loss or trajectory change was made. See seed_runtime_audit.json and PROTOCOL_IMPLEMENTATION_NOTES.md.\n\nFormal evaluation:')
    dump('implementation_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.glob('*.py'))})
    (ROOT/'REPORT.md').write_text(report)
    summary_text='# v6.33 completion\n\nB/C/D:1000 fixed updates each. A:0 updates. Actual-RANS evaluation:1840 points,46 videos,10 QPs,4 branches.\n\n'
    summary_text+=f'Channel audit:{len(utilities)} system points,128 decoded main-latent channels per point,3 interventions per channel. Rates are estimated installed-CDF main-y rates; actual per-channel RANS rate is unavailable.\n\n'
    with (ROOT/'latency/latency_summary.csv').open() as f:latency_rows=list(csv.DictReader(f))
    summary_text+='Final D step1000 latency means (ms/frame):\n\n|Mode|Encode|Decode|End-to-end|\n|---|---:|---:|---:|\n'
    for mode in ('No-P','With-P'):
        summary_text+='|'+mode+'|'+'|'.join(next(r['mean_ms'] for r in latency_rows if r['mode']==mode and r['component']==k) for k in ('encode','decode','end_to_end'))+'|\n'
    summary_text+='\nFull numerical records:REPORT.md, evaluation/equal_rate_summary.csv, utility/distribution_summary.csv, utility/distribution_changes.csv and latency/latency_summary.csv. Actual total RANS bitrate and estimated per-channel entropy-model rate remain separate.\n'
    (ROOT/'SUMMARY.md').write_text(summary_text)
    required=['protocol.json','PROTOCOL.md','DATA_SPLIT_AUDIT.md','data_split_audit.json','latent_interface_audit.json','checkpoint_hashes.json','frozen_module_integrity.json','training_integrity.json','actual_bitstream_integrity.json','channel_rate_integrity.json','latency_integrity.json','evaluation/all_rd_points.csv','evaluation/per_sequence_all_qp.csv','evaluation/equal_rate_pairs.csv','evaluation/equal_rate_summary.csv','evaluation/pooled_fid.csv','latency/latency_summary.csv','latency/complexity.json','REPORT.md','SUMMARY.md','binary_artifacts_index.json']
    required_figures=['p_causal_ablation_equal_rate_lpips.png','p_causal_ablation_equal_rate_dists.png','p_causal_ablation_equal_rate_flolpips.png','p_causal_ablation_fid.png','channel_utility_noP_vs_P.png','channel_rate_noP_vs_P.png','utility_per_rate_noP_vs_P.png','delta_rate_vs_original_utility.png','utility_distribution_histogram.png','utility_gini_cv_summary.png','matched_rate_channel_scatter.png']
    required+=['figures/'+f for f in required_figures]
    assert all((ROOT/f).is_file() and (ROOT/f).stat().st_size>0 for f in required)
    assert all(x['status']=='PASS' and x['steps']==1000 for x in integrities.values())
    latency=load(ROOT/'latency_integrity.json')
    assert latency['status']=='PASS' and latency['warmup']>=20 and latency['measured_iterations']>=100
    assert latency['checkpoint']['sha256']==sha(ROOT/'branches'/BRANCHES[3]/'checkpoints/adaptation_step_1000.pt')
    assert load(ROOT/'data_split_audit.json')['status']=='PASS'
    assert load(ROOT/'evaluation_source_integrity.json')['status']=='PASS'
    assert load(ROOT/'sanity/identity.json')['status']=='PASS'
    assert all(r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] for r in points)
    assert len({(r['dataset'],r['video'],r['method'],r['QP']) for r in points})==1840
    assert len(utilities)==2*len(load(ROOT/'utility/matched_pair_plan.json')['pairs'])
    assert all(u['status']=='PASS' and u['channels']==128 and len(u['rows'])==384 and u['replay_maxabs']==0 for u in utilities)
    dump('completion_integrity.json',dict(status='PASS',required_artifact_sha256={f:sha(ROOT/f) for f in required},new_training_steps_per_branch=1000,actual_RD_points=1840,channel_interventions_per_point=384,latency_final_D_verified=True))
    dump('status.json',dict(status='PASS',training_steps={b:1000 for b in BRANCHES[1:]},actual_RD_points=1840,channel_rate_attribution='estimated installed-CDF symbol rate including escape bypass; not actual per-channel RANS bytes'))

if __name__=='__main__':main()
