"""Four measured deployment/precision/joint curves, objective tables and final assertions."""
from common_s2 import *
from reuse import read_csv
from evaluate import videos,METRICS
from precision_comparison import update as precision_update
import collections
QUALITY=('PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC')
LABELS={'GVC_RT_original':'GVC-RT original (independent)', 'B1500_historical':'B+1500 historical deployment (FP16 TCG)',
        'B1500_matched_FP32':'B+1500 matched FP32-TCG (no training)', 'Joint_best':'B+1500 + internal history gate (Router/TCG joint)'}

def group(v):
    if v['domain']=='ulong_new' and v['split']=='new_holdout_test':return 'ULong_new_holdout_32'
    if v['domain']=='ulong' and v['split']=='historical_reference':return 'ULong_historical_6'
    if v['domain']=='hevc_b' and v['split']=='historical_reference':return 'HEVC_B_5'
    assert v['domain']=='uvg_holdout' and v['split']=='historical_reference' and v['sequence'] in ['HoneyBee','Jockey']
    return 'UVG_historical_HoneyBee_Jockey_2'

def plot(path,rows,title):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(19,9));axes=axes.flatten()
    styles=['-','--',':','-.']
    for ax,metric in zip(axes,QUALITY):
        for (model,label),style in zip(LABELS.items(),styles):
            a=sorted([r for r in rows if r['model_state']==model],key=lambda r:float(r['bpp']));assert len(a)==3
            ax.plot([float(r['bpp']) for r in a],[float(r[metric]) for r in a],marker='o',linestyle=style,label=label)
        ax.set_xlabel('Measured bpp');ax.set_ylabel(metric+(' ↑' if metric in ['PSNR','MS_SSIM'] else ' ↓'));ax.grid(alpha=.25)
    axes[-1].axis('off');axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center',fontsize=8)
    fig.suptitle(title+'; QP0/5/9; measured points only');fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)

def main():
    env();verify_sources();selection=load(ROOT/'selection.json');control=load(ROOT/'stopping_state.json');best=selection['best_trained_joint']['step']
    precision=precision_update();assert precision['status']=='COMPLETE'
    original=read_csv(ROOT/'evaluation/reused/original_final.csv');historical=read_csv(ROOT/'evaluation/reused/B_historical_final.csv')
    matched=read_csv(ROOT/'evaluation'/evaluation_id('final',candidate='matched_baseline')/'metrics.csv')
    joint=read_csv(ROOT/'evaluation'/evaluation_id('final',best)/'metrics.csv')
    for r in matched:r['model_state']='B1500_matched_FP32'
    for r in joint:r['model_state']='Joint_best';r['committed_joint_step']=best
    allrows=original+historical+matched+joint;vmap={v['sequence']:v for v in videos('final')}
    assert len(allrows)==540 and len({(r['model_state'],r['sequence'],int(r['QP'])) for r in allrows})==540
    for r in allrows:r['dataset_plot_group']=group(vmap[r['sequence']])
    write_csv(ROOT/'metrics_joint.csv',joint);write_csv(ROOT/'metrics_comparison.csv',allrows)
    bykey={(r['model_state'],r['sequence'],int(r['QP'])):r for r in allrows};streams=[];deltas=[]
    for v in videos('final'):
        for q in [0,5,9]:
            m=bykey[('B1500_matched_FP32',v['sequence'],q)];j=bykey[('Joint_best',v['sequence'],q)];h=bykey[('B1500_historical',v['sequence'],q)]
            streams.append(dict(sequence=v['sequence'],QP=q,matched_bytes=m['real_bytes'],joint_bytes=j['real_bytes'],historical_bytes=h['real_bytes'],
                matched_sha256=m['bitstream_sha256'],joint_sha256=j['bitstream_sha256'],historical_sha256=h['bitstream_sha256'],
                same_bytes=int(m['real_bytes'])==int(j['real_bytes'])==int(h['real_bytes']),same_stream_sha256=m['bitstream_sha256']==j['bitstream_sha256']==h['bitstream_sha256']))
            for comparator in ['B1500_matched_FP32','B1500_historical','GVC_RT_original']:
                other=bykey[(comparator,v['sequence'],q)]
                for metric in METRICS:
                    d=float(j[metric])-float(other[metric]);deltas.append(dict(sequence=v['sequence'],QP=q,dataset=group(v),candidate='Joint_best',reference=comparator,
                        metric=metric,absolute_delta=d,relative_delta=None if float(other[metric])==0 else d/float(other[metric])))
    write_csv(ROOT/'stream_identity_checks.csv',streams);write_csv(ROOT/'paired_deltas.csv',deltas)
    datasetrows=[]
    for name,expected in [('ULong_new_holdout_32',32),('ULong_historical_6',6),('HEVC_B_5',5),('UVG_historical_HoneyBee_Jockey_2',2)]:
        rows=[]
        for model in LABELS:
            for q in [0,5,9]:
                a=[r for r in allrows if r['model_state']==model and r['dataset_plot_group']==name and int(r['QP'])==q];assert len(a)==expected
                rows.append(dict(dataset=name,model_state=model,QP=q,video_count=expected,weighting='equal video within this split/domain; no cross-dataset average',
                    **{k:sum(float(r[k]) for r in a)/expected for k in METRICS}))
        datasetrows+=rows;plot(ROOT/f'plots/datasets/{name}.png',rows,name)
    write_csv(ROOT/'rd_points_by_dataset.csv',datasetrows)
    for name,v in vmap.items():plot(ROOT/f'plots/per_video/{v["domain"]}_{name}.png',[r for r in allrows if r['sequence']==name],f'{v["domain"]}: {name}')
    training=[];masks=[];coverage={};plan=plan_rows();n=len(list((ROOT/'receipts').glob('update_*.json.gz')))
    for p in sorted((ROOT/'receipts').glob('update_*.json.gz')):
        receipt=load(p)
        for r in receipt['ranks']:
            row=dict(step=receipt['step'],rank=r['rank'],sample_key=r['sample_key'],domain=r['domain'],video=r['video'],external_qp=r['external_qp'],
                window_start=r['source_frame_indices'][0],crop=json.dumps(r['crop']),seconds=r['seconds_before_commit'],
                gradient_norm_local=r['local_gradient_norm'],gradient_norm_mean=r['mean_gradient_norm'],joint_change_L2=r['parameter_change_L2'],**r['losses'])
            for key in ['Router','TCG']:
                row.update({key+'_gradient_norm_local':r['group_gradient_norms_local'][key],key+'_gradient_norm_mean':r['group_gradient_norms_mean'][key],
                    key+'_parameter_change_L2':r['group_parameter_changes_L2'][key],key+'_relative_change_from_B_or_init':r['group_relative_parameter_changes_from_B_or_new_Router_initialization'][key]})
            training.append(row);masks.append(dict(kind='training_last_frame',step=receipt['step'],rank=r['rank'],**r['mask_statistics']))
    write_csv(ROOT/'training_history.csv',training)
    for domain in ['uvg','ulong']:
        rows=[r for r in plan[:4*n] if r['domain']==domain];identities={r['video'] for r in rows}
        windows={(r['video'],tuple(r['source_frame_indices'])) for r in rows};crops={(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y']) for r in rows}
        exact={(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y'],r['external_qp']) for r in rows}
        coverage[domain]=dict(samples=len(rows),distinct_identities=len(identities),distinct_windows=len(windows),distinct_crops=len(crops),exact_combinations=len(exact),
            repeated_exact_samples=len(rows)-len(exact),identity_visits=dict(collections.Counter(r['video'] for r in rows)),QP_counts=dict(collections.Counter(str(r['external_qp']) for r in rows)),
            pool=8470 if domain=='ulong' else None,pool_identity_coverage=len(identities)/8470 if domain=='ulong' else None,first_project_exposure_claim=False)
    dump(ROOT/'training_coverage.json',dict(committed_joint_updates=n,rank_samples=4*n,source_plan_cursor=1500+n,domains=coverage))
    development=[]
    for candidate,step in [('matched_baseline',None)]+[('joint',s) for s in [0]+[c['step'] for c in control['checks']]]:
        rows=read_csv(ROOT/'evaluation'/evaluation_id('development',step,candidate)/'metrics.csv')
        for r in rows:r['candidate']=candidate;r['committed_joint_step']=0 if step is None else step
        development+=rows
    write_csv(ROOT/'development_metrics.csv',development)
    for candidate,step in [('matched_baseline',None),('joint',best)]:
        for p in sorted((ROOT/'evaluation'/evaluation_id('final',step,candidate)/'masks').glob('*.csv')):
            masks += [dict(kind='final_'+candidate,point=p.stem,step=0 if step is None else step,**r) for r in read_csv(p)]
    write_csv(ROOT/'mask_statistics.csv',masks)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    means=collections.defaultdict(list)
    for r in training:means[int(r['step'])].append(r)
    steps=sorted(means);fig,ax=plt.subplots(figsize=(10,5))
    for metric in ['total_loss','LPIPS','weighted_DISTS','weighted_motion_loss','weighted_reconstruction_L1']:
        ax.plot(steps,[sum(x[metric] for x in means[s])/4 for s in steps],label=metric,linewidth=.8)
    ax.set_xlabel('Committed joint update');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/training_loss.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.plot([0]+[c['step'] for c in control['checks']],[control['enabled0']['score']]+[c['score'] for c in control['checks']],marker='o',label='Joint development FloLPIPS score')
    ax.axhline(1,color='gray',linestyle='--',label='Matched-FP32 B1500 baseline');ax.set_xlabel('Committed joint update');ax.set_ylabel('Normalized development FloLPIPS ↓');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/development_trajectory.png',dpi=150);plt.close(fig)
    from integrity import final_check
    audit=final_check()
    report=['# Internal history gate Router/TCG joint execution report',f'UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',
        f'Integrity:{audit["status"]}. Committed joint updates:{n}. Stop reason:{control["status"]}.',
        f'Best trained joint checkpoint:{best}; development score:{selection["best_trained_joint"]["score"]:.12g}; final selection:{selection["final_selection"]}.',
        f'Raw score below matched baseline:{selection["raw_better_than_baseline"]}; preset0.001 threshold met:{selection["meaningful_improvement_threshold_met"]}. These flags and selection are records, not scientific effectiveness claims.',
        'Base B+1500 cumulative TCG6500, exact pretrained TCG retained; new Router seed20261106 and M0=.2. Only Router33889+TCG137760 parameters update. Fresh AdamW learning rates1e-4/1e-5; no restored historical optimizer, no added loss or training group.',
        'Gate only previous_adapter output before concat; current_adapter retained. Router/TCG tensors and operations FP32; output casts to incoming dtype. Final rounding remains possible. M1 is not the TCG-off mode.',
        'Historical FP16 deployment B and matched-FP32 B inference remain distinct. Matched B is the sole normalization baseline. Per-point precision differences are in precision_baseline_comparison.json and precision_baseline_deltas.csv.',
        'New joint135points used fresh real RANS and independent decoding. Matched B used audited old streams with new independent FP32-TCG decoding. Original135/historical B135 metrics reused only after point/stream/source/frame audits.',
        'Four groups are kept separate with equal video weights per split/domain/QP. UVG historical comprises only HoneyBee/Jockey. Three measured QP points, no extrapolation or BD-rate.',
        'GT-flow and reconstruction-flow PWC errors are separate. The inherited sampler convention is preserved; these are not claimed pure flicker measures or strict S2VC reproduction.',
        'Diagnostic videos and frames were fixed before new quality scoring; masks use a fixed0–1 scale. Preview PNGs are8-bit viewing exports; metrics use original unquantized decoded tensors.',
        '|Dataset|Model|QP|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|WarpGT|WarpRecon|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in datasetrows:report.append('|'+ '|'.join([r['dataset'],LABELS[r['model_state']],str(r['QP'])]+[f'{float(r[k]):.9g}' for k in ['bpp',*QUALITY]])+'|')
    report+=['\nDevelopment and stopping records:\n','```json',json.dumps(control,indent=2),'```','\nCoverage:\n','```json',json.dumps({k:{n:v for n,v in d.items() if n!='identity_visits'} for k,d in coverage.items()},indent=2),'```',
        '\nEngineering and limitations: implementation/gradient/cache/FP32/RANS evidence and any retries are in audits/ and local logs. No new-data independence, motion segmentation, occlusion correctness or mathematical convergence claim is made. Operational shared-GPU timings are not dedicated inference latency benchmarks.']
    (ROOT/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    mutable={'artifact_manifest.json','run_status.json','active_processes.json','runner_identity.json','launcher_receipt.json','publication_status.json','publication_allowlist.json','publication_remote_receipt.json'}
    dump(ROOT/'artifact_manifest.json',dict(scope='Immutable task results/receipts; mutable execution/publication snapshots excluded',excluded_root_snapshots=sorted(mutable),
        files=[dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.png','.gz']
        and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','checkpoints','runtime_tree','reference_tree','vendor']) and not (p.parent==ROOT and p.name in mutable)]))
if __name__=='__main__':main()
