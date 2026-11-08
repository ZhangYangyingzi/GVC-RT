"""Objective tables, coverage, measured plots and final assertions; no model selection here."""
from common_s2 import *
from evaluate import videos,read_csv,METRICS
from evaluation_support import key,validate_point
from integrity import final_check,training_integrity,checkpoint_manifest

QUALITY=['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']
LOWER=set(QUALITY)-{'PSNR','MS_SSIM'}

def dataset(v):
    if v['domain']=='mcljcv':return 'MCL_JCV_30'
    if v['domain']=='hevc_b':return 'HEVC_B_5'
    if v['domain'] in ['uvg','uvg_holdout']:return 'UVG_historical_HoneyBee_Jockey_2'
    return 'ULong_new_holdout_32' if v['split']=='new_holdout_test' or v['domain']=='ulong_new' else 'ULong_historical_6'

def training_tables():
    check,receipts=training_integrity();samples=[];history=[]
    for r in receipts:
        for x in r['logical_samples']:
            samples.append(x)
            history.append(dict(additional_step=r['step'],absolute_LoRA_optimizer_step=2000+r['step'],logical_rank=x['logical_rank'],physical_GPU=x['physical_GPU'],
                sample_key=x['sample_key'],source_group=x['source_group'],video=x['video'],domain=x['domain'],external_QP=x['external_qp'],
                first_frame=x['source_frame_indices'][0],crop_x=x['crop'][0],crop_y=x['crop'][1],sample_gradient_norm=x['sample_gradient_norm'],
                mean_reduced_gradient_norm=r['physical_ranks'][0]['mean_gradient_norm'],maximum_rank_seconds=max(a['seconds_before_commit'] for a in r['physical_ranks']),
                maximum_peak_allocated_bytes=max(a['peak_allocated_bytes'] for a in r['physical_ranks']),LoRA_update_L2=r['physical_ranks'][0]['parameter_change_L2'],**x['losses']))
    write_csv(ROOT/'training_history.csv',history)
    videos_seen={x['video'] for x in samples};groups={x['source_group'] for x in samples}
    windows={(x['video'],tuple(x['source_frame_indices'])) for x in samples};crops={(x['video'],tuple(x['source_frame_indices']),tuple(x['crop'])) for x in samples}
    exact={(x['video'],tuple(x['source_frame_indices']),tuple(x['crop']),x['external_qp']) for x in samples}
    counts={v:sum(x['video']==v for x in samples) for v in sorted(videos_seen)}
    byvideo={v:[x['source_frame_indices'][0] for x in samples if x['video']==v] for v in videos_seen};overlaps=[]
    for v,starts in byvideo.items():
        for i,a in enumerate(starts):
            for b in starts[i+1:]:
                if abs(a-b)<64:overlaps.append(dict(video=v,start_a=a,start_b=b,shared_frames=64-abs(a-b),fraction_of_64=(64-abs(a-b))/64))
    coverage=dict(status='PASS',available_training_videos=8470,available_source_groups=7969,actual_ULong_samples=len(samples),actual_UVG_samples=0,actual_Vimeo_samples=0,
        actual_unique_videos=len(videos_seen),actual_unique_source_groups=len(groups),unique_windows=len(windows),unique_crops=len(crops),unique_exact_combinations=len(exact),
        repeated_windows=len(samples)-len(windows),repeated_exact_combinations=len(samples)-len(exact),overlapping_window_pairs=len(overlaps),
        video_coverage_fraction=len(videos_seen)/8470,source_group_coverage_fraction=len(groups)/7969,logical_plan_samples=12000,completed_plan_fraction=len(samples)/12000,
        QP_counts={str(q):sum(x['external_qp']==q for x in samples) for q in range(10)},per_video_visits=counts,
        total_measured_four_rank_processing_GPU_seconds=sum(sum(p['seconds_before_commit'] for p in r['physical_ranks']) for r in receipts),
        timing_scope='training sample processing and reductions before atomic commit;startup,validation and I/O outside this interval not included',
        historical_UVG_exposed_start=True,overlapping_windows_are_not_independent_scenes=True)
    dump(ROOT/'training_coverage.json',coverage)
    if overlaps:write_csv(ROOT/'overlapping_windows.csv',overlaps)
    # Raw per-update receipts remain local. Bounded, deterministic compressed shards are published.
    for i in range(0,len(receipts),100):dump(ROOT/f'published_receipts/updates_{i+1:05d}_{min(i+100,len(receipts)):05d}.json.gz',receipts[i:i+100])
    for i in range(0,len(samples),200):
        batch=[load(ROOT/f'audits/window_reads/{x["sample_key"]}.json.gz') for x in samples[i:i+200]]
        dump(ROOT/f'published_read_audits/samples_{i:05d}.json.gz',batch)
    return check,coverage

def main():
    env();verify_sources();selection=load(ROOT/'selection.json');states=load(ROOT/'final_states.json')['states'];control=load(ROOT/'stopping_state.json')
    training,coverage=training_tables();dev=[]
    for item in control['checks']:
        for r in read_csv(eval_dir('development',item['step'],'A_control')/'metrics.csv'):
            dev.append(dict(additional_step=item['step'],S=item['S'],qualified=item['qualified'],bad_check_count=item['bad_check_count'],**r))
    write_csv(ROOT/'development_metrics.csv',dev);dump(ROOT/'validation_history.json',control)
    measurements=[];counts={};by={}
    for state in states:
        rows=[];new=reused=mcl=0
        for v in videos('final'):
            for qp in [0,5,9]:
                p=validate_point(ROOT/state['batch'],v,qp,state['candidate'],state.get('checkpoint'))[0]
                reused+=bool(p.get('measurement_reused'));new+=not bool(p.get('measurement_reused'));mcl+=v['domain']=='mcljcv'
                r=dict(model=state['name'],label=state['label'],dataset=dataset(v),measurement_reused=bool(p.get('measurement_reused')),**p['result'])
                measurements.append(r);rows.append(r)
        assert len(rows)==225 and mcl==90;counts[state['name']]=dict(completed=225,expected=225,new=new,reused=reused,MCL_completed=90)
        by[state['name']]=rows
    write_csv(ROOT/'metrics.csv',measurements)
    aggregates=[]
    expected={'ULong_new_holdout_32':32,'ULong_historical_6':6,'HEVC_B_5':5,'UVG_historical_HoneyBee_Jockey_2':2,'MCL_JCV_30':30}
    for state in states:
        for ds,n in expected.items():
            for qp in [0,5,9]:
                rows=[r for r in by[state['name']] if r['dataset']==ds and int(r['QP'])==qp];assert len(rows)==n
                aggregates.append(dict(model=state['name'],label=state['label'],dataset=ds,external_QP=qp,equal_video_count=n,
                    **{m:sum(float(r[m]) for r in rows)/n for m in METRICS}))
    write_csv(ROOT/'rd_points_by_dataset.csv',aggregates)
    deltas=[];video_deltas=[]
    for r in aggregates:
        for reference in ['Original_GVC_RT','A2000_s05']:
            b=next(x for x in aggregates if x['model']==reference and x['dataset']==r['dataset'] and x['external_QP']==r['external_QP'])
            for m in QUALITY:
                delta=r[m]-b[m]
                deltas.append(dict(dataset=r['dataset'],QP=r['external_QP'],candidate=r['model'],reference=reference,metric=m,video_count=r['equal_video_count'],
                    candidate_mean=r[m],reference_mean=b[m],absolute_delta=delta,relative_percent=(r[m]/b[m]-1)*100,lower_is_better=m in LOWER))
    for r in measurements:
        for reference in ['Original_GVC_RT','A2000_s05']:
            b=next(x for x in by[reference] if x['domain']==r['domain'] and x['sequence']==r['sequence'] and int(x['QP'])==int(r['QP']))
            for m in QUALITY:
                d=float(r[m])-float(b[m]);video_deltas.append(dict(dataset=r['dataset'],sequence=r['sequence'],QP=r['QP'],candidate=r['model'],reference=reference,
                    metric=m,absolute_delta=d,relative_percent=(float(r[m])/float(b[m])-1)*100,lower_is_better=m in LOWER))
    write_csv(ROOT/'paired_deltas_by_dataset_qp.csv',deltas);write_csv(ROOT/'paired_deltas_by_video.csv',video_deltas)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=[]
    for ds in expected:
        fig,axes=plt.subplots(2,4,figsize=(19,9));axes=list(axes.flat)
        for ax,m in zip(axes,QUALITY):
            for state in states:
                r=sorted([x for x in aggregates if x['dataset']==ds and x['model']==state['name']],key=lambda x:x['bpp'])
                ax.plot([x['bpp'] for x in r],[x[m] for x in r],marker='o',label=state['label'])
            ax.set(xlabel='Measured bpp',ylabel=m+(' ↓' if m in LOWER else ' ↑'));ax.grid(alpha=.25)
        axes[-1].axis('off');handles,labels=axes[0].get_legend_handles_labels();axes[-1].legend(handles,labels,loc='center',fontsize=9)
        fig.suptitle(ds+' | equal video weights, measured QP 0/5/9; no extrapolation');fig.tight_layout()
        path=ROOT/f'plots/RD_{ds}.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for m in ['total_loss','DISTS','LPIPS','motion_loss']:
        rows=read_csv(ROOT/'training_history.csv');steps=sorted({int(x['additional_step']) for x in rows})
        axes[0].plot(steps,[sum(float(x[m]) for x in rows if int(x['additional_step'])==s)/4 for s in steps],label=m)
    axes[0].legend();axes[0].set(xlabel='Committed additional updates',ylabel='Four-sample mean training loss component')
    axes[1].plot([x['step'] for x in control['checks']],[x['S'] for x in control['checks']],marker='o',label='Normalized development FloLPIPS S')
    axes[1].axhline(1,color='grey');axes[1].legend();axes[1].set(xlabel='Committed additional updates',ylabel='Development normalized score')
    fig.tight_layout();fig.savefig(ROOT/'plots/training_and_development.png',dpi=150);plt.close(fig)
    check=final_check();assert len(paths)==5 and all(Path(p).is_file() for p in paths)
    check.update(dataset_plot_paths=paths,model_point_counts=counts,frozen_components_unchanged=True,training_updates=training['additional_updates'])
    dump(ROOT/'final_integrity.json',check)
    report=['# ULong-only A2000 continuation: execution record','',
        'This record contains configuration, measurements, selection arithmetic and execution checks. No research conclusion is assigned.','',
        f"Parent A2000 SHA256: `{INIT_SHA}`. B+1500 TCG cumulative optimizer step: 6500, frozen. Initial LoRA step: 2000.",
        'The initial checkpoint has historical UVG exposure. This run uses ULong-only continuation; it is not lifelong ULong-only training.',
        'Train gain=1.0; development and final gain=0.5; intrinsic alpha/r=1. AdamW state restored, LR=5e-5 fixed. Only 59,392 LoRA parameters update.',
        'The existing test set and MCL-JCV are previously used evaluation sources. MCL uses real I/P RANS, including the first I stream; no GT-I bypass.',
        '',f"Stop reason: {control['status']}; additional updates={training['additional_updates']}; absolute LoRA step={training['absolute_LoRA_optimizer_step']}.",
        f"Selection: {selection['final_selection']}; additional step={selection['selected_additional_step']}. Selection was frozen before final quality evaluation.",
        f"Available ULong videos/groups=8470/7969; consumed unique videos/groups={coverage['actual_unique_videos']}/{coverage['actual_unique_source_groups']}; ULong samples={coverage['actual_ULong_samples']}; UVG=0; Vimeo=0.",
        '', '| Additional step | FloLPIPS S | D(QP0) | D(QP5) | D(QP9) | Qualified | Bad checks |','|---:|---:|---:|---:|---:|:---:|---:|']
    for c in control['checks']:report.append(f"|{c['step']}|{c['S']:.12g}|{c['D_q']['0']:.12g}|{c['D_q']['5']:.12g}|{c['D_q']['9']:.12g}|{c['qualified']}|{c['bad_check_count']}|")
    report+=['','Qualification is strict: all three development DISTS ratios <=1 and mean normalized FloLPIPS <1. Patience anchor min_delta=0.0005, patience=4.',
        'Spatial metrics use 64 frames; temporal metrics use 63 transitions. PSNR uses total pixel SSE. Video weights are equal within each dataset/QP.',
        'PWC backwarp sampling and metric networks are inherited unchanged. GT-flow and reconstruction-flow errors are separate fields.',
        '', '| Dataset | Model | QP | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FloLPIPS | GT-flow error | Recon-flow error |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:report.append('|'+ '|'.join([r['dataset'],r['label'],str(r['external_QP']),*[f'{r[m]:.12g}' for m in ['bpp','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']]])+'|')
    report+=['',f'Local integrity: PASS. Unique model states: {len(states)}, 225/225 points each, including 90 MCL points each.',
        'All compared video/QP stream bytes, SHA256, actual bpp, codec feature traces and GT-flow hashes match. Sources and original weights remain unchanged.',
        'All plots connect only measured QP0/5/9 points. No cross-dataset mean or BD-rate is calculated.',
        'Sparse content/source screening does not prove exhaustive detection of all shifted/cropped/re-encoded relationships. Window overlap is reported, not counted as independent scenes.',
        '', 'Full numeric records: metrics.csv, development_metrics.csv, rd_points_by_dataset.csv, paired_deltas_by_dataset_qp.csv, paired_deltas_by_video.csv, training_history.csv, training_coverage.json.',
        'Publication status is recorded separately in publication_status.json.']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    result=dict(status='COMPLETE',complete=True,experiment_directory=str(ROOT),initial_A2000_SHA256=INIT_SHA,s_train=1.,s_eval=.5,
        additional_updates=training['additional_updates'],absolute_LoRA_optimizer_step=training['absolute_LoRA_optimizer_step'],TCG_cumulative_step=6500,
        stop_reason=control['status'],selection=selection,training_coverage=coverage,model_test_counts=counts,MCL_points_per_unique_model=90,
        integrity_status='PASS',report_path=str(ROOT/'REPORT.md'),metrics_paths=[str(ROOT/p) for p in ['metrics.csv','development_metrics.csv','paired_deltas_by_dataset_qp.csv','stream_identity_checks.csv']],
        dataset_plot_paths=paths,github_branch=load(ROOT/'protocol.json')['branch'],publication_status='PENDING_RESULTS_PUBLICATION',time=time.time())
    dump(ROOT/'status.json',result)
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and any(str(p.relative_to(ROOT)).startswith(x) for x in ['plots/','published_receipts/','published_read_audits/'])}
    for name in ['REPORT.md','metrics.csv','development_metrics.csv','rd_points_by_dataset.csv','paired_deltas_by_dataset_qp.csv','paired_deltas_by_video.csv','training_history.csv','training_coverage.json','final_integrity.json']:
        files[name]=sha(ROOT/name)
    dump(ROOT/'result_hashes.json',dict(SHA256=files,time=time.time()))
    print('LOCAL_EXPERIMENT_COMPLETE',json.dumps(counts))

if __name__=='__main__':main()
