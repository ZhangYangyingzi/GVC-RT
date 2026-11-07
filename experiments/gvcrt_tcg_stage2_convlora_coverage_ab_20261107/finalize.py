"""Full measured tables, matched-budget records and dataset-separated three-line curves."""
from common_s2 import *
from evaluate import videos,METRICS,read_csv
import collections

QUALITY=['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']
LABELS={'baseline':'B+1500 matched-FP32 baseline (pilot receiver + TCG)',
    'A_control':'A: stage3 + head','B_expanded':'B: upsample.conv1 + stage3 + head'}

def group(v):
    if v['domain']=='ulong_new' and v['split']=='new_holdout_test':return 'ULong_new_holdout_32'
    if v['domain']=='ulong' and v['split']=='historical_reference':return 'ULong_historical_6'
    if v['domain']=='hevc_b' and v['split']=='historical_reference':return 'HEVC_B_5'
    assert v['domain']=='uvg_holdout' and v['sequence'] in ['HoneyBee','Jockey']
    return 'UVG_historical_HoneyBee_Jockey_2'

def plot(path,rows,title,steps):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,9));axes=axes.flatten()
    for ax,metric in zip(axes,QUALITY):
        for (name,label),style in zip(LABELS.items(),['--','-',':']):
            pts=sorted([r for r in rows if r['model']==name],key=lambda x:float(x['bpp']));assert len(pts)==3
            suffix='' if name=='baseline' else f' (dev-best step{steps[name]})'
            ax.plot([float(x['bpp']) for x in pts],[float(x[metric]) for x in pts],marker='o',linestyle=style,label=label+suffix)
        ax.set_xlabel('Actual bpp');ax.set_ylabel(metric+(' ↑' if metric in ['PSNR','MS_SSIM'] else ' ↓'));ax.grid(alpha=.25)
    axes[-1].axis('off');axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center',fontsize=8)
    fig.suptitle(title+'; measured QP0/5/9; existing test set reused');fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)

def main():
    env();verify_sources()
    controls={g:load(group_dir(g)/'stopping_state.json') for g in GROUPS};selections={g:load(group_dir(g)/'selection.json') for g in GROUPS}
    steps={g:selections[g]['best_trained']['step'] for g in GROUPS};vmap={v['sequence']:v for v in videos('final')};allrows=[]
    for model in ['baseline',*GROUPS]:
        step=None if model=='baseline' else steps[model];rows=read_csv(eval_dir('final',step,model)/'metrics.csv');assert len(rows)==135
        for r in rows:r.update(model=model,LoRA_step=0 if step is None else step,dataset=group(vmap[r['sequence']]),
            checkpoint_sha256='' if model=='baseline' else selections[model]['best_trained']['checkpoint_sha256'],
            TCG_initialization_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,existing_test_set_reused=True,
            baseline_quality_measurement_reused=model=='baseline')
        allrows+=rows
    assert len(allrows)==405 and len({(r['model'],r['sequence'],int(r['QP'])) for r in allrows})==405
    write_csv(ROOT/'metrics.csv',allrows)
    datasetrows=[];paired=[];bykey={(r['model'],r['sequence'],int(r['QP'])):r for r in allrows}
    for dataset,count in [('ULong_new_holdout_32',32),('ULong_historical_6',6),('HEVC_B_5',5),('UVG_historical_HoneyBee_Jockey_2',2)]:
        rows=[]
        for model in ['baseline',*GROUPS]:
            for qp in [0,5,9]:
                sample=[r for r in allrows if r['model']==model and r['dataset']==dataset and int(r['QP'])==qp];assert len(sample)==count
                rows.append(dict(dataset=dataset,model=model,QP=qp,video_count=count,LoRA_step=0 if model=='baseline' else steps[model],
                    weighting='equal videos within this dataset/split/QP',**{m:sum(float(r[m]) for r in sample)/count for m in METRICS}))
        datasetrows+=rows;plot(ROOT/f'plots/datasets/{dataset}.png',rows,dataset,steps)
        members=[v for v in videos('final') if group(v)==dataset]
        for qp in [0,5,9]:
            for candidate,reference in [('A_control','baseline'),('B_expanded','baseline'),('B_expanded','A_control')]:
                for metric in METRICS:
                    lower=metric not in ['PSNR','MS_SSIM'];a=[float(bykey[(candidate,v['sequence'],qp)][metric]) for v in members];b=[float(bykey[(reference,v['sequence'],qp)][metric]) for v in members]
                    d=[x-y for x,y in zip(a,b)];am=sum(a)/count;bm=sum(b)/count
                    paired.append(dict(dataset=dataset,QP=qp,candidate=candidate,reference=reference,metric=metric,video_count=count,
                        candidate_mean=am,reference_mean=bm,absolute_mean_delta=am-bm,relative_mean_delta=None if bm==0 else (am-bm)/bm,
                        mean_pointwise_relative_delta=sum(x/y for x,y in zip(d,b))/count if all(y!=0 for y in b) else None,
                        lower_is_better=lower,lower_or_higher_favorable_count=sum(x<0 if lower else x>0 for x in d),
                        unfavorable_count=sum(x>0 if lower else x<0 for x in d),equal_count=sum(x==0 for x in d),
                        candidate_best_step=0 if candidate=='baseline' else steps[candidate],reference_best_step=0 if reference=='baseline' else steps[reference],
                        comparison_scope='separate dev-best models; budgets may differ; not a matched-budget attribution'))
    write_csv(ROOT/'rd_points_by_dataset.csv',datasetrows);write_csv(ROOT/'paired_deltas_by_dataset_qp.csv',paired)
    for name,v in vmap.items():plot(ROOT/f'plots/per_video/{v["domain"]}_{name}.png',[r for r in allrows if r['sequence']==name],v['domain']+': '+name,steps)
    development={};coverage={};training_by_group={}
    for g in GROUPS:
        home=group_dir(g);training=[];files=sorted((home/'receipts').glob('update_*.json.gz'));n=len(files)
        for p in files:
            receipt=load(p)
            for s in receipt['logical_samples']:
                training.append(dict(step=receipt['step'],logical_rank=s['logical_rank'],physical_rank=s['physical_rank'],physical_GPU=s['physical_GPU'],
                    sample_key=s['sample_key'],video=s['video'],domain=s['domain'],external_QP=s['external_qp'],source_frame_start=s['source_frame_indices'][0],crop=json.dumps(s['crop']),
                    sample_gradient_norm=s['sample_gradient_norm'],**s['losses']))
        write_csv(home/'training_history.csv',training);training_by_group[g]=training
        plan=plan_rows()[:4*n];domains={}
        for domain in ['uvg','ulong']:
            sample=[r for r in plan if r['domain']==domain];ids={r['video'] for r in sample}
            windows={(r['video'],tuple(r['source_frame_indices'])) for r in sample};exact={(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y'],r['external_qp']) for r in sample}
            domains[domain]=dict(samples=len(sample),identities=len(ids),windows=len(windows),exact_combinations=len(exact),exact_repeats=len(sample)-len(exact),
                QP_counts=dict(collections.Counter(str(r['external_qp']) for r in sample)),ULong_pool_identity_coverage=len(ids)/8470 if domain=='ulong' else None)
        launches=[load(p) for p in (home/'audits').glob(f'train_{g}_to_*.json') if not p.name.endswith('_0000.json')]
        wall=sum(r['elapsed_wall_seconds'] for r in launches)
        assigned_GPU_wall=sum(r['elapsed_wall_seconds']*len(r['GPUs'].split(',')) for r in launches)
        coverage[g]=dict(committed_optimizer_updates=n,rank_samples=4*n,source_cursor_start=1500,next_cursor=1500+n,domains=domains,
            actual_training_launch_wall_seconds=wall,assigned_GPU_wall_seconds=assigned_GPU_wall,shared_operational_timing=True)
        dump(home/'training_coverage.json',coverage[g])
        data=[]
        for c in controls[g]['checks']:
            rows=read_csv(eval_dir('development',c['step'],g)/'metrics.csv')
            for r in rows:r.update(model=g,LoRA_step=c['step'])
            data+=rows
        development[g]=data;write_csv(home/'development_metrics.csv',data)
        for start in range(0,len(files),500):dump(home/f'receipts_archive/updates_{start+1:05d}_{min(start+500,len(files)):05d}.json.gz',[load(p) for p in files[start:start+500]])
    dump(ROOT/'training_coverage.json',coverage)
    common=sorted(set(c['step'] for c in controls['A_control']['checks'])&set(c['step'] for c in controls['B_expanded']['checks']))
    matched=[]
    for step in common:
        for qp in [0,5,9]:
            a=[r for r in development['A_control'] if r['LoRA_step']==step and int(r['QP'])==qp]
            b=[r for r in development['B_expanded'] if r['LoRA_step']==step and int(r['QP'])==qp];assert len(a)==len(b)==32
            for metric in METRICS:
                am=sum(float(r[metric]) for r in a)/32;bm=sum(float(r[metric]) for r in b)/32
                matched.append(dict(step=step,dataset='ULong_development_32',QP=qp,metric=metric,A_mean=am,B_mean=bm,B_minus_A=bm-am,
                    videos=32,optimizer_updates_each=step,logical_samples_each=4*step,ULong_samples_each=2*step,source_cursor_next=1500+step))
        ca=next(c for c in controls['A_control']['checks'] if c['step']==step);cb=next(c for c in controls['B_expanded']['checks'] if c['step']==step)
        matched.append(dict(step=step,dataset='ULong_development_32',QP='all',metric='normalized_dev_FloLPIPS',A_mean=ca['score'],B_mean=cb['score'],B_minus_A=cb['score']-ca['score'],
            videos=32,optimizer_updates_each=step,logical_samples_each=4*step,ULong_samples_each=2*step,source_cursor_next=1500+step))
    write_csv(ROOT/'matched_step_comparison.csv',matched)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for g in GROUPS:ax.plot([0]+[c['step'] for c in controls[g]['checks']],[1.]+[c['score'] for c in controls[g]['checks']],marker='o',label=LABELS[g])
    ax.axhline(1,color='gray',linestyle='--',label=LABELS['baseline']);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Normalized development FloLPIPS ↓');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/development_curves.png',dpi=150);plt.close(fig)
    baseline_dev=read_csv(eval_dir('development',group='baseline')/'metrics.csv');fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,qp in zip(axes,[0,5,9]):
        base=sum(float(r['DISTS']) for r in baseline_dev if int(r['QP'])==qp)/32
        for g in GROUPS:
            nodes=[c['step'] for c in controls[g]['checks']]
            vals=[sum(float(r['DISTS']) for r in development[g] if r['LoRA_step']==step and int(r['QP'])==qp)/32-base for step in nodes]
            ax.plot(nodes,vals,marker='o',label=LABELS[g])
        ax.axhline(0,color='gray',linestyle='--');ax.set_title(f'Development QP{qp},32 equal videos');ax.set_xlabel('Optimizer updates');ax.set_ylabel('DISTS - baseline DISTS (lower ↓)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(ROOT/'plots/DISTS_changes.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,g in zip(axes,GROUPS):
        grouped=collections.defaultdict(list)
        for r in training_by_group[g]:grouped[r['step']].append(r)
        ss=sorted(grouped)
        for metric in ['total_loss','LPIPS','weighted_DISTS','weighted_motion_loss','weighted_reconstruction_L1']:
            ax.plot(ss,[sum(float(r[metric]) for r in grouped[s])/4 for s in ss],label=metric,linewidth=.8)
        ax.set_title(g);ax.set_xlabel('Optimizer updates');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'plots/training_losses.png',dpi=150);plt.close(fig)
    from integrity import final_check
    audit=final_check(persist=False)
    report=['# Generator Conv-LoRA coverage A/B execution report','UTC: '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        f'Integrity:{audit["status"]}. Runtime:{audit["selected_runtime"]}.',
        'One A/B architecture contrast. A: four stage3 depthwise-parallel branches plus head,59392 parameters. B adds only upsample.conv1(512→2048),75776 additional parameters,total135168. LoRA rank4,alpha/r1,no added biases,up weights zero at initialization.',
        'Order:stage1→stage2→tcg_low→ada3→upsample→tcg_mid→stage3→ada4→stage4→tcg_out→ada_final→head. The added branch is before depth_to_space,after tcg_low/ada3 and before tcg_mid. Actual256crop feature shapes are in initialization_and_dispatch.json:low512×16×16,mid512×32×32,out320×32×32. No claim of three distinct spatial resolutions.',
        'Both start from strictly loaded original I/P + pilot Bridge/generator/buffers + exact FP32 B1500 TCG overlay. Composite baseline is not original no-TCG GVC-RT. Frozen TCG cumulative step6500 stays fixed. No Router or old trained LoRA checkpoint loaded.',
        'Seed20261106. Shared five branch tensors and four logical RNG streams match at formal step0. B extra initialization uses an independent restored RNG fork. Original parameters,TCG and persistent buffers stay frozen; temporal caches still update,detach and reset as specified.',
        'Fresh AdamW lr5e-5,betas(.9,.999),eps1e-8,weight_decay1e-4. Same four planned samples per update(2UVG+2ULong),source cursor1500,64frames256crop,I0/P1–48 warmup/P49–63 mean15. VJPs summed across four logical samples,divide4 once,clip1 once,one optimizer update.',
        'Historical loss and4096/unscale ordering unchanged. Rate has zero LoRA gradient and is monitoring/constant; Identity proxy zero. No added loss,normalization,gate,alignment or TCG joint tuning.',
        'Two-rank accumulation was compared with four-rank reference from identical pre-step checkpoints; original unanchored engineering comparison and correction remain in audits. Tolerance1e-3 was not relaxed, and no bitwise equivalence between reduction orders is claimed. All engineering updates were discarded.',
        'Separate early stopping:validation every250,cap3000,min_delta0.0005,patience4. Raw best and threshold status are separate. Both development selections were frozen before any final quality evaluation in this run.',
        'Public baseline quality is reused from the reference only after source/protocol/weight/stream/frame/transition audits and a native RANS reproduction check. A/B independently decode and score their own outputs. Same pre-encoded stream bytes,bpp,SHA and codec traces are checked pointwise.',
        'The45 existing test videos were already evaluated in prior experiments. This is a reused test set,not a new unseen confirmation set. Its legacy subgroup names are retained. No test-based adjustment was made.',
        'Matched-update development comparisons are the main budget-controlled records in matched_step_comparison.csv. Separate best-trained final models can have different budgets; these differences are not attributed solely to added coverage.',
        'Each dataset/split/QP average weights videos equally. Four dataset groups stay separate; UVG historical contains only HoneyBee/Jockey. Curves connect only measured QP0/5/9; no extrapolation or BD-rate.',
        'GT-flow and reconstruction-flow PWC errors use the unchanged inherited sampler and are separately named. No scientific-effectiveness,convergence or next-research-direction conclusion is drawn.',
        '## Group execution and selection','```json',json.dumps(audit['groups'],indent=2),'```',
        '## Common complete validation steps','```json',json.dumps(common),'```',
        '## Final metrics by dataset and QP',
        '|Dataset|Model|Best step|QP|Videos|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|GT warp|Recon warp|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in datasetrows:report.append('|'+ '|'.join([r['dataset'],LABELS[r['model']],str(r['LoRA_step']),str(r['QP']),str(r['video_count'])]+[f'{float(r[k]):.10g}' for k in ['bpp','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']])+'|')
    report+=['\n## Validation/patience records','```json',json.dumps(controls,indent=2),'```',
        '\n## Actual data coverage and training wall time','```json',json.dumps(coverage,indent=2),'```',
        '\n## Decode timing','Warmup1 full sequence,3 repeated synchronized decodes per model on physicalGPU4,preselected development video andQP5. I/O,model load and metric networks excluded. Inherited TCG diagnostics remain included. Shared GPU snapshots are recorded; no exclusive realtime-performance claim.',
        '```json',json.dumps(load(ROOT/'decoding_timing.json'),indent=2),'```',
        '\n## Products','metrics.csv; paired_deltas_by_dataset_qp.csv; matched_step_comparison.csv; stream_identity_checks.csv; checkpoints_manifest.json; A_control/ and B_expanded/ histories,selections and checkpoint receipts; decode_timing.csv; plots/datasets/,development_curves.png,DISTS_changes.png,training_losses.png. Checkpoints,media and streams remain local. Publication status is separate.']
    import re
    (ROOT/'REPORT.md').write_text(re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(report)+'\n'))
    excluded={'artifact_manifest.json','status.json','publication_status.json','publication_allowlist.json','publication_remote_receipt.json','active_processes.json','runner_identity.json','launcher_receipt.json','final_integrity.json'}
    files=[dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
        and p.suffix in ['.json','.csv','.md','.png','.gz'] and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','checkpoints','logs','runtime_tree','reference_tree','vendor','receipts'])
        and not (p.parent==ROOT and p.name in excluded)]
    raw=(json.dumps(audit,indent=2,allow_nan=False)+'\n').encode();files.append(dict(path='final_integrity.json',size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    dump(ROOT/'artifact_manifest.json',dict(scope='Immutable task products; source has separate manifest',files=files))
    dump(ROOT/'final_integrity.json',audit);print('AB_FINAL_METRICS_REPORT_PLOTS_INTEGRITY_PASS')

if __name__=='__main__':main()
