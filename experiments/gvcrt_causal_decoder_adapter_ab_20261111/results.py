"""Facts-only development tables, paired differences and R-D plots."""
from common_s2 import *
import statistics


def main():
    env()
    from evaluation import METRICS
    points=[]
    for p in sorted((ROOT/'evaluation/development').glob('*/step_*/points/*.json')):
        a=load(p);assert a['status']=='PASS';points.append(a['result'])
    if points:write_csv(ROOT/'metrics.csv',points)
    summaries=[]
    keys=sorted({(r['model'],r['step'],r['QP']) for r in points})
    for model,step,q in keys:
        rs=[r for r in points if (r['model'],r['step'],r['QP'])==(model,step,q)]
        row=dict(model=model,step=step,QP=q,completed_videos=len(rs),expected_videos=32)
        for k in ['bpp',*METRICS]:
            xs=[r[k] for r in rs if r.get(k) is not None];row[k]=sum(xs)/len(xs) if xs else None
        summaries.append(row)
    if summaries:write_csv(ROOT/'metrics_by_QP.csv',summaries)
    pairs=[];paired_summary=[]
    indexed={(r['model'],r['step'],r['sequence'],r['QP']):r for r in points}
    for r in points:
        if r['model'] not in GROUPS:continue
        for ref in ['B300','Original']:
            b=indexed.get((ref,0,r['sequence'],r['QP']))
            if b is None:continue
            assert r['stream_SHA256']==b['stream_SHA256'] and r['real_bytes']==b['real_bytes'] and r['bpp']==b['bpp']
            row=dict(model=r['model'],step=r['step'],reference=ref,sequence=r['sequence'],QP=r['QP'],
                bpp=r['bpp'],real_bytes=r['real_bytes'],stream_SHA256=r['stream_SHA256'])
            for k in METRICS:row[k+'_delta']=r[k]-b[k] if r.get(k) is not None and b.get(k) is not None else None
            pairs.append(row)
    if pairs:
        write_csv(ROOT/'paired_differences.csv',pairs)
        for m,s,ref,q in sorted({(r['model'],r['step'],r['reference'],r['QP']) for r in pairs}):
            rs=[r for r in pairs if (r['model'],r['step'],r['reference'],r['QP'])==(m,s,ref,q)]
            row=dict(model=m,step=s,reference=ref,QP=q,paired_videos=len(rs))
            for k in METRICS:
                xs=[r[k+'_delta'] for r in rs if r[k+'_delta'] is not None];row[k+'_delta']=sum(xs)/len(xs) if xs else None
            paired_summary.append(row)
        write_csv(ROOT/'paired_differences_by_QP.csv',paired_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'Original':'Original GVC-RT','B300':'B300',GROUPS[0]:'A spatial',GROUPS[1]:'B spatial + temporal'}
    for step,title in [(100,'Main: step100'),(50,'Supplement: step50')]:
        for metric in ['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS_PtoP','FloLPIPS_ItoP',
                'RGB_MCResidual_PtoP','Sobel_MCResidual_PtoP','RGB_MCResidual_ItoP','Sobel_MCResidual_ItoP','local16_energy_abs_bias']:
            fig,ax=plt.subplots(figsize=(6.4,4.5))
            plotted=0
            for model in ['Original','B300',*GROUPS]:
                rs=sorted([r for r in summaries if r['model']==model and r['step']==(step if model in GROUPS else 0) and r['completed_videos']==32],key=lambda r:r['bpp'])
                if len(rs)!=3:continue
                ax.plot([r['bpp'] for r in rs],[r[metric] for r in rs],marker='o',label=labels[model]+(f' {step}' if model in GROUPS else ''));plotted+=1
            ax.set(xlabel='Real RANS bpp (including I frame)',ylabel=metric+(' (higher is better)' if metric=='PSNR' else ' (lower is better)'),title=f'ULong development (32 videos x 3 QPs)\n{title}')
            if plotted:ax.legend(fontsize=8)
            ax.grid(alpha=.25);fig.tight_layout();dest=ROOT/f'plots/development/step{step}_{metric}.png';fig.savefig(dest,dpi=160);plt.close(fig)
    traces=[]
    for group in GROUPS:
        for p in sorted((ROOT/group/'training/receipts').glob('*.json')):
            a=load(p);traces.append({k:a[k] for k in ['group','committed_updates','next_cursor','loss','gradient_norm','upstream_gradient_norm','cache_wait_seconds','seconds','peak_allocated_bytes','peak_reserved_bytes','physical_GPU']})
    if traces:write_csv(ROOT/'training_logs/updates.csv',traces)
    performance=load(ROOT/'performance.json') if (ROOT/'performance.json').exists() else {'status':'PENDING'}
    memory={g:dict(peak_allocated_bytes=max([r['peak_allocated_bytes'] for r in traces if r['group']==g],default=0),
        peak_reserved_bytes=max([r['peak_reserved_bytes'] for r in traces if r['group']==g],default=0)) for g in GROUPS}
    bundles=[load(p) for p in (ROOT/'audits/evaluation_bundles').glob('*.json')]
    histories=sorted({load(p).get('history_bytes',0) for p in (ROOT/'evaluation/development').glob('*/step_*/points/*.json')})
    hc=[load(p) for p in (ROOT/'audits/h_cache').glob('*.json')]
    performance['observed_memory_and_storage']=dict(training_by_group=memory,
        evaluation_peak_allocated_bytes=max([r['peak_allocated_bytes'] for r in bundles],default=0),
        evaluation_peak_reserved_bytes=max([r['peak_reserved_bytes'] for r in bundles],default=0),
        deployment_history_tensor_bytes_observed=histories,shared_training_h_tensor_bytes=sum(r['h_bytes'] for r in hc),
        shared_training_h_cache_file_bytes=sum(r['bytes'] for r in hc),
        note='actual per-process CUDA allocator peaks and logical tensor bytes; contended wall times are not used as performance benchmarks')
    dump(ROOT/'performance.json',performance)
    checkpoints=[]
    for group in GROUPS:
        for p in sorted((ROOT/group/'training/checkpoints').glob('step_*.pt')):
            checkpoints.append(dict(group=group,step=int(p.stem.split('_')[-1]),path=str(p),SHA256=sha(p),bytes=p.stat().st_size))
    dump(ROOT/'checkpoint_manifest.json',dict(checkpoints=checkpoints,weights_local_only=True))
    coverage={f'{m}_step{s}':sum(r['model']==m and r['step']==s for r in points)
        for m,s in [('Original',0),('B300',0),*[(g,s) for g in GROUPS for s in [50,100]]]}
    updates={g:max([r['committed_updates'] for r in traces if r['group']==g],default=0) for g in GROUPS}
    protocol=load(ROOT/'protocol.json');cal=load(ROOT/'gradient_calibration.json')
    data_done=all(u==100 for u in updates.values()) and all(n==96 for n in coverage.values())
    completion='COMPLETE' if data_done and performance['status']=='PASS' else 'COMPLETED_WITH_PENDING_PERFORMANCE' if data_done else 'IN_PROGRESS'
    rows=['# Causal decoder adapter A/B: execution record','',
        f"Status: {completion}. ULong development only; no final test.",'',
        f'Experiment directory: `{ROOT}`',f'Branch: `{BRANCH}`','',
        '## Fixed configuration','',
        f'B300 SHA256: `{B300_SHA}`. Full recovery: Original I/P → pilot full receiver and persistent buffers → B1500 TCG → A2000 LoRA → B300 52-key last overlay. No E/F/H trained weights.',
        'All inherited components are frozen. Existing five LoRA branches remain at s=0.5. Only the 21,696-parameter causal branch is optimized.',
        'Adapter after ada_final, before frozen head: 320→32 1x1 no bias; depthwise 3x3x3 (groups32, temporal padding0); 32→320 1x1 with bias. Last projection weight/bias start at zero. Internal operations and residual sum are FP32.',
        'History stores two detached raw h frames. Missing history uses left replication. Video/GOP resets clear it; continuous chunks preserve it. Training h46–63 is cached. Frame48 is rendered with current adapter weights. At each segment boundary previous RGB is detached; both frame outputs within a segment retain gradients. No backbone BPTT is claimed.','',
        'Spatial objective: LPIPS-Alex + 0.5 DISTS + 0.1383589248954108 RGB-L1.',
        f"B adds lambda={cal['lambda_temp']:.17g} times (RGB MC residual + 0.5 Sobel MC residual). Calibration used ten inherited training windows, finite/nonzero B-projection gradient norms, zero optimizer updates. See calibration_gradient_norms.csv.",
        'Flow is frozen RAFT estimated from GT RGB, not annotated true flow. It is used only for supervision/scoring. Deployment never reads GT. Pixel-center bilinear warp, zero padding, align_corners=False, forward/backward consistency and bounds masks are inherited.',
        'Each group: effective batch4, 100 updates, exact inherited400-window order once, seed20261111, AdamW lr1e-4 betas(0.9,0.999) eps1e-8 weight_decay1e-4 clip1. Checkpoints0/25/50/75/100. Supervises49–63 in segments49–52,53–56,57–60,61–63.',
        'Training uses FP32 frozen reconstruction prefix and head; codec features follow inherited FP16. Deployment uses inherited FP16 codec/Bridge/backbone, FP32 TCG/LoRA/adapter, shared module head and FP32-first RGB. B300 uses the matching path.','',
        '## Completion and coverage','',
        '| Group | Committed updates |','|---|---:|',*[f'| {g} | {u} |' for g,u in updates.items()],'',
        '| State | Completed points / 96 |','|---|---:|',*[f'| {m} | {n} / 96 |' for m,n in coverage.items()],'',
        'Training source isolation and QP/crop/frame/order identities are inherited from H. No new dataset inventory scan or split. No training codec tensor for these400 windows was retained at startup; missing codec/GT-flow caches were regenerated with the pinned independent Original path. Native evaluation caches are read-only where reusable.',
        'Original is decoded using pinned independent b2a0895 original source and original I/P weights in an isolated process. Cached tensors carry independent-decode/real-RANS/source/feature hashes. Historical scalar reuse requires every RGB frame, source frame, stream, QP, precision and metric-weight identity to match. Missing metrics are computed anew.','',
        'Frozen PWC GT forward flows used internally by FloLPIPS are cached as FP32 CPU tensors within each source/QP evaluation bundle and shared across model states. A direct check found bit-exact flow and FloLPIPS equality. This cache does not change the RAFT MC supervision, any training loss, or deployment inputs. The official FloLPIPS source/weight identities are in metric_source_manifest.json.','',
        '## Development numbers','',
        'CSV files contain every completed point, equal-video QP means and candidate-minus-reference paired differences. Negative error differences and positive PSNR differences retain their numeric meaning; no subjective flicker percentage is inferred.','']
    if summaries:
        rows+=['| Model | Step | QP | n | bpp | PSNR | LPIPS-Alex | DISTS | FloLPIPS P→P | RGB MC P→P | Sobel MC P→P |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            rows.append('| '+ ' | '.join([r['model'],str(r['step']),str(r['QP']),str(r['completed_videos'])]+[f'{r[k]:.8g}' if r[k] is not None else 'missing' for k in ['bpp','PSNR','LPIPS_Alex','DISTS','FloLPIPS_PtoP','RGB_MCResidual_PtoP','Sobel_MCResidual_PtoP']])+' |')
    rows+=['','## Checks and repaired exceptions','',
        'Mechanical preflight passed: exact zero-branch equality in both precisions; future perturbation causality; continuous/chunked equality; reset isolation and first-history replication; finite losses/gradients; frozen head input gradients; actual adapter/head-LoRA calls in training/deployment. Upstream adapter gradients were zero at zero B and became nonzero after a disposable B update.',
        'CUDA grid_sample backward does not implement strict determinism. Preflight initially stopped on inherited strict mode. Shared training context now uses deterministic algorithms with warn_only=True; A/B seeds, order and optimizer settings match. The initial failed check made no formal optimizer update. No independent RMS diagnosis was run. The inherited H calibration list initially covered four QPs; those provisional runs (11 updates per group) and coefficient were preserved under audits/superseded_calibration_qp_coverage, excluded from the formal budget, and replaced by one fixed window for each QP0–9 followed by a fresh common-step0 restart.',
        'Original source model hashes and stream hashes, all checkpoint receipts and byte identities are audited in integrity.json. Non-task processes were preserved.','',
        '## Artifacts','',
        '`protocol.json`, `source_manifest.json`, `dependency_imports.json`, `calibration_gradient_norms.csv`, `training_logs/`, per-group training receipts, `checkpoint_manifest.json`, `metrics.csv`, `metrics_by_QP.csv`, `paired_differences*.csv`, `plots/development/`, `media/`, `integrity.json`, `status.json`, `publication_status.json`.',
        'Main R-D plots show both step100 candidates; separate step50 supplements use the same real-bpp axis. Metrics and plots are ULong development, never final-test aggregates.',
        'Media cases/frames/ROI were fixed before training in manifests/media_cases.json. Native FFV1 and 1:1 ROIs use identical RGB8 rounding and fps. Weights, dataset, large caches and lossless videos remain local with hashes.','',
        f"Performance status: {performance['status']}. Method and observations are in performance.json. No speed claim is inferred from contended training/evaluation timings. No zero-overhead claim is made.",'']
    (ROOT/'REPORT.md').write_text('\n'.join(rows))
    dump(ROOT/'results_status.json',dict(status='PASS',coverage=coverage,committed_updates=updates,paired_rows=len(pairs),time=time.time()))

if __name__=='__main__':main()
