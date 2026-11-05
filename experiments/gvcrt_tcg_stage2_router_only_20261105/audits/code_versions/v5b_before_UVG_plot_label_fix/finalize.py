"""Persist measured tables and plots, then run independent integrity assertions."""
from common_s2 import *
from reuse import read_csv
from evaluate import videos,METRICS
import collections
QUALITY=('PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC')
LABELS={'GVC_RT_original':'GVC-RT original (independent)','TCG_B1500':'TCG B+1500','Router_best':'TCG B+1500 + Router (best trained)'}

def plot_rd(path,rows,title):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,8));axes=axes.flatten()
    for ax,metric in zip(axes,QUALITY):
        for model,label in LABELS.items():
            a=sorted([r for r in rows if r['model_state']==model],key=lambda r:float(r['bpp']))
            assert len(a)==3
            ax.plot([float(r['bpp']) for r in a],[float(r[metric]) for r in a],marker='o',label=label)
        ax.set_xlabel('Measured bpp');ax.set_ylabel(metric+(' ↑' if metric in ('PSNR','MS_SSIM') else ' ↓'));ax.grid(alpha=.25)
    axes[-1].axis('off');axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center',fontsize=9)
    fig.suptitle(title+'; QP0/5/9; measured points only');fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)

def group(v):
    if v['split']=='new_holdout_test':return 'ULong_new_holdout_32'
    if v['domain']=='ulong':return 'ULong_historical_6'
    if v['domain']=='hevc_b':return 'HEVC_B_5'
    assert v['domain']=='uvg' and v['sequence'] in ['HoneyBee','Jockey'];return 'UVG_historical_HoneyBee_Jockey_2'

def main():
    env();verify_sources();selection=load(ROOT/'selection.json');state=load(ROOT/'stopping_state.json');best=selection['best_trained_router']['step']
    original=read_csv(ROOT/'evaluation/reused/original_final.csv');base=read_csv(ROOT/'evaluation/reused/B_final.csv')
    gate=read_csv(ROOT/f'evaluation/final_router_{best:04d}/metrics.csv')
    for r in gate:r['model_state']='Router_best';r['Router_committed_step']=best
    allrows=original+base+gate;vmap={v['sequence']:v for v in videos('final')}
    assert len(allrows)==405 and len({(r['model_state'],r['sequence'],int(r['QP'])) for r in allrows})==405
    for r in allrows:r['dataset_plot_group']=group(vmap[r['sequence']])
    write_csv(ROOT/'metrics_gate.csv',gate);write_csv(ROOT/'metrics_comparison.csv',allrows)
    bykey={(r['model_state'],r['sequence'],int(r['QP'])):r for r in allrows};streams=[];deltas=[]
    for v in videos('final'):
        for q in [0,5,9]:
            b=bykey[('TCG_B1500',v['sequence'],q)];g=bykey[('Router_best',v['sequence'],q)]
            streams.append(dict(sequence=v['sequence'],QP=q,baseline_real_bytes=b['real_bytes'],Router_real_bytes=g['real_bytes'],
                baseline_stream_sha256=b['bitstream_sha256'],Router_stream_sha256=g['bitstream_sha256'],same_bytes=int(b['real_bytes'])==int(g['real_bytes']),same_stream_sha256=b['bitstream_sha256']==g['bitstream_sha256']))
            for comparator in ['TCG_B1500','GVC_RT_original']:
                other=bykey[(comparator,v['sequence'],q)]
                for m in METRICS:
                    d=float(g[m])-float(other[m]);deltas.append(dict(sequence=v['sequence'],QP=q,dataset=group(v),candidate='Router_best',reference=comparator,metric=m,absolute_delta=d,relative_delta=None if float(other[m])==0 else d/float(other[m])))
    write_csv(ROOT/'stream_identity_checks.csv',streams);write_csv(ROOT/'paired_deltas.csv',deltas)
    datasetrows=[]
    for name,expected in [('ULong_new_holdout_32',32),('ULong_historical_6',6),('HEVC_B_5',5),('UVG_historical_HoneyBee_Jockey_2',2)]:
        rows=[]
        for model in LABELS:
            for q in [0,5,9]:
                members=[r for r in allrows if r['model_state']==model and r['dataset_plot_group']==name and int(r['QP'])==q]
                assert len(members)==expected
                rows.append(dict(dataset=name,model_state=model,QP=q,video_count=expected,weighting='equal video within this split and domain; no cross-dataset averaging',**{k:sum(float(r[k]) for r in members)/expected for k in METRICS}))
        datasetrows.extend(rows);plot_rd(ROOT/f'plots/datasets/{name}.png',rows,name)
    write_csv(ROOT/'rd_points_by_dataset.csv',datasetrows)
    for sequence,v in vmap.items():plot_rd(ROOT/f'plots/per_video/{v["domain"]}_{sequence}.png',[r for r in allrows if r['sequence']==sequence],f'{v["domain"]}: {sequence}')
    training=[];masks=[]
    for p in sorted((ROOT/'receipts').glob('update_*.json.gz')):
        x=load(p)
        for r in x['ranks']:
            training.append(dict(step=x['step'],rank=r['rank'],domain=r['domain'],video=r['video'],external_qp=r['external_qp'],seconds=r['seconds_before_commit'],
                gradient_norm_local=r['local_gradient_norm'],gradient_norm_mean=r['mean_gradient_norm'],Router_change_L2=r['parameter_change_L2'],**r['losses']))
            masks.append(dict(kind='training_last_frame',step=x['step'],rank=r['rank'],**r['mask_statistics']))
    write_csv(ROOT/'training_history.csv',training)
    development=read_csv(ROOT/'evaluation/reused/B_development.csv')
    for r in development:r['candidate']='baseline_bypass';r['Router_updates']=0
    for s in [0]+[c['step'] for c in state['checks']]:
        a=read_csv(ROOT/f'evaluation/development_router_{s:04d}/metrics.csv')
        for r in a:r['candidate']='enabled0' if s==0 else 'trained_router';r['Router_updates']=s
        development+=a
    write_csv(ROOT/'development_metrics.csv',development)
    for p in sorted((ROOT/f'evaluation/final_router_{best:04d}/masks').glob('*.csv')):
        for r in read_csv(p):masks.append(dict(kind='final',point=p.stem,step=best,**r))
    write_csv(ROOT/'mask_statistics.csv',masks)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for component in ['total_loss','LPIPS','weighted_DISTS','weighted_motion_loss','weighted_reconstruction_L1']:
        steps=sorted({int(r['step']) for r in training});ax.plot(steps,[sum(r[component] for r in training if r['step']==s)/4 for s in steps],label=component,linewidth=.8)
    ax.set_xlabel('Committed Router update');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/training_loss.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.plot([0]+[c['step'] for c in state['checks']],[state['enabled0']['score']]+[c['score'] for c in state['checks']],marker='o',label='Router development score');ax.axhline(1,label='B+1500 bypass',color='gray',linestyle='--');ax.set_xlabel('Committed Router update');ax.set_ylabel('Normalized development FloLPIPS ↓');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/development_trajectory.png',dpi=150);plt.close(fig)
    from integrity import final_check
    audit=final_check()
    coverage={}
    for domain in ['uvg','ulong']:
        a=[r for r in training if r['domain']==domain];coverage[domain]=dict(samples=len(a),distinct_video_identities=len({r['video'] for r in a}))
    dump(ROOT/'training_coverage.json',coverage)
    report=['# Stage2 Router-only execution report',f'Snapshot UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',
        f'Integrity: {audit["status"]}. Committed Router updates: {audit["committed_router_updates"]}. Stop reason: {state["status"]}.',
        f'Best trained Router: step{best}; normalized development FloLPIPS={selection["best_trained_router"]["score"]:.12g}. Final selection: {selection["final_selection"]}.',
        'Only Router was trained. Baseline is B+1500 at cumulative TCG step6500. Receiver and all frozen components are recorded in baseline_provenance.json.',
        'Protocol: protocol.json. Training loss is the actual inherited Stage1 implementation; not claimed identical to the original paper training loss.',
        'Final testing:135 fresh Router RANS encodes with new independent decoders. Original135 and B+1500 135 reused only after hash and aggregation audits; provenance in audits/reuse_*.json.',
        'Historical subsets remain historical references. UVG contains only HoneyBee and Jockey. Equal video weighting applies separately within each dataset/split and QP.',
        'Mask denotes historical residual suppression weight. PWC sampler retains the documented existing pixel-center/W−1 normalization behavior. No strict S2VC reproduction claim.',
        'Seven quality metrics and actual bitrate are tabulated below; no rate interpolation or BD-rate is computed.',
        '|Dataset|Model|QP|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|WarpGT|WarpRecon|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in datasetrows:report.append('|'+ '|'.join([r['dataset'],LABELS[r['model_state']],str(r['QP'])]+[f'{float(r[k]):.9g}' for k in ['bpp',*QUALITY]])+'|')
    report+=['\nDevelopment checks:\n','```json',json.dumps(state,indent=2),'```','\nCoverage:\n','```json',json.dumps(coverage,indent=2),'```',
        '\nLimitations and execution records: initialized enabled0 is measured, not identical to bypass; repeated historical samples are not new independent videos. Initial engineering harness call/dtype fixes and all retries remain in logs and repair audits. Latency timing, if shared GPUs are active, is not presented as a dedicated speed measurement.']
    (ROOT/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    mutable={'artifact_manifest.json','run_status.json','active_processes.json','runner_identity.json','launcher_receipt.json','publication_status.json','publication_allowlist.json','publication_remote_receipt.json'}
    dump(ROOT/'artifact_manifest.json',dict(scope='Immutable experiment results and receipts; mutable execution/publication snapshots are excluded',excluded_root_snapshots=sorted(mutable),
        files=[dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.png','.gz'] and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','checkpoints','reference_tree','runtime_tree','vendor']) and not (p.parent==ROOT and p.name in mutable)]))
if __name__=='__main__':main()
