"""Measured two-model dataset/sequence curves and factual final report."""
from common_s2 import *
from evaluate import videos,METRICS,read_csv
import collections

QUALITY=['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']
LABELS={'B1500_matched_FP32':'B+1500 matched-FP32 baseline (pilot receiver + TCG)',
    'Generator_ConvLoRA':'Generator Conv-LoRA rank4 (dev-selected checkpoint)'}

def group(v):
    if v['domain']=='ulong_new' and v['split']=='new_holdout_test':return 'ULong_new_holdout_32'
    if v['domain']=='ulong' and v['split']=='historical_reference':return 'ULong_historical_6'
    if v['domain']=='hevc_b' and v['split']=='historical_reference':return 'HEVC_B_5'
    assert v['domain']=='uvg_holdout' and v['sequence'] in ['HoneyBee','Jockey'] and v['split']=='historical_reference'
    return 'UVG_historical_HoneyBee_Jockey_2'

def plot(path,rows,title):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,9));axes=axes.flatten()
    for ax,metric in zip(axes,QUALITY):
        for (model,label),style in zip(LABELS.items(),['--','-']):
            pts=sorted([r for r in rows if r['model']==model],key=lambda r:float(r['bpp']));assert len(pts)==3
            ax.plot([float(r['bpp']) for r in pts],[float(r[metric]) for r in pts],marker='o',linestyle=style,label=label)
        ax.set_xlabel('Measured bpp');ax.set_ylabel(metric+(' ↑' if metric in ['PSNR','MS_SSIM'] else ' ↓'));ax.grid(alpha=.25)
    axes[-1].axis('off');axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='center',fontsize=8)
    fig.suptitle(title+'; three measured QPs0/5/9, no extrapolation');fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)

def main():
    env();verify_sources();selection=load(ROOT/'selection.json');control=load(ROOT/'stopping_state.json');best=selection['best_trained']['step']
    vmap={v['sequence']:v for v in videos('final')};rows=[]
    for candidate,step,label in [('baseline',None,'B1500_matched_FP32'),('lora',best,'Generator_ConvLoRA')]:
        data=read_csv(ROOT/'evaluation'/evaluation_id('final',step,candidate)/'metrics.csv');assert len(data)==135
        for r in data:r.update(model=label,LoRA_step=0 if step is None else step,LoRA_checkpoint_sha256='' if step is None else selection['best_trained']['checkpoint_sha256'],
            TCG_initialization_sha256=BASELINE_SHA,pilot_receiver_sha256=PILOT_SHA,dataset=group(vmap[r['sequence']]))
        rows+=data
    assert len({(r['model'],r['sequence'],int(r['QP'])) for r in rows})==270
    write_csv(ROOT/'metrics.csv',rows);write_csv(ROOT/'metrics_lora.csv',[r for r in rows if r['model']=='Generator_ConvLoRA'])
    bykey={(r['model'],r['sequence'],int(r['QP'])):r for r in rows};deltas=[]
    for v in videos('final'):
        for q in [0,5,9]:
            a=bykey[('Generator_ConvLoRA',v['sequence'],q)];b=bykey[('B1500_matched_FP32',v['sequence'],q)]
            for m in METRICS:
                d=float(a[m])-float(b[m]);deltas.append(dict(dataset=group(v),sequence=v['sequence'],QP=q,metric=m,absolute_delta=d,
                    relative_delta=None if float(b[m])==0 else d/float(b[m])))
    write_csv(ROOT/'paired_deltas.csv',deltas)
    datasetrows=[]
    for name,count in [('ULong_new_holdout_32',32),('ULong_historical_6',6),('HEVC_B_5',5),('UVG_historical_HoneyBee_Jockey_2',2)]:
        data=[]
        for model in LABELS:
            for q in [0,5,9]:
                subset=[r for r in rows if r['model']==model and r['dataset']==name and int(r['QP'])==q];assert len(subset)==count
                data.append(dict(dataset=name,model=model,QP=q,video_count=count,weighting='equal videos within this split/domain/QP',
                    **{m:sum(float(r[m]) for r in subset)/count for m in METRICS}))
        datasetrows+=data;plot(ROOT/f'plots/datasets/{name}.png',data,name)
    write_csv(ROOT/'rd_points_by_dataset.csv',datasetrows)
    for name,v in vmap.items():plot(ROOT/f'plots/per_video/{v["domain"]}_{name}.png',[r for r in rows if r['sequence']==name],v['domain']+': '+name)
    training=[];runtime=[];plan=plan_rows();receipt_files=sorted((ROOT/'receipts').glob('update_*.json.gz'));n=len(receipt_files)
    for p in receipt_files:
        receipt=load(p)
        for r in receipt['ranks']:
            x=dict(step=receipt['step'],rank=r['rank'],sample_key=r['sample_key'],video=r['video'],domain=r['domain'],external_QP=r['external_qp'],
                window_start=r['source_frame_indices'][0],crop=json.dumps(r['crop']),seconds_before_commit=r['seconds_before_commit'],
                gradient_norm_local=r['local_gradient_norm'],gradient_norm_mean=r['mean_gradient_norm'],LoRA_change_L2=r['parameter_change_L2'],**r['losses'])
            for key in ['stage3_LoRA','head_LoRA']:
                x[key+'_gradient_norm_mean']=r['group_gradient_norms_mean'][key]
                x[key+'_parameter_change_L2']=r['group_parameter_changes_L2'][key]
                x[key+'_relative_change_from_initialization']=r['group_relative_changes_from_LoRA_initialization'][key]
            training.append(x);runtime.append(dict(kind='training_last_frame',step=receipt['step'],rank=r['rank'],**r['runtime_statistics']))
    write_csv(ROOT/'training_history.csv',training)
    coverage={}
    for domain in ['uvg','ulong']:
        data=[r for r in plan[:4*n] if r['domain']==domain];ids={r['video'] for r in data}
        windows={(r['video'],tuple(r['source_frame_indices'])) for r in data}
        crops={(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y']) for r in data}
        exact={(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y'],r['external_qp']) for r in data}
        coverage[domain]=dict(samples=len(data),unique_identities=len(ids),unique_windows=len(windows),unique_crops=len(crops),exact_combinations=len(exact),
            exact_repeats=len(data)-len(exact),QP_counts=dict(collections.Counter(str(r['external_qp']) for r in data)),
            identity_visits=dict(collections.Counter(r['video'] for r in data)),ULong_pool=8470 if domain=='ulong' else None,
            ULong_identity_coverage=len(ids)/8470 if domain=='ulong' else None,new_independent_source_claim=False)
    dump(ROOT/'training_coverage.json',dict(committed_optimizer_updates=n,rank_samples=4*n,initial_cursor=1500,next_cursor=1500+n,domains=coverage))
    development=[];streams=[]
    for candidate,step in [('baseline',None)]+[('lora',s) for s in [0]+[x['step'] for x in control['checks']]]:
        root=ROOT/'evaluation'/evaluation_id('development',step,candidate);data=read_csv(root/'metrics.csv')
        for r in data:r.update(candidate=candidate,LoRA_step=0 if step is None else step)
        development+=data
        if candidate=='lora':streams+=read_csv(root/'stream_identity_checks.csv')
    streams+=read_csv(ROOT/'evaluation'/evaluation_id('final',best)/'stream_identity_checks.csv')
    write_csv(ROOT/'development_metrics.csv',development);write_csv(ROOT/'stream_identity_checks.csv',streams)
    for candidate,step in [('baseline',None),('lora',best)]:
        for p in (ROOT/'evaluation'/evaluation_id('final',step,candidate)/'masks').glob('*.csv'):
            runtime += [dict(kind='final_'+candidate,point=p.stem,**r) for r in read_csv(p)]
    write_csv(ROOT/'LoRA_runtime_statistics.csv',runtime)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    means=collections.defaultdict(list)
    for r in training:means[r['step']].append(r)
    steps=sorted(means);fig,ax=plt.subplots(figsize=(10,5))
    for metric in ['total_loss','LPIPS','weighted_DISTS','weighted_motion_loss','weighted_reconstruction_L1']:
        ax.plot(steps,[sum(x[metric] for x in means[s])/4 for s in steps],label=metric,linewidth=.8)
    ax.set_xlabel('Committed LoRA optimizer update');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/training_loss.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.plot([0]+[c['step'] for c in control['checks']],[1.]+[c['score'] for c in control['checks']],marker='o',label='LoRA normalized dev FloLPIPS')
    ax.axhline(1,linestyle='--',color='gray',label='Matched B1500 baseline');ax.set_xlabel('Committed LoRA optimizer update');ax.set_ylabel('Normalized dev FloLPIPS ↓');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots/development_trajectory.png',dpi=150);plt.close(fig)
    from integrity import final_check
    audit=final_check(persist=False)
    report=['# Generator Conv-LoRA execution report',
        'UTC: '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        f'Integrity:{audit["status"]}. Committed LoRA updates:{n}. Stop reason:{control["status"]}.',
        f'Best trained dev step:{best}; score:{selection["best_trained"]["score"]:.12g}; final retention selection:{selection["final_selection"]}. Both baseline and best trained LoRA were tested135points each.',
        'Baseline: original I/P codec plus exact pilot Bridge/generator and B+1500 FP32 TCG overlay (TCG cumulative6500). This composite is not original GVC-RT weights. No Router or prior gating checkpoint was loaded.',
        'The pilot SHA transcription correction was explicitly authorized by the user; original blocked records and correction reason are preserved in audits/sha_correction_20261106/. All actual weights were rehashed; checks were not waived.',
        'Only59392 LoRA parameters train. Four rank4 parallel1x1+3x3 updates sit beside stage3 depthwise3x3 layers (320→4→320), not a dense weight factorization; the head branch is320→4→192. alpha/r=1, no added bias, up kernels zero-initialized. TCG,codec,Bridge,original generator and quality networks remain frozen.',
        'Fresh AdamW lr5e-5,betas(.9,.999),eps1e-8,weight_decay1e-4,global clip1. No LR schedule or joint unfreezing. Effective batch4=2UVG+2ULong; source cursor1500; native64frames/256crop; original warmup and15 supervised-frame mean.',
        'Loss.py and VJP4096/unscale/order retain historical SHA. LPIPS,DISTS,RGBL1 andRAFT-motion gradient contributions were checked. Frozen codec rate is a constant/monitoring term for LoRA; Identity proxy is zero.',
        'LoRA and TCG are FP32 internally; deployment nonTCG retains historical FP16 and scoring is RGB float-before-clamp. Stage3/head explicitly use safe PyTorch operators from both generator entries; actual CUDA-kernel availability is recorded. No native fused LoRA kernel is claimed.',
        'Baseline and every candidate independently decoded the same verified pre-encoded streams. Fresh encoding was used only for the fixed engineering RANS identity probes. Every dev/final point has exact paired bpp,byte and stream-SHA checks; decoded codec feature traces agree.',
        'Four dataset/split groups are separate; video weights are equal within each group/QP. UVG historical is only HoneyBee/Jockey. Each curve uses the3 measured QPs0/5/9; no extrapolation orBD-rate.',
        'GT-flow and reconstruction-flow PWC metrics remain separately named with the inherited sampler. No scientific-effectiveness, convergence or next-stage conclusion is drawn.',
        '|Dataset|Model|QP|Videos|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|GT-flow warp|Recon-flow warp|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in datasetrows:
        report.append('|'+ '|'.join([r['dataset'],LABELS[r['model']],str(r['QP']),str(r['video_count'])]+[f'{float(r[k]):.10g}' for k in ['bpp','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']])+'|')
    report += ['\nDevelopment validation and patience records:\n','```json',json.dumps(control,indent=2),'```',
        '\nActual training data coverage (identity_visits are in training_coverage.json):\n','```json',json.dumps({k:{n:v for n,v in d.items() if n!='identity_visits'} for k,d in coverage.items()},indent=2),'```',
        '\nFiles: metrics.csv,development_metrics.csv,training_history.csv,rd_points_by_dataset.csv,stream_identity_checks.csv,checkpoints_manifest.json,selection.json,plots/datasets/,plots/per_video/,audits/. Publication status is separate in publication_status.json. Checkpoints and streams stay local. Shared-device operational timings are not a dedicated inference benchmark.']
    import re
    text=re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(report)+'\n')
    (ROOT/'REPORT.md').write_text(text)
    excluded={'artifact_manifest.json','status.json','run_status.json','active_processes.json','runner_identity.json','publication_status.json','publication_allowlist.json','publication_remote_receipt.json','final_integrity.json'}
    files=[dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
        and p.suffix in ['.json','.csv','.md','.png','.gz'] and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts','checkpoints','logs','runtime_tree','reference_tree','vendor'])
        and not (p.parent==ROOT and p.name in excluded)]
    raw=(json.dumps(audit,indent=2,allow_nan=False)+'\n').encode();files.append(dict(path='final_integrity.json',size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    dump(ROOT/'artifact_manifest.json',dict(scope='Immutable task products; source code has separate source_manifest.json',files=files))
    dump(ROOT/'final_integrity.json',audit)
    print('FINAL_REPORT_METRICS_PLOTS_INTEGRITY_PASS')

if __name__=='__main__':main()
