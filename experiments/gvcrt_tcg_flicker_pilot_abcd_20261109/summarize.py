"""Pre-registered arithmetic and objective execution records, no interpretation."""
from common_s2 import *
import statistics,collections
PRIMARY='MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP'

def results(group,step):
    p=ROOT/f'evaluation/{group}/step_{step:04d}/points';rows=[load(x)['result'] for x in p.glob('*.json')]
    keys={(r['sequence'],r['QP']) for r in rows};assert len(rows)==len(keys)
    return rows

def means(rows,key,q=None):
    rr=[r for r in rows if (q is None or r['QP']==q) and r.get(key) is not None];assert rr
    return statistics.mean(r[key] for r in rr)

def _screening():
    base=results('A2000',0)
    if len(base)!=36:return {}
    bm={(r['sequence'],r['QP']):r for r in base};history={};deltas=[];metrics=[];all_transitions=[]
    for group in ['Original','A2000',*GROUPS]:
        history[group]=[]
        for directory in sorted((ROOT/f'evaluation/{group}').glob('step_*')):
            step=int(directory.name.split('_')[1]);rr=results(group,step)
            if len(rr)!=36:continue
            assert {(r['sequence'],r['QP']) for r in rr}==set(bm)
            metrics+=rr
            for point in (directory/'points').glob('*.json'):
                item=load(point)['result']
                with (directory/'transitions'/(point.stem+'.csv')).open() as f:
                    for transition in csv.DictReader(f):all_transitions.append(dict(model=group,step=step,sequence=item['sequence'],QP=item['QP'],**transition))
            ratios=collections.defaultdict(list)
            for r in rr:
                b=bm[(r['sequence'],r['QP'])]
                assert (r['bitstream_SHA256'],r['real_bytes'],r['bpp'],r['codec_trace_SHA256'],r['source_RGB_SHA256'])==(b['bitstream_SHA256'],b['real_bytes'],b['bpp'],b['codec_trace_SHA256'],b['source_RGB_SHA256']),'EVALUATION_STREAM_OR_SOURCE_MISMATCH'
                record=dict(model=group,step=step,sequence=r['sequence'],QP=r['QP'],reference='fixed_A2000_s05')
                for key in ['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','MCResidual_RGB_RAFT_pixelcenter_v1_PtoP',PRIMARY]:
                    record[key+'_absolute_delta']=None if r.get(key) is None or b.get(key) is None else r[key]-b[key]
                    record[key+'_relative_percent']=None if r.get(key) is None or b.get(key) is None or abs(b[key])<=1e-12 else (r[key]/b[key]-1)*100
                deltas.append(record)
                if group in 'CD':
                    own={(t['sequence'],t['QP']):t for t in results(group,0)}
                    if (r['sequence'],r['QP']) in own:
                        ob=own[(r['sequence'],r['QP'])];od=dict(model=group,step=step,sequence=r['sequence'],QP=r['QP'],reference=f'{group}_step0_alignment_enabled')
                        for metric in ['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','MCResidual_RGB_RAFT_pixelcenter_v1_PtoP',PRIMARY]:
                            od[metric+'_absolute_delta']=None if r.get(metric) is None or ob.get(metric) is None else r[metric]-ob[metric]
                            od[metric+'_relative_percent']=None if r.get(metric) is None or ob.get(metric) is None or abs(ob[metric])<=1e-12 else (r[metric]/ob[metric]-1)*100
                        deltas.append(od)
                if r.get(PRIMARY) is not None and b.get(PRIMARY) is not None and b[PRIMARY]>1e-12:ratios[r['sequence']].append(r[PRIMARY]/b[PRIMARY])
            applicable=len(ratios)==12 and all(len(v)==3 for v in ratios.values())
            video_ratios={s:statistics.mean(v) for s,v in ratios.items() if len(v)==3}
            median=statistics.median(video_ratios.values()) if applicable else None
            better=sum(v<1 for v in video_ratios.values());guards={}
            for q in [0,5,9]:
                guards[str(q)]={k:means(rr,k,q)/means(base,k,q) for k in ['LPIPS','DISTS']}
            spatial_ok=all(v<=1.01 for d in guards.values() for v in d.values())
            passed=bool(applicable and median<=.95 and better>=8 and spatial_ok)
            paired_primary=[(r[PRIMARY],bm[(r['sequence'],r['QP'])][PRIMARY]) for r in rr if r.get(PRIMARY) is not None and bm[(r['sequence'],r['QP'])].get(PRIMARY) is not None]
            current_primary=statistics.mean(a for a,b in paired_primary) if paired_primary else None
            baseline_primary=statistics.mean(b for a,b in paired_primary) if paired_primary else None
            no_temporal_improvement=None if current_primary is None or baseline_primary is None else current_primary>=baseline_primary
            prior=[r for r in history[group] if r['step']>0]
            consecutive=passed and bool(prior and prior[-1]['single_validation_pass']) and step>0
            history[group].append(dict(step=step,complete_points=36,median_video_mean_QP_Sobel_ratio=median,videos_ratio_lt1=better,video_ratios=video_ratios,
                spatial_QP_ratios=guards,ratio_applicable=applicable,usable_temporal_points=len(paired_primary),absolute_Sobel_delta=None if current_primary is None or baseline_primary is None else current_primary-baseline_primary,single_validation_pass=passed,
                screening_status='NUMERIC_SCREEN_PASS' if consecutive else 'ONE_VALIDATION_PASS' if passed else 'NOT_APPLICABLE_NEAR_ZERO' if not applicable else 'NOT_PASSED',
                own_step0_comparison_available=group in 'CD' and len(results(group,0))==36,
                no_temporal_improvement=no_temporal_improvement,spatial_regression_over_one_percent=not spatial_ok))
    if metrics:write_csv(ROOT/'development_metrics.csv',metrics)
    if all_transitions:write_csv(ROOT/'temporal_metrics_by_frame.csv',all_transitions)
    if deltas:write_csv(ROOT/'paired_deltas_by_video_qp.csv',deltas)
    dump(ROOT/'screening_history.json',dict(version=1,primary=PRIMARY,scope='12 ULong development groups; no cross-dataset claim',history=history,numeric_candidates=[dict(group=g,validation_step=r['step'],status=r['screening_status']) for g,records in history.items() if g in GROUPS for r in records if r['screening_status']=='NUMERIC_SCREEN_PASS'],no_statistical_significance_claim=True))
    return history

def screening():
    with lock('locks/summary.lock',blocking=True):return _screening()

def should_stop(group):
    h=screening()[group];by_step={r['step']:r for r in h}
    assert 100 in by_step and 200 in by_step
    return all(by_step[s]['no_temporal_improvement'] and by_step[s]['spatial_regression_over_one_percent'] for s in [100,200])

def training_summary():
    from loss import TERMS
    plan=plan_rows();rows=[];coverage={};checkpoints=[];profiles={}
    for group in GROUPS:
        rr=[load(p) for p in sorted((ROOT/group/'training/receipts').glob('*.json'))]
        assert [r['additional_step'] for r in rr]==list(range(1,len(rr)+1))
        seen=[]
        for r in rr:
            expected=plan[4*(r['additional_step']-1):4*r['additional_step']]
            assert r['sample_keys']==[x['sample_key'] for x in expected]
            seen+=expected
            out=dict(group=group,step=r['additional_step'],seconds=r['seconds'],data_wait_seconds=r['data_wait_seconds'],gradient_norm=r['gradient_norm'],post_clip_norm=r['post_clip_norm'],peak_allocated_bytes=r['peak_allocated_bytes'],physical_GPU=r['physical_GPU'])
            for k in ['LPIPS','DISTS','RGB_L1','RAFT_motion','RGB_temporal','Sobel_temporal']:
                vals=[x['losses'][k] for x in r['samples'] if x['losses'][k] is not None]
                out[k]=statistics.mean(vals) if vals else None
            valid_frames=[x.get('frames_temporal_valid',sum(t['valid_fraction']>0 for t in x.get('raw_trace',[]))) for x in r['samples']]
            out['valid_temporal_supervised_frames']=sum(valid_frames)
            out['empty_temporal_supervised_frames']=sum(x.get('frames_temporal_empty',15-v) for x,v in zip(r['samples'],valid_frames))
            out['valid_temporal_windows']=r.get('batch_temporal_valid_windows',sum(v>0 for v in valid_frames))
            out['empty_temporal_windows']=r.get('batch_temporal_empty_windows',4-out['valid_temporal_windows'])
            total_losses=[x['total_loss'] for x in r['samples'] if x['total_loss'] is not None]
            out['total_loss']=statistics.mean(total_losses) if total_losses else None
            for k,v in r['group_gradient_norms'].items():out['gradient_'+k]=v
            for k,v in r['parameter_change_L2'].items():out['parameter_change_'+k]=v
            for term,parts in r['global_component_gradient_norms'].items():
                for part,norm in parts.items():out['gradient_'+term+'_'+part]=norm
            cosines=[x['spatial_temporal_cosine'] for x in r['samples'] if x['spatial_temporal_cosine'] is not None]
            out['mean_window_spatial_temporal_cosine']=statistics.mean(cosines) if cosines else None
            rows.append(out)
        coverage[group]=dict(committed_updates=len(rr),windows=len(seen),videos=len({x['identity'] for x in seen}),source_groups=len({x['source_group'] for x in seen}),
            source_pool_videos=8470,source_pool_groups=7969,QP_counts=dict(collections.Counter(x['external_qp'] for x in seen)),UVG=0,Vimeo=0,
            unique_windows=len({(x['identity'],tuple(x['source_frame_indices'])) for x in seen}),unique_crops=len({(x['identity'],tuple(x['source_frame_indices']),x['crop_x'],x['crop_y']) for x in seen}),
            exact_window_repeats=len(seen)-len({(x['identity'],tuple(x['source_frame_indices'])) for x in seen}),plan_coverage_fraction=len(seen)/1200,
            each_video_visits=dict(collections.Counter(x['identity'] for x in seen)),source_group_coverage_fraction=len({x['source_group'] for x in seen})/7969)
        profiles[group]=dict(first20_updates=[r for r in rows if r['group']==group][:20],total_training_seconds=sum(r['seconds'] for r in rr),
            functional_training_updates=load(ROOT/group/'smoke/completion.json')['committed_updates'] if (ROOT/group/'smoke/completion.json').exists() else None)
        for cp in sorted((ROOT/group/'training/checkpoints').glob('*.pt')):
            checkpoints.append(dict(group=group,filename=cp.name,local_path=str(cp),bytes=cp.stat().st_size,sha256=sha(cp)))
    if rows:write_csv(ROOT/'training_history.csv',rows)
    ga_path=ROOT/'gradient_audit.json'
    if ga_path.exists():
        ga=load(ga_path);ga['formal_effective_batch_gradient_angles']={g:[dict(step=r['additional_step'],angles=r.get('effective_batch_spatial_temporal_gradient_angles',{}),global_component_gradient_norms=r['global_component_gradient_norms'],parameter_change_from_A2000=r.get('parameter_change_from_A2000')) for r in [load(p) for p in sorted((ROOT/g/'training/receipts').glob('*.json'))] if r.get('effective_batch_spatial_temporal_gradient_angles')] for g in GROUPS};dump(ga_path,ga)
    dump(ROOT/'training_coverage.json',coverage);dump(ROOT/'runtime_profile.json',profiles);dump(ROOT/'checkpoint_manifest.json',dict(initial_checkpoint=dict(path=str(INIT),sha256=INIT_SHA),new_checkpoints=checkpoints))
    return coverage

def plots():
    env();import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=screening();p=ROOT/'plots';p.mkdir(exist_ok=True)
    rows=[]
    for group in GROUPS:
        for directory in sorted((ROOT/f'evaluation/{group}').glob('step_*')):
            step=int(directory.name.split('_')[1]);rr=results(group,step)
            if len(rr)==36:rows.append((group,step,rr))
    for metric in [PRIMARY,'MCResidual_RGB_RAFT_pixelcenter_v1_PtoP','LPIPS','DISTS','FloLPIPS']:
        fig,ax=plt.subplots(figsize=(8,4))
        for group in GROUPS:
            values=[(s,means(rr,metric)) for g,s,rr in rows if g==group and any(r.get(metric) is not None for r in rr)]
            if values:ax.plot([s for s,m in values],[m for s,m in values],marker='o',label=group)
        base=results('A2000',0)
        if len(base)==36 and any(r.get(metric) is not None for r in base):ax.axhline(means(base,metric),color='black',linestyle='--',label='Fixed A2000 s0.5')
        ax.set(xlabel='Committed additional updates',ylabel=metric+' (lower better)',title='ULong development subset:12 videos ×3 measured QPs');ax.legend();fig.tight_layout();fig.savefig(p/(metric+'_by_step.png'),dpi=140);plt.close(fig)
    for q in [0,5,9]:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,metric in zip(axes,['LPIPS','DISTS',PRIMARY]):
            for group in GROUPS:
                values=[(s,means(rr,metric,q)) for g,s,rr in rows if g==group and any(r.get(metric) is not None for r in rr)]
                if values:ax.plot([s for s,m in values],[m for s,m in values],marker='o',label=group)
            ax.set(xlabel='Committed updates',ylabel=metric+' ↓')
        axes[0].legend();fig.suptitle(f'ULong development12; external QP{q}; video equal weights');fig.tight_layout();fig.savefig(p/f'quality_temporal_QP{q}.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for group in GROUPS:
        hh=h.get(group,[])
        if hh:
            r=hh[-1];v=sorted(r['video_ratios'].items());ax.plot(range(len(v)),[t[1] for t in v],marker='o',label=f'{group} step{r["step"]}')
    ax.axhline(1,color='black',linestyle='--');ax.set(xlabel='Fixed sequence-sorted development video index',ylabel='Mean QP Sobel temporal ratio to A2000 ↓',title='Per-video distribution; QP0/5/9,12 ULong development sources');ax.legend();fig.tight_layout();fig.savefig(p/'per_video_temporal_ratio.png',dpi=140);plt.close(fig)

def report():
    coverage=training_summary();h=screening()
    lines=['# TCG flicker pilot A/B/C/D','',f'Snapshot UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}','',
        'Scope:12 existing ULong development sources, external QP0/5/9. No final test evaluation. Four independent single-GPU models.','',
        'Initialization: original A2000 LoRA over pilot receiver + B+1500 FP32 TCG. Fresh AdamW. Both training/evaluation inference gain0.5.','',
        'A: TCG+LoRA; B: A+Bridge mlp[3]; C: A+fixed causal original-reconstruction RAFT history warp; D: C+Bridge mlp[3].','',
        'Main temporal metric: MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP. I→P is separately recorded. Historical PWC metrics keep their prior sampler.','',
        'Empty GT-valid regions are recorded as missing, never as zero temporal error. Temporal metrics average only valid transitions; training temporal terms average valid frames within each nonempty window and then nonempty windows. Spatial and RAFT terms retain 15-frame and four-window averaging.','',
        'Standard CUDA grid_sample backward may use nondeterministic atomic accumulation; own-repeat relative gradient error is audited under1e-3. No bitwise gradient reproducibility claim.','',
        '|Group|Committed updates|Videos/groups/windows consumed|Stop reason|','|---|---:|---|---|']
    for g in GROUPS:
        c=coverage[g];p=ROOT/g/'training/completion.json';stop=load(p)['stop_reason'] if p.exists() else 'RUNNING' if (ROOT/g/'training/progress.json').exists() else 'NOT_STARTED'
        fail=ROOT/g/'training/failure.json'
        if fail.exists() and not p.exists():stop='FAILED: '+load(fail)['exception']
        lines.append(f'|{g}|{c["committed_updates"]}|{c["videos"]}/{c["source_groups"]}/{c["windows"]}|{stop}|')
    lines+=['','## Validation records','','|Group|Step|Median per-video Sobel ratio|Videos ratio<1|Numeric screen|','|---|---:|---:|---:|---|']
    for g,records in h.items():
        for r in records:lines.append(f'|{g}|{r["step"]}|{r["median_video_mean_QP_Sobel_ratio"]}|{r["videos_ratio_lt1"]}|{r["screening_status"]}|')
    lines+=['','Full per-video/QP/frame/transition values: development_metrics.csv and evaluation/*/step_*/. Raw and weighted training terms plus component gradient norms: training_history.csv and compressed receipts.','',
        'The numeric screen is a pre-registered pilot threshold; it is not a significance test or visual acceptance record. No automatic expansion or deployment.','',
        'Unavailable/failed stages remain incomplete in status.json and final_integrity.json. Videos and checkpoints are local only; manifests contain SHA256.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':env();report();plots()
