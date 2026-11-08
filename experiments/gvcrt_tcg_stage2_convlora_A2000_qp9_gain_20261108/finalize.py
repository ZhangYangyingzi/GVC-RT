"""Measured fixed-QP gain tables and plots, with no scientific interpretation."""
from common_s2 import *
from audit import *

def main():
    env();verify_sources();assert load(ROOT/'audits/smoke.json')['status']=='PASS'
    formal_root=ROOT/'evaluation/s05';files=list((formal_root/'points').glob('*.json'));assert len(files)==32
    endpoint=read_csv(ROOT/'endpoint_metrics.csv');assert len(endpoint)==64
    combined=[];half=[];checks=[]
    for r0 in endpoint:
        r=dict(r0)
        for k in METRICS:r[k]=float(r[k])
        r['QP']=int(r['QP']);r['inference_gain']=float(r['inference_gain']);combined.append(r)
    for v in video_list():
        b=validate_point(BASE,v);f=validate_point(FULL,v);m=validate_point(formal_root,v,.5);compare_streams(v,[b,m,f])
        assert b[0]['metric_hashes']==m[0]['metric_hashes']==f[0]['metric_hashes']
        r=dict(m[0]['result']);r.update(inference_gain=.5,measurement_reused=False,independent_decode_performed_this_run=True,checkpoint_sha256=CHECKPOINT_SHA)
        half.append(r);combined.append(r)
        checks.append(dict(domain=v['domain'],sequence=v['sequence'],split='development',external_QP=9,
            bitstream_sha256=r['bitstream_sha256'],real_bytes=r['real_bytes'],bpp=r['bpp'],bytes_equal=True,bpp_equal=True,hash_equal=True,
            source_RGB_equal=True,frame_indices_equal=True,actual_QPs_equal=True,codec_feature_trace_equal=True,GT_flow_hashes_equal=True))
    assert len(combined)==96 and len({(*paired_key(r),r['inference_gain']) for r in combined})==96
    assert all(math.isfinite(float(r[m])) for r in combined for m in METRICS)
    write_csv(ROOT/'metrics_s05.csv',half);write_csv(ROOT/'metrics_combined.csv',combined);write_csv(ROOT/'stream_identity_checks.csv',checks)
    indexed={(paired_key(r),r['inference_gain']):r for r in combined};deltas=[];means={};counts={}
    for gain in [0.,.5,1.]:
        rows=[r for r in combined if r['inference_gain']==gain];assert len(rows)==32
        means[str(gain)]={m:sum(float(r[m]) for r in rows)/32 for m in METRICS}
    changes={}
    for reference in [0.,1.]:
        name=f's0.5_vs_s{reference:g}';changes[name]={};counts[name]={}
        for metric in METRICS:
            cv=means['0.5'][metric];rv=means[str(reference)][metric];lower=metric not in ['PSNR','MS_SSIM']
            changes[name][metric]=dict(candidate_mean=cv,reference_mean=rv,absolute_delta=cv-rv,
                relative_percent=(cv/rv-1)*100 if rv!=0 else None,lower_is_better=lower if metric not in ['bpp','kbps'] else None)
            ds=[]
            for v in video_list():
                pair=(v['domain'],v['sequence'],'development',9);cand=indexed[pair,.5];ref=indexed[pair,reference]
                x=float(cand[metric]);y=float(ref[metric]);delta=x-y;ds.append(delta)
                deltas.append(dict(domain=v['domain'],sequence=v['sequence'],split='development',QP=9,candidate_gain=.5,reference_gain=reference,
                    metric=metric,candidate=x,reference=y,absolute_delta=delta,relative_percent=(x/y-1)*100 if y!=0 else None,
                    lower_is_better=lower if metric not in ['bpp','kbps'] else None))
            counts[name][metric]=dict(videos=32,improved=sum(d<0 if lower else d>0 for d in ds),
                worsened=sum(d>0 if lower else d<0 for d in ds),equal=sum(d==0 for d in ds),
                direction='rate_identity_only' if metric in ['bpp','kbps'] else 'lower_is_better' if lower else 'higher_is_better')
    write_csv(ROOT/'paired_deltas.csv',deltas)
    summary=dict(status='COMPLETE',videos=32,external_QP=9,checkpoint_sha256=CHECKPOINT_SHA,training_updates=0,inference_gains=[0.,.5,1.],
        weighting='equal videos; inherited64frame/63transition within-video aggregation; PSNR from total pixel SSE',means=means,
        changes=changes,video_direction_counts=counts,relative_change_definition='(mean_candidate/mean_reference-1)*100',
        endpoint_method='64 audited historical endpoints reused after exact full-sequence endpoint reproduction',
        objective_comparisons=dict(s05_mean_FloLPIPS_and_DISTS_lower_than_s1=all(means['0.5'][m]<means['1.0'][m] for m in ['FloLPIPS','DISTS']),
            s05_mean_FloLPIPS_lower_than_s0=means['0.5']['FloLPIPS']<means['0.0']['FloLPIPS'],
            s05_mean_DISTS_higher_than_s0=means['0.5']['DISTS']>means['0.0']['DISTS']))
    dump(ROOT/'summary.json',summary)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    (ROOT/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,m in zip(axes,['FloLPIPS','DISTS','LPIPS']):
        ax.plot([0,.5,1],[means[str(g)][m] for g in [0.,.5,1.]],marker='o');ax.set_xticks([0,.5,1]);ax.set_xlabel('Extra inference gain s')
        ax.set_ylabel(m+' (lower is better)');ax.grid(alpha=.3)
    fig.suptitle('A2000, fixed external QP9,32 development videos; measured gain points, not an R-D curve');fig.tight_layout();fig.savefig(ROOT/'plots/mean_metrics_vs_gain.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(17,19));xx=list(range(32));names=[v['sequence'] for v in video_list()]
    for ax,m in zip(axes.flatten(),QUALITY):
        for reference,marker in [(0.,'o'),(1.,'x')]:
            vals=[next(r['absolute_delta'] for r in deltas if r['sequence']==seq and r['metric']==m and r['reference_gain']==reference) for seq in names]
            ax.plot(xx,vals,marker=marker,linewidth=.7,label=f's0.5 minus s{reference:g}')
        ax.axhline(0,color='gray',linestyle='--');ax.set_ylabel(m+(' (higher better)' if m in ['PSNR','MS_SSIM'] else ' (lower better)'))
        ax.set_xticks(xx);ax.set_xticklabels([str(i+1) for i in xx],fontsize=7);ax.set_xlabel('Video index, fixed lexicographic order');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes.flatten()[-1].axis('off');axes.flatten()[-1].text(0,1,'Video index mapping: plots/video_index.csv\nFixed external QP9, A2000 only.\nAbsolute metric differences; no BD-rate.',va='top')
    fig.tight_layout();fig.savefig(ROOT/'plots/per_video_paired_deltas.png',dpi=150);plt.close(fig)
    write_csv(ROOT/'plots/video_index.csv',[dict(index=i+1,sequence=seq) for i,seq in enumerate(names)])
    report=['# A2000 fixed-QP inference gain diagnostic',
        'Status: COMPLETE (local evaluation, audits, tables and plots). Publication is recorded separately in publication_status.json.',
        f'Checkpoint: {CHECKPOINT.name}; SHA256 {CHECKPOINT_SHA}. A_control LoRA step2000; frozen TCG initialization cumulative6500. Composite B1500 matched-FP32 is not original GVC-RT.',
        'No training,optimizer,loss changes,weight updates or new checkpoint. Five existing LoRA branches only: '+', '.join(TARGETS)+'. Runtime multiplier s multiplies the FP32 LoRA delta, then FP32 addition and cast to base output dtype. rank4,alpha4,alpha/r1 unchanged. TCG,Bridge,base convolutions and final RGB are not gain-scaled.',
        'Scope: existing32 development videos,externalQP9 only,full64frames. The same gain applies to all63P frames including inherited internalQP changes. Separate new model and empty temporal state per measured video/gain. Smoke sequence chosen lexicographically before reading new metrics.',
        'Source/crop/frame RGB/actual QP/stream bytes/bpp/SHA/codec traces and GT-flow hashes agree. Full-sequence endpoint reconstructions are exact; inherited metric tolerances unchanged.64 historical endpoint points reused after audits;32 s0.5 points independently decoded,including the matching smoke point counted once.',
        'Metrics: matched_rgb_fp32_v2,float before clamp/range conversion and native crop. Spatial metrics aggregate64frames; temporal metrics63transitions; PSNR uses total pixel SSE. PWC sampler remains pixel-center,align_cornersFalse,W-1/H-1 normalization,invalidzero,full-image denominator. GT-flow and reconstruction-flow errors remain separate.',
        'Means weight videos equally. Relative percent=(mean_candidate/mean_reference-1)*100. No otherQP,checkpoint,finaltest,training or gain scan was run. These are fixed-QP gain curves,not R-D curves or BD-rate.',
        '## Measured means', '|s|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|GT-flow warp|Recon-flow warp|', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for gain in [0.,.5,1.]:report.append('|'+str(gain)+'|'+'|'.join(f'{means[str(gain)][m]:.12g}' for m in ['bpp',*QUALITY])+'|')
    report+=['## Absolute and relative changes / video counts','```json',json.dumps(dict(changes=changes,counts=counts),indent=2),'```',
        '## Objective boolean records','```json',json.dumps(summary['objective_comparisons'],indent=2),'```',
        '## Files','metrics_s05.csv;endpoint_metrics.csv;metrics_combined.csv;paired_deltas.csv;summary.json;stream_identity_checks.csv;plots/mean_metrics_vs_gain.png;plots/per_video_paired_deltas.png;plots/video_index.csv; evaluation/s05/{points,frames,transitions,codec_trace,masks};endpoints/s0/ and endpoints/s1/;audits/smoke.json;audits/endpoint_reuse.json;sources.json;final_integrity.json. Historical source files remain unchanged.']
    (ROOT/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    verify_sources()
    plots=['plots/mean_metrics_vs_gain.png','plots/per_video_paired_deltas.png']
    assert all((ROOT/p).is_file() for p in plots)
    assert not list(ROOT.rglob('*.pt')) and not list(ROOT.rglob('*.pth')) and not list(ROOT.rglob('*.bin'))
    integrity=dict(status='PASS',formal_unique_points=32,endpoint_points=64,combined_unique_points=96,checkpoint_sha256=CHECKPOINT_SHA,
        training_updates=0,optimizer_created=False,no_weight_changes=True,source_experiment_unchanged=True,all_five_gain_branches_executed=True,
        smoke_endpoint_reconstructions_exact=True,inherited_metric_tolerances_unchanged=True,bpp_hash_codec_trace_checks='PASS',
        source_RGB_actual_QPs_GTflow_equal=True,frame_and_transition_aggregation=True,no_nonfinite_metrics=True,
        final_test_run=False,other_QPs_run=False,other_checkpoints_run=False,all_new_outputs_under_DEST=True,plots=plots,time=time.time())
    dump(ROOT/'final_integrity.json',integrity)
    dump(ROOT/'artifact_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.json','.csv','.md','.png'] and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','logs','large_artifacts'])
        and p.name not in ['artifact_manifest.json','status.json','publication_status.json','publication_remote_receipt.json','active_processes.json','runner_identity.json','launcher_receipt.json']]))
    print('GAIN32_LOCAL_FINAL_INTEGRITY_PASS')

if __name__=='__main__':main()
