"""Per-QP measured gain tables and figures; no pooled-QP model selection."""
from common_s2 import *
from audit import *

def typed(row):
    row=dict(row);row['QP']=int(row['QP']);row['inference_gain']=float(row['inference_gain'])
    for m in METRICS:row[m]=float(row[m])
    return row

def plots_for_qp(qp,means,deltas):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    names=[v['sequence'] for v in video_list()];folder=ROOT/'plots';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(17,9))
    for ax,m in zip(axes.flatten(),QUALITY):
        ax.plot([0,.5,1],[means[str(g)][m] for g in [0.,.5,1.]],marker='o');ax.set_xticks([0,.5,1]);ax.set_xlabel('Extra inference gain s')
        ax.set_ylabel(m+(' (higher is better)' if m in ['PSNR','MS_SSIM'] else ' (lower is better)'));ax.grid(alpha=.25)
    axes.flatten()[-1].axis('off');axes.flatten()[-1].text(0,.8,'s0: B+1500 matched-FP32, no LoRA contribution\ns0.5/s1: fixed A2000, rank4, intrinsic alpha4\nMeasured gain points, not an R-D curve.\nNo cross-QP pooled mean or automatic selection.',fontsize=9)
    fig.suptitle(f'ULong development32 videos, fixed A2000, external QP{qp}; equal videos');fig.tight_layout();fig.savefig(folder/f'qp{qp}_mean_metrics_vs_gain.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(17,19));xs=list(range(32))
    for ax,m in zip(axes.flatten(),QUALITY):
        for reference,marker in [(0.,'o'),(1.,'x')]:
            vals=[next(r['absolute_delta'] for r in deltas if r['sequence']==seq and r['metric']==m and r['QP']==qp and r['reference_gain']==reference) for seq in names]
            ax.plot(xs,vals,marker=marker,linewidth=.7,label=f's0.5 minus s{reference:g}')
        ax.axhline(0,color='gray',linestyle='--');ax.set_ylabel(m+(' (higher better)' if m in ['PSNR','MS_SSIM'] else ' (lower better)'))
        ax.set_xticks(xs);ax.set_xticklabels([str(i+1) for i in xs],fontsize=7);ax.set_xlabel('Video index, fixed lexicographic order');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes.flatten()[-1].axis('off');axes.flatten()[-1].text(0,1,'Index mapping: plots/video_index.csv\nOnly fixed A2000; no checkpoint comparison.\nAbsolute quality deltas at the same video/QP/stream.\nQP9 half measurements are reused, not decoded again.',va='top')
    fig.suptitle(f'ULong development32 videos, fixed A2000, external QP{qp}, s0.5 paired deltas');fig.tight_layout();fig.savefig(folder/f'qp{qp}_per_video_paired_deltas.png',dpi=150);plt.close(fig)

def main():
    env();verify_sources();assert load(ROOT/'audits/smoke.json')['status']=='PASS'
    assert load(ROOT/'audits/source_reuse.json')['status']=='PASS'
    newroot=ROOT/'evaluation/s05';assert len(list((newroot/'points').glob('*.json')))==64
    half=[];endpoint=[typed(r) for r in read_csv(ROOT/'endpoint_metrics.csv')];reuse9=[typed(r) for r in read_csv(ROOT/'metrics_s05_qp9_reused.csv')]
    assert len(endpoint)==192 and len(reuse9)==32
    checks=[];components=set();GTflows={}
    for v in video_list():
        for qp in [0,5,9]:
            b=validate_point(BASE,v,qp);f=validate_point(FULL,v,qp)
            if qp in [0,5]:
                m=validate_point(newroot,v,qp,.5)
                r=dict(m[0]['result']);r.update(inference_gain=.5,measurement_reused=False,independent_decode_performed_this_run=True,
                    checkpoint_SHA256=CHECKPOINT_SHA,model='A2000 half LoRA strength')
                half.append(r)
            else:m=validate_point(QP9_SOURCE/'evaluation/s05',v,qp,.5,QP9_SOURCE)
            compare_streams(v,[b,m,f]);assert b[0]['metric_hashes']==m[0]['metric_hashes']==f[0]['metric_hashes']
            components.add(digest(m[0]['component_hashes']))
            for t in m[2]:GTflows.setdefault((v['sequence'],int(t['to_frame'])),set()).add(t['GT_flow_sha256'])
            result=m[0]['result'];checks.append(dict(domain=v['domain'],sequence=v['sequence'],split='development',external_QP=qp,
                real_bytes=result['real_bytes'],bpp=result['bpp'],bitstream_SHA256=result['bitstream_sha256'],same_video_same_QP_only=True,
                bpp_equal=True,bytes_equal=True,stream_SHA_equal=True,source_RGB_equal=True,frame_indices_equal=True,actual_QPs_equal=True,
                codec_feature_trace_equal=True,metric_weight_hashes_equal=True,GT_flow_hashes_equal=True,half_measurement_reused=qp==9))
    assert len(components)==1 and all(len(x)==1 for x in GTflows.values())
    assert len(half)==64 and all(sum(int(r['QP'])==q for r in half)==32 for q in [0,5])
    allhalf=half+reuse9;combined=endpoint+allhalf
    assert len(allhalf)==96 and len(combined)==288
    assert len({(*paired_key(r),float(r['inference_gain'])) for r in combined})==288
    assert all(math.isfinite(float(r[m])) for r in combined for m in METRICS)
    expected={(v['domain'],v['sequence'],'development',q,g) for v in video_list() for q in [0,5,9] for g in [0.,.5,1.]}
    assert {(*paired_key(r),float(r['inference_gain'])) for r in combined}==expected
    write_csv(ROOT/'metrics_s05_qp05.csv',half);write_csv(ROOT/'metrics_s05_all_qps.csv',allhalf);write_csv(ROOT/'metrics_combined.csv',combined)
    write_csv(ROOT/'stream_identity_checks.csv',checks)
    indexed={(paired_key(r),float(r['inference_gain'])):r for r in combined};summary={};meanrows=[];deltas=[]
    for qp in [0,5,9]:
        means={};changes={};counts={}
        for gain in [0.,.5,1.]:
            rows=[r for r in combined if int(r['QP'])==qp and float(r['inference_gain'])==gain];assert len(rows)==32
            means[str(gain)]={m:sum(float(r[m]) for r in rows)/32 for m in METRICS}
            meanrows.append(dict(dataset='ULong_development_32',external_QP=qp,inference_gain=gain,videos=32,weighting='equal videos within QP',**means[str(gain)]))
        for reference in [0.,1.]:
            label=f's0.5_vs_s{reference:g}';changes[label]={};counts[label]={}
            for m in METRICS:
                cv=means['0.5'][m];rv=means[str(reference)][m];lower=m not in ['PSNR','MS_SSIM'];ds=[]
                changes[label][m]=dict(candidate_mean=cv,reference_mean=rv,absolute_delta=cv-rv,relative_percent=(cv/rv-1)*100 if rv!=0 else None)
                for v in video_list():
                    pair=(v['domain'],v['sequence'],'development',qp);a=float(indexed[pair,.5][m]);b=float(indexed[pair,reference][m]);d=a-b;ds.append(d)
                    deltas.append(dict(domain=v['domain'],sequence=v['sequence'],split='development',QP=qp,candidate_gain=.5,reference_gain=reference,metric=m,
                        candidate=a,reference=b,absolute_delta=d,relative_percent=(a/b-1)*100 if b!=0 else None,
                        direction='rate_identity_only' if m in ['bpp','kbps'] else 'lower_is_better' if lower else 'higher_is_better'))
                counts[label][m]=dict(videos=32,improved=sum(d<0 if lower else d>0 for d in ds),worsened=sum(d>0 if lower else d<0 for d in ds),equal=sum(d==0 for d in ds),
                    direction='rate_identity_only' if m in ['bpp','kbps'] else 'lower_is_better' if lower else 'higher_is_better')
        summary[str(qp)]=dict(means=means,changes=changes,video_direction_counts=counts)
    write_csv(ROOT/'paired_deltas.csv',deltas);write_csv(ROOT/'summary_by_qp.csv',meanrows)
    dump(ROOT/'summary.json',dict(status='COMPLETE',new_formal_points=dict(QP0=32,QP5=32),QP9_half_reused=32,half_total=96,endpoint_reused=192,combined_unique=288,
        checkpoint_SHA256=CHECKPOINT_SHA,training_updates=0,inference_gain=.5,per_QP=summary,overall_QP_pooled_mean_computed=False,
        automatic_gain_or_model_selection=False,relative_change_definition='(mean_candidate/mean_reference-1)*100',weighting='32 equal videos within each external QP',
        metric_directions={m:('rate_identity_only' if m in ['bpp','kbps'] else 'higher_is_better' if m in ['PSNR','MS_SSIM'] else 'lower_is_better') for m in METRICS}))
    for qp in [0,5,9]:plots_for_qp(qp,summary[str(qp)]['means'],deltas)
    write_csv(ROOT/'plots/video_index.csv',[dict(index=i+1,sequence=v['sequence'],domain=v['domain'],split='development') for i,v in enumerate(video_list())])
    plots=[f'plots/qp{q}_{name}.png' for q in [0,5,9] for name in ['mean_metrics_vs_gain','per_video_paired_deltas']]
    assert all((ROOT/n).exists() for n in plots)
    report=['# A2000 half-gain development completion: QP0/5 with audited QP9 reuse','',
        'Local evaluation, tables, plots and integrity: COMPLETE. GitHub publication is separately recorded in publication_status.json.','',
        f'Fixed A_control step2000: SHA256 {CHECKPOINT_SHA}. TCG initialization cumulative6500 remains frozen. No optimizer,training update,loss change,new checkpoint,Router,Bridge modulation or altered LoRA architecture.','',
        'New work:32 development videos × externalQP0/5 × gain0.5 =64 full64frame decodes. Two matching half-gain smoke points counted once. Four endpoint smoke decodes reproduce all per-frame reconstruction/RGB SHAs exactly. Endpoint tolerances are unchanged. QP9 is not decoded again.','',
        'Reused:32 verified QP9 half-gain points,with sources/protocol/model components/metric hashes/frame windows and raw records rechecked.192 original endpoint points are reused. Combined records:288 unique(video,QP,gain) tuples. measurement_reused and source references are explicit.','',
        's0 is B+1500 matched-FP32 with no LoRA contribution,not original GVC-RT. Five A2000 branches remain stage3.blocks.{0..3}.dc.2 and head; rank4,intrinsic alpha4,alpha/r1. inference_gain multiplies FP32 LoRA delta; sum inFP32,then cast to base output dtype. No TCG/Bridge/base-convolution/RGB scaling.','',
        'Each video/QP/gain uses an independent decoder and reset DPB/TCG buffers. Original I-frame,actual internal frameQP schedule,safe dispatch,non-LoRA precision and matched_rgb_fp32_v2 remain unchanged. A single gain is used for all63P frames.','',
        'Spatial aggregation:64frames. Temporal aggregation:63transitions. PSNR:total pixel_SSE/total native pixels,not mean framePSNR. PWC GT-flow and reconstruction-flow errors stay distinct and retain the inherited sampler. Source/RGB/QP/bytes/bpp/SHA/codec trace/metric hashes match within the same video and externalQP; different QPs are not required to share streams.','',
        'Means below weight32videos equally within each QP. Relative percent=(mean_candidate/mean_reference-1)*100. No pooled-QP mean,automatic gain/model selection,finaltest,other checkpoint or new gain. Plots are fixed-QP gain curves,not R-D or BD-rate.','',
        '## Per-QP measured means','',
        '|QP|s|Videos|bpp|kbps|PSNR ↑|MS-SSIM ↑|LPIPS ↓|DISTS ↓|FloLPIPS ↓|GT warp ↓|Recon warp ↓|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in meanrows:report.append('|'+str(r['external_QP'])+'|'+str(r['inference_gain'])+'|32|'+'|'.join(f'{r[m]:.12g}' for m in METRICS)+'|')
    report+=['','## Per-QP differences and direction counts','','```json',json.dumps({q:dict(changes=x['changes'],video_direction_counts=x['video_direction_counts']) for q,x in summary.items()},indent=2),'```','',
        '## Products','',
        'metrics_s05_qp05.csv;metrics_s05_all_qps.csv;endpoint_metrics.csv;metrics_combined.csv;paired_deltas.csv;summary_by_qp.csv;summary.json;stream_identity_checks.csv. Raw64frame/63transition/codec/gain records: evaluation/s05/,reused_qp9/s05/,endpoints/s0/,endpoints/s1/. Audits:sources.json,audits/source_preflight.json,audits/smoke.json,audits/source_reuse.json. Figures:plots/qp{0,5,9}_mean_metrics_vs_gain.png and qp{0,5,9}_per_video_paired_deltas.png;video mapping:plots/video_index.csv.','']
    (ROOT/'REPORT.md').write_text('\n'.join(report))
    verify_sources();assert sha(CHECKPOINT)==CHECKPOINT_SHA
    assert not list(ROOT.rglob('*.pt')) and not list(ROOT.rglob('*.pth')) and not list(ROOT.rglob('*.bin'))
    for p in ROOT.rglob('*'):assert p.resolve().is_relative_to(ROOT),'NEW_OUTPUT_ESCAPED_DEST'
    integrity=dict(status='PASS',new_QP0=32,new_QP5=32,new_formal_unique=64,QP9_half_reused=32,half_total=96,endpoint_reused=192,combined_unique=288,
        checkpoint_SHA256=CHECKPOINT_SHA,training_updates=0,optimizer_created=False,weights_and_TCG_unchanged=True,SOURCE_and_QP9_SOURCE_unchanged=True,
        five_branches_FP32_gain_and_add=True,endpoint_smoke_exact=True,all_64frame_63transition_aggregations=True,all_metrics_finite=True,
        bpp_hash_codec_trace_checks='PASS',metric_weights_equal=True,source_RGB_QP_equal=True,model_components_match_QP9=True,
        cache_identity_and_record_hashes_verified=True,no_final_test=True,no_QP9_new_decode=True,no_other_checkpoints_or_gains=True,
        all_new_outputs_under_DEST=True,no_cross_QP_pooled_mean=True,no_automatic_model_selection=True,plots=plots,time=time.time())
    dump(ROOT/'final_integrity.json',integrity)
    excluded={'artifact_manifest.json','status.json','publication_status.json','publication_remote_receipt.json','publication_allowlist.json','active_processes.json','runner_identity.json','launcher_receipt.json','heartbeat.json'}
    dump(ROOT/'artifact_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
        and p.suffix in ['.py','.json','.csv','.md','.png'] and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','logs','large_artifacts']) and p.name not in excluded]))
    print('QP05_64_NEW_96_HALF_288_COMBINED_INTEGRITY_PASS')

if __name__=='__main__':main()
