"""Paired actual measurements, independent aggregation checks and factual report."""
from common_s2 import *
from task import tasks,point_path,verify

METRICS=['PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC','MCResidual_RGB_RAFT_pixelcenter_v1_PtoP','MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP']

def main():
    assert verify();rows=[];deltas=[];streams=[];source_records=[];frame_count=transition_count=0
    for task in tasks():
        profile=task['profile'];q=task['QP'];points={model:load(point_path(profile,q,model)) for model in ['Original','B']}
        reference=points['Original']['result']
        for model,point in points.items():
            assert point['status']=='PASS' and point['frozen_components_unchanged'] and point['codec_features_match_original']
            r=point['result'];assert r['frames']==128 and r['transitions']==127 and r['QP']==q and r['sequence']==profile['sequence']
            assert point['identity']['implementation_SHA256']==sha(ROOT/'source_manifest.json')
            for key in ['real_bytes','bpp','bitstream_SHA256','source_RGB_SHA256','codec_trace_SHA256']:assert r[key]==reference[key]
            assert point['actual_qps']==points['Original']['actual_qps'] and point['bytes_consumed']==r['real_bytes']
            for path,h in point['record_SHA256'].items():assert sha(ROOT/path)==h
            base=point_path(profile,q,model).parents[1]
            with (base/'frames'/f'{profile["sequence"]}_qp{q}.csv').open() as f:frames=list(csv.DictReader(f))
            with (base/'transitions'/f'{profile["sequence"]}_qp{q}.csv').open() as f:temporal=list(csv.DictReader(f))
            assert len(frames)==128 and len(temporal)==127
            assert [int(f['source_frame']) for f in frames]==profile['row']['source_frame_indices']
            assert sum(int(f['real_bits']) for f in frames)==r['real_bytes']*8
            mse=sum(float(f['pixel_SSE']) for f in frames)/(128*1920*1080*3)
            assert abs(r['PSNR']+10*math.log10(max(mse,1e-15)))<1e-9
            for key in ['LPIPS','DISTS','MS_SSIM']:assert abs(r[key]-sum(float(f[key]) for f in frames)/128)<1e-9
            for key in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:
                assert abs(r[key]-sum(float(t[key]) for t in temporal)/127)<1e-9
            assert all(math.isfinite(r[k]) for k in METRICS if r[k] is not None)
            clip=point['lossless_video'];assert sha(clip['path'])==clip['SHA256']
            raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',clip['path'],'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
            assert hashlib.sha256(raw).hexdigest()==frames[0]['viewing_RGB8_SHA256'],'VIEWING_RGB_COLOR_OR_IDENTITY_MISMATCH'
            rows.append(dict(alias=profile['alias'],model_label='Original GVC-RT' if model=='Original' else 'B300 TCG+LoRA+Bridge_last s0.5',measurement_reused=False,**r))
            source_records.append(dict(model=model,QP=q,sequence=profile['sequence'],path=str(point_path(profile,q,model)),SHA256=sha(point_path(profile,q,model))))
            frame_count+=128;transition_count+=127
        d=dict(alias=profile['alias'],sequence=profile['sequence'],QP=q,reference='Original GVC-RT',candidate='B300',real_bytes=reference['real_bytes'],bpp=reference['bpp'])
        for key in METRICS:
            a=points['B']['result'][key];b=reference[key]
            d[key+'_absolute_delta']=None if a is None or b is None else a-b
            d[key+'_relative_percent']=None if a is None or b in (None,0) else 100*(a/b-1)
        deltas.append(d);streams.append(dict(sequence=profile['sequence'],QP=q,real_bytes=reference['real_bytes'],bpp=reference['bpp'],stream_SHA256=reference['bitstream_SHA256'],actual_QPs_identical=True,codec_trace_identical=True,source_RGB_identical=True,bytes_and_bpp_identical=True))
    assert len(rows)==8 and len({(r['model'],r['sequence'],r['QP']) for r in rows})==8
    write_csv(ROOT/'metrics.csv',rows);write_csv(ROOT/'paired_deltas.csv',deltas);write_csv(ROOT/'stream_identity_checks.csv',streams)
    comparisons=load(ROOT/'comparison_manifest.json');assert comparisons['status']=='PASS' and len(comparisons['comparisons'])==4
    for c in comparisons['comparisons']:
        for f in c['files']:assert sha(f['path'])==f['SHA256']
    dump(ROOT/'final_integrity.json',dict(status='PASS',complete=True,measurement_points=8,models=2,videos=2,QP_values=[0,5],frames_per_point=128,
        total_scored_frames=frame_count,total_scored_transitions=transition_count,fresh_original_streams=4,canonical_shared_stream_identity='PASS',codec_trace_identity='PASS',
        no_training_or_optimizer=True,checkpoint_SHA256=sha(B_FINAL),checkpoint_matches_fixed_B300=sha(B_FINAL)==B_FINAL_SHA,
        source_points=source_records,completed_flicker_source_protected=True,ROI_frozen_before_model_metrics=True,full_metric_aggregates_recomputed=True,time=time.time()))
    dump(ROOT/'baseline_provenance.json',dict(Original_GVC_RT=dict(original_commit=ORIGINAL_COMMIT,weights={n:sha(PROJECT/'checkpoints'/n) for n in ['GVC-RT_I.pt','GVC-RT_P.pt']},pilot=False,TCG=False,LoRA=False),
        B300=dict(checkpoint=str(B_FINAL),SHA256=B_FINAL_SHA,initial_A2000_SHA256=INIT_SHA,initial_TCG_B1500_SHA256=BASELINE_SHA,pilot_receiver_SHA256=PILOT_SHA,
            overlay_components=['TCG','LoRA','Bridge_mlp3'],new_flicker_updates=300,TCG_cumulative_updates=6800,LoRA_cumulative_updates=2300,inference_gain=.5),training_updates_in_this_task=0))
    lines=['# Person/dog and Cactus: fixed B300 vs independent Original','',f'Snapshot UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}','',
        'User-requested post-hoc comparison of existing held-out/historical sources. Fixed B300 checkpoint; no training, checkpoint selection or loss changes.128 native consecutive frames, real I0 plus127 P frames, external QP0/5. Person/dog uses source128..255 at30000/1001fps; Cactus uses0..127 at50fps. These128-frame results are separate from previous64-frame tables.','',
        'Original: pinned b2a0895 original I/P weights without pilot,TCG orLoRA. B300: original codec plus pilot receiver,B+1500 TCG,A2000 LoRA,and the completed300-step branchB TCG/LoRA/Bridge_mlp3 overlay. No Router or history alignment. Inference gain0.5. Codec reference trajectory remains the verified original trajectory, matching the completed pilot protocol.','',
        'RGB scoring: float before clamp and normalization, then native crop. Spatial metrics use128frames; temporal metrics127transitions. PSNR comes from total pixel SSE. Old PWC samplers unchanged; new MCResidual metrics retain their distinct RAFT pixel-center version.','',
        '|Video|QP|Model|bpp|kbps|PSNR ↑|MS-SSIM ↑|LPIPS ↓|DISTS ↓|FloLPIPS ↓|','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['alias']+'|'+str(r['QP'])+'|'+r['model_label']+'|'+'|'.join(f'{r[k]:.6f}' for k in ['bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS'])+'|')
    lines+=['','Both old PWC warping errors and new RGB/Sobel temporal residuals, I-to-P and P-to-P detail, real stream hashes and paired absolute/relative deltas are retained in metrics.csv,paired_deltas.csv and the per-frame/per-transition records. No cross-video/dataset pooling or BD-rate estimate.','',
        'Full source geometry1920x1080. The overview uses640x360 panels. Fixed native640x640 crops use no resizing. Additional320x320 crops use explicitly labeled2x nearest-neighbor magnification. ROI coordinates were defined from GT-only first/middle/last views before model metrics. All models use identical windows/crops within each video/QP.','',
        'FFV1 files retain lossless RGB8 viewing exports. H264 CRF12 MP4 files are viewing previews; neural metrics are computed on original float32 decoder outputs. Codec stream bytes and kbps in filenames do not refer to MP4 container size. Large videos/cache/weights remain local.','',
        'Integrity: PASS.8/8points,4 freshly encoded and independently verified canonical RANS streams,1024scoredframes,1016transitions. B300 independently decoded each matching stream; measured bpp/bytes/stream hashes/codec traces identical for each paired video/QP. No optimizer or weight updates.','',
        'See comparison_manifest.json for exact full-native/overview/crop/zoom viewing paths; rois.json for GT-defined coordinates; sources.json for protected source hashes. Publication is separately recorded in publication_status.json.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    # Actual two-QP points only; each figure is for a single video.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (ROOT/'plots').mkdir(exist_ok=True)
    for profile in load(ROOT/'profiles.json'):
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,key in zip(axes.flat,['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP']):
            for model in ['Original','B']:
                rr=sorted([r for r in rows if r['sequence']==profile['sequence'] and r['model']==model],key=lambda r:r['bpp'])
                ax.plot([r['bpp'] for r in rr],[r[key] for r in rr],marker='o',label='Original GVC-RT' if model=='Original' else 'B300 s0.5')
            ax.set(xlabel='Measured bpp',ylabel=key+(' ↑' if key in ['PSNR','MS_SSIM'] else ' ↓'))
        axes.flat[0].legend();fig.suptitle(profile['alias']+'128frames; measured QP0/5 only');fig.tight_layout();fig.savefig(ROOT/'plots'/f'{profile["alias"]}_measured_points.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
