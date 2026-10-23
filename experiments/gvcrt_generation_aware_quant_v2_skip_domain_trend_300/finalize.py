"""Validate complete artifacts and generate descriptive, actual-rate comparisons."""
import csv, hashlib, itertools, json, math, shutil, sys, subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
V1=P.parent/'gvcrt_generation_aware_quant_visual_condition_v1_trend_300'
BRANCHES=['A','H00','H10','H01','H11'];VIDEOS=['UVal0','UVal1','ParkScene','Jockey'];METRICS=['LPIPS','DISTS','FloLPIPS','PSNR']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(path,rows):
    assert rows,path
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def interp(rows,x,key):
    rows=sorted(rows,key=lambda r:r['actual_bpp'])
    for a,b in zip(rows,rows[1:]):
        if a['actual_bpp']<=x<=b['actual_bpp']:
            if b['actual_bpp']==a['actual_bpp']:return a[key]
            t=(x-a['actual_bpp'])/(b['actual_bpp']-a['actual_bpp'])
            return a[key]+t*(b[key]-a[key])
    raise ValueError(('no_extrapolation',x,key))

def main():
    from run import run, PYTHON
    run([str(P/'make_previews.py')],7,P/'server_artifacts/preview_decode.log')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image,ImageDraw
    assert json.loads((P/'preflight.json').read_text())['status']=='PASS'
    if not (P/'checkpoint_content_audit.json').exists():
        subprocess.run([PYTHON,str(P/'audit_checkpoints.py')],check=True)
    assert json.loads((P/'checkpoint_content_audit.json').read_text())['status']=='PASS'
    training=[];audits={};indices={};training_provenance=[]
    old_rows=list(csv.DictReader((V1/'training_log.csv').open()))
    for b in BRANCHES:
        if b in ('A','H00'):
            old='A' if b=='A' else 'B'
            rows=[{**r,'branch':b,'training_origin':'reused_v1_'+old} for r in old_rows if r['branch']==old]
            training_provenance.append({'branch':b,'source':str(V1/'training_log.csv'),'source_sha256':sha(V1/'training_log.csv')})
            root=V1/'server_artifacts'/old
        else:
            rows=[{**r,'training_origin':'new_v2'} for r in csv.DictReader((P/f'training_{b}.csv').open())]
            audits[b]=json.loads((P/f'parameter_audit_{b}.json').read_text())
            assert audits[b]['updates']==300 and audits[b]['frozen_parameters_unchanged']
            assert audits[b]['initial']['G']!=audits[b]['final']['G'] and audits[b]['initial']['H']!=audits[b]['final']['H']
            root=P/'server_artifacts'/b
        assert len(rows)==300 and [int(r['step']) for r in rows]==list(range(1,301))
        assert all(int(r['optimizer_internal_step'])==int(r['step']) and float(r['parameter_delta'])>0 for r in rows)
        training+=rows
        indices[b]={str(step):{'path':str(root/f'step_{step}.pt'),'sha256':sha(root/f'step_{step}.pt')} for step in [0,100,300]}
    write(P/'training_log.csv',training)
    summary=[]
    for b in BRANCHES:
        rr=[r for r in training if r['branch']==b]
        last=rr[-1]
        out={'branch':b,'effective_updates':len(rr),'training_origin':last['training_origin']}
        for k in ['loss','LPIPS','L1','estimated_bpp','grad_norm','G_grad_norm','H_grad_norm','G_parameter_delta','H_parameter_delta','H_multiplier_mean','actual_rho_mean','q_lower_clamp_fraction','pass0_skip_fraction','pass1_skip_fraction','pass0_saturation_fraction','pass1_saturation_fraction','estimated_y_bits','estimated_z_bits']:
            if k in last:out['update300_'+k]=last[k]
        for k in ['G_grad_norm','H_grad_norm','G_parameter_delta','H_parameter_delta']:
            if k in last:
                vv=[float(r[k]) for r in rr]
                assert all(math.isfinite(v) and v>0 for v in vv),(b,k)
                out[k+'_minimum']=min(vv);out[k+'_maximum']=max(vv)
        summary.append(out)
    write(P/'training_summary.csv',summary)
    (P/'parameter_change_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (P/'checkpoint_index.json').write_text(json.dumps(indices,indent=2)+'\n')
    (P/'training_provenance.json').write_text(json.dumps(training_provenance,indent=2)+'\n')
    items=[]
    for b in BRANCHES:
        for step in [0,100,300]:
            for source in (VIDEOS if step==300 else VIDEOS[:2]):
                for qp in [2,7]:
                    path=P/'server_artifacts/evaluation'/f'{b}_s{step}_{source}_q{qp}.json'
                    assert path.exists(),path
    for f in sorted((P/'server_artifacts/evaluation').glob('*.json')):
        r=json.loads(f.read_text());assert r['status']=='PASS' and r['independent_decode'] and r['all_64_reference_states_synchronized']
        assert sha(r['stream_path'])==r['stream_sha256']
        assert r['checkpoint_sha256']==indices[r['branch']][str(r['step'])]['sha256']
        assert all(math.isfinite(r[k]) for k in ['actual_bpp',*METRICS])
        assert r['actual_bpp']>0 and r['real_total_bytes']>0
        r['task_file_sha256']=sha(f)
        r['supplemental_qp']=r['external_qp'] not in [2,7]
        assert r['real_total_bytes']==r['payload_bytes']+r['header_bytes']
        items.append(r)
    for branch in BRANCHES:
        for seq in VIDEOS:
            supplemental={r['external_qp'] for r in items if r['branch']==branch and r['sequence']==seq and r['step']==300}-{2,7}
            assert len(supplemental)<=3 and all(0<=q<=9 for q in supplemental)
    write(P/'evaluation.csv',items)
    baseline_parity=[]
    for r in items:
        if r['branch'] not in ('A','H00') or r['measurement']=='reused_v1_measurement':continue
        old='A' if r['branch']=='A' else 'B'
        source=V1/'server_artifacts/evaluation'/f"{old}_s{r['step']}_{r['sequence']}_q{r['external_qp']}.json"
        if not source.exists():continue
        before=json.loads(source.read_text())
        out={'branch':r['branch'],'step':r['step'],'sequence':r['sequence'],'external_qp':r['external_qp'],'v1_task_sha256':sha(source),'real_bytes_delta':r['real_total_bytes']-before['real_total_bytes']}
        assert out['real_bytes_delta']==0
        for k in METRICS:
            out[k+'_delta']=r[k]-before[k]
            assert abs(out[k+'_delta'])<=1e-7
        baseline_parity.append(out)
    write(P/'baseline_remeasurement_parity.csv',baseline_parity)
    initial_parity=[]
    for seq in VIDEOS[:2]:
        for qp in [2,7]:
            initial={r['branch']:r for r in items if r['step']==0 and r['sequence']==seq and r['external_qp']==qp}
            for b in BRANCHES[1:]:
                out={'sequence':seq,'external_qp':qp,'branch':b,'reference':'A','real_total_bytes_delta':initial[b]['real_total_bytes']-initial['A']['real_total_bytes']}
                assert out['real_total_bytes_delta']==0
                for k in METRICS:
                    out[k+'_delta']=initial[b][k]-initial['A'][k]
                    assert abs(out[k+'_delta'])<=1e-7
                initial_parity.append(out)
    write(P/'initial_full_sequence_identity.csv',initial_parity)
    perframe=[];transitions=[];segments=[];rates=[]
    frame_dir=P/'per_frame';frame_dir.mkdir(exist_ok=True)
    for r in items:
        key=f"{r['branch']}_s{r['step']}_{r['sequence']}_q{r['external_qp']}"
        f=P/'server_artifacts/evaluation_frames'/f'{key}.csv'
        t=P/'server_artifacts/evaluation_transitions'/f'{key}.csv'
        rows=list(csv.DictReader(f.open()));trs=list(csv.DictReader(t.open()))
        assert len(rows)==64 and len(trs)==63
        assert sum(int(x['real_bytes']) for x in rows)==r['real_total_bytes']
        assert sum(int(x['payload_bytes']) for x in rows[1:])==r['P_payload_bytes']
        idcols={k:r[k] for k in ['branch','step','sequence','external_qp','measurement']}
        perframe.extend({**idcols,**x} for x in rows);transitions.extend({**idcols,**x} for x in trs)
        # FloLPIPS transitions are assigned to the segment containing their destination frame.
        for label,start,end in [('first15',0,15),('later49',15,64)]:
            part=rows[start:end];n=len(part);pp=[x for x in part if int(x['frame'])>0]
            flow=[float(x['FloLPIPS']) for x in trs if start<=int(x['to_frame'])<end]
            s={**idcols,'segment':label,'first_frame':start,'last_frame':end-1,'frames':n,
               'real_total_bytes':sum(int(x['real_bytes']) for x in part),
               'P_payload_bytes':sum(int(x['payload_bytes']) for x in pp)}
            s['actual_bpp']=s['real_total_bytes']*8/(n*r['height']*r['width'])
            for k in ['LPIPS','DISTS']:s[k]=sum(float(x[k]) for x in part)/n
            s['PSNR']=-10*math.log10(max(sum(float(x['pixel_SSE']) for x in part)/(n*r['height']*r['width']*3),1e-15))
            s['FloLPIPS']=sum(flow)/len(flow)
            if all('estimated_y_bits' in x and x['estimated_y_bits'] for x in pp):
                s['estimated_P_y_bits']=sum(float(x['estimated_y_bits']) for x in pp)
                s['estimated_P_z_bits']=sum(float(x['estimated_z_bits']) for x in pp)
            segments.append(s)
        rates.append({**idcols,'P_frames':'1-63','estimated_y_bits':r['estimated_P_main_latent_bits'],
                      'estimated_z_bits':r['estimated_P_hyper_latent_bits'],'estimated_P_bits':r['estimated_P_total_bits'],
                      'actual_P_payload_bits':r['actual_P_payload_bits'],'estimate_to_actual_ratio':r['estimated_P_total_bits']/r['actual_P_payload_bits'],
                      'actual_total_bytes':r['real_total_bytes'],'header_bytes':r['header_bytes'],
                      'actual_y_z_split':'unavailable_interleaved_RANS'})
    write(frame_dir/'frames.csv',perframe);write(frame_dir/'transitions.csv',transitions)
    write(P/'frame_segment_summaries.csv',segments);write(P/'estimated_vs_actual_P_rate.csv',rates)
    matched=[];interactions=[]
    pairs=[('A',h) for h in BRANCHES[1:]]+list(itertools.combinations(BRANCHES[1:],2))
    for seq in VIDEOS:
        by={b:[r for r in items if r['branch']==b and r['step']==300 and r['sequence']==seq] for b in BRANCHES}
        for a,b in pairs:
            aa,bb=by[a],by[b]
            low=max(min(r['actual_bpp'] for r in aa),min(r['actual_bpp'] for r in bb))
            high=min(max(r['actual_bpp'] for r in aa),max(r['actual_bpp'] for r in bb))
            common={'sequence':seq,'reference':a,'method':b,'delta_definition':'method_minus_reference','overlap_low_bpp':low,'overlap_high_bpp':high}
            if low>=high:
                matched.append({**common,'status':'NO_OVERLAP'});continue
            for name,point in [('low',low),('midpoint',(low+high)/2),('high',high)]:
                row={**common,'status':'OVERLAP','position':name,'actual_bpp':point}
                for k in METRICS:
                    row['reference_'+k]=interp(aa,point,k);row['method_'+k]=interp(bb,point,k)
                    row['delta_'+k]=row['method_'+k]-row['reference_'+k]
                matched.append(row)
        hs=BRANCHES[1:]
        low=max(min(r['actual_bpp'] for r in by[b]) for b in hs)
        high=min(max(r['actual_bpp'] for r in by[b]) for b in hs)
        inter={'sequence':seq,'common_low_bpp':low,'common_high_bpp':high,'definition':'H11-H10-H01+H00_at_same_actual_bpp','status':'OVERLAP' if low<high else 'NO_COMMON_OVERLAP'}
        if low<high:
            x=(low+high)/2;inter['actual_bpp']=x
            for k in METRICS:inter[k]=interp(by['H11'],x,k)-interp(by['H10'],x,k)-interp(by['H01'],x,k)+interp(by['H00'],x,k)
        interactions.append(inter)
    write(P/'matched_rate_comparisons.csv',matched);write(P/'interaction_comparisons.csv',interactions)
    equal=[]
    for seq in VIDEOS:
        for qp in [2,7]:
            base={r['branch']:r for r in items if r['step']==300 and r['sequence']==seq and r['external_qp']==qp}
            for a,b in pairs:
                row={'sequence':seq,'external_qp':qp,'reference':a,'method':b,'comparison':'same_QP_not_matched_rate'}
                for k in ['actual_bpp',*METRICS]:row['delta_'+k]=base[b][k]-base[a][k]
                equal.append(row)
    write(P/'same_qp_differences.csv',equal)
    figures=P/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for b in BRANCHES:
        rr=[r for r in training if r['branch']==b]
        xx=list(range(1,301))
        for ax,key in zip(axes.flat,['loss','estimated_bpp','actual_rho_mean','q_lower_clamp_fraction']):
            if key not in rr[0]:continue
            yy=[float(r[key]) for r in rr]
            avg=[sum(yy[max(0,i-9):i+1])/min(i+1,10) for i in range(300)]
            ax.plot(xx,avg,label=b);ax.set(xlabel='Effective update',ylabel=key);ax.grid(alpha=.25)
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('Training only; trailing 10-update means');fig.tight_layout();fig.savefig(figures/'training_diagnostics.png',dpi=160);plt.close(fig)
    for metric in METRICS:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,seq in zip(axes.flat,VIDEOS):
            for b in BRANCHES:
                points=sorted([r for r in items if r['step']==300 and r['sequence']==seq and r['branch']==b],key=lambda r:r['actual_bpp'])
                ax.plot([r['actual_bpp'] for r in points],[r[metric] for r in points],marker='o',label=b)
                for r in points:ax.annotate(str(r['external_qp']),(r['actual_bpp'],r[metric]),fontsize=7)
            ax.set(title=seq,xlabel='Actual total bpp',ylabel=metric);ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('Step 300; line segments are descriptive, no extrapolation');fig.tight_layout();fig.savefig(figures/f'RD_{metric}.png',dpi=160);plt.close(fig)
    for seq in VIDEOS:
        canvas=Image.new('RGB',(256*6,286),'white');draw=ImageDraw.Draw(canvas)
        names=[('GT',P/'previews'/f'{seq}_gt_frame15.png')]+[(b,P/'previews'/f'{b}_s300_{seq}_q2_frame15.png') for b in BRANCHES]
        for i,(label,path) in enumerate(names):
            if not path.exists() and label in ['A','H00']:
                old='A' if label=='A' else 'B'
                alternatives=[V1/'previews'/f'{old}_s300_{seq}_q2_frame15.png',V1/'figures/previews'/f'{old}_s300_{seq}_q2_frame15.png']
                path=next((x for x in alternatives if x.exists()),path)
            assert path.exists(),path
            image=Image.open(path).convert('RGB');assert image.size==(256,256)
            canvas.paste(image,(i*256,30));draw.text((i*256+5,8),label,fill='black')
        canvas.save(figures/f'fixed_preview_{seq}_frame15_q2.png')
    # All executable dependencies of this local driver are recorded, including server-local v1 references.
    repo=P.parents[1]
    deps=[repo/'src/models/common_model.py',repo/'src/models/video_model_gvcrt.py',repo/'src/models/entropy_models.py',repo/'src/layers/cuda_inference.py',repo/'experiments/gvcrt_neural_wrapper_v2_joint/core.py',repo/'expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py',repo/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py',repo/'experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/fullqp_common.py',Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback/flolpips_compat.py')]
    deps+=list(P.glob('*.py'))+[P/'effective_config.json',P/'data_manifest.json',P/'train_plan.json',V1/'train.py',V1/'runtime.py',V1/'evaluate.py',V1/'effective_config.json']
    (P/'dependency_manifest.json').write_text(json.dumps({str(f):sha(f) for f in deps},indent=2)+'\n')
    reused=sum(r['measurement']=='reused_v1_measurement' for r in items)
    lines=['# Measured v2 skip/domain trend results','',
           'H10, H01 and H11 completed exactly 300 effective optimizer updates each from the common G_basic_100 and v1 B initial H. A/H00 reuse v1 A/B training; neither was retrained. No V or new probability parameters were added.','',
           f"Preflight: PASS. All {len(items)} evaluation tasks independently decoded persisted streams and checked all 64 causal references; {reused} are explicitly reused v1 measurements and {len(items)-reused} are new measurements. New tasks additionally check every P latent/qmap exactly. The declared tolerances and actual maximum errors are in preflight.json. Frozen-parameter hashes are unchanged in all three new groups.",'',
           '## Step 300 fixed-QP measurements','',
           'These are measured at the same external QP, which does not imply the same actual bitrate. Differences and rate/quality conclusions below use actual-bpp overlap.','',
           '| Video | Group | QP | Actual bpp | LPIPS | DISTS | FloLPIPS | PSNR dB |','|---|---|---:|---:|---:|---:|---:|---:|']
    for seq in VIDEOS:
        for b in BRANCHES:
            for r in sorted([r for r in items if r['sequence']==seq and r['branch']==b and r['step']==300 and not r['supplemental_qp']],key=lambda r:r['external_qp']):
                lines.append(f"| {seq} | {b} | {r['external_qp']} | {r['actual_bpp']:.6f} | {r['LPIPS']:.5f} | {r['DISTS']:.5f} | {r['FloLPIPS']:.5f} | {r['PSNR']:.3f} |")
    lines+=['','## Matched actual-rate differences','',
            'The following descriptive piecewise-linear comparisons evaluate the midpoint of the measured overlap in each video. Delta is method minus reference: negative LPIPS/DISTS/FloLPIPS and positive PSNR favor the method. Full endpoints and intervals appear in matched_rate_comparisons.csv. No extrapolation or formal BD-rate is used. Supplemental QP choices and the rate-only evidence are in supplement_selection.json.','',
            '| Video | Method − reference | Overlap bpp | Midpoint bpp | ΔLPIPS | ΔDISTS | ΔFloLPIPS | ΔPSNR dB |','|---|---|---|---:|---:|---:|---:|---:|']
    mids=[r for r in matched if r.get('position')=='midpoint' or r['status']=='NO_OVERLAP']
    for r in mids:
        if r['status']=='NO_OVERLAP':lines.append(f"| {r['sequence']} | {r['method']} − {r['reference']} | no overlap | — | — | — | — | — |")
        else:lines.append(f"| {r['sequence']} | {r['method']} − {r['reference']} | {r['overlap_low_bpp']:.6f}–{r['overlap_high_bpp']:.6f} | {r['actual_bpp']:.6f} | {r['delta_LPIPS']:+.5f} | {r['delta_DISTS']:+.5f} | {r['delta_FloLPIPS']:+.5f} | {r['delta_PSNR']:+.3f} |")
    lines+=['','## Declared comparisons and interaction','']
    for ref,method,label in [('H00','H10','Skipped-symbol gradient treatment'),('H00','H01','Coordinate reparameterization'),('H10','H11','Adding coordinates to the skip treatment'),('H01','H11','Adding skip treatment to coordinates')]:
        rs=[r for r in mids if r['reference']==ref and r['method']==method and r['status']=='OVERLAP']
        lower=sum(r['delta_LPIPS']<0 for r in rs)
        lines.append(f"- {label}, {method}−{ref}: lower interpolated LPIPS in {lower}/{len(rs)} overlapping videos; see the per-video deltas above. These observations do not identify a causal mechanism beyond the controlled intervention.")
    for b in BRANCHES[1:]:
        rs=[r for r in mids if r['reference']=='A' and r['method']==b and r['status']=='OVERLAP']
        lines.append(f"- {b} versus A: lower interpolated LPIPS in {sum(r['delta_LPIPS']<0 for r in rs)}/{len(rs)} overlapping videos. DISTS, FloLPIPS and PSNR are reported separately rather than treating one metric as universal quality.")
    lines+=['','The descriptive interaction contrast uses H11−H10−H01+H00 at the midpoint of the common four-H overlap. A nonzero value is only an observed contrast in this small run; it is not a statistical interaction test.','',
            '| Video | Common actual bpp | LPIPS contrast | DISTS contrast | FloLPIPS contrast | PSNR contrast |','|---|---:|---:|---:|---:|---:|']
    for r in interactions:
        if r['status']!='OVERLAP':lines.append(f"| {r['sequence']} | no common overlap | — | — | — | — |")
        else:lines.append(f"| {r['sequence']} | {r['actual_bpp']:.6f} | {r['LPIPS']:+.5f} | {r['DISTS']:+.5f} | {r['FloLPIPS']:+.5f} | {r['PSNR']:+.3f} |")
    valid_interactions=[r for r in interactions if r['status']=='OVERLAP']
    if valid_interactions:
        narrow=min(valid_interactions,key=lambda r:r['common_high_bpp']-r['common_low_bpp'])
        lines+=['',f"The smallest common four-H interval is {narrow['sequence']}: {narrow['common_low_bpp']:.9f}–{narrow['common_high_bpp']:.9f} bpp (width {narrow['common_high_bpp']-narrow['common_low_bpp']:.9f}). Its interaction comparison describes only this very narrow region."]
    lines+=['','## Audits, plots and limits','',
            '- CPU skip check: original dy_hat/dq=−1.5, corrected=.5, local finite difference≈.5 for the fixed skipped state. Active-branch STE and mean/dequantization Jacobians were checked separately.',
            '- H10 and v1 H00 retain the same hard forward under identical nonidentity weights. Actual-rho, q clamp, skip/non-skip, both passes and signed saturation endpoints are covered in synthetic checks.',
            '- A dense-saturation synthetic stream triggered a native C++ RANS segmentation fault during preflight development. Final boundary coverage uses sparse saturation in both passes; the native coder and symbol range were not modified. Formal training was gated until the final complete preflight passed. Dense pathological-stream robustness is not established.',
            '- Training logs include multiplier/rho moments and quantiles, clamp/skip/saturation fractions, G/H gradient norms and update norms, and explicitly estimated y/z bits. Baseline fields unavailable in v1 are left empty rather than reconstructed.',
            '- estimated_vs_actual_P_rate.csv compares analytical estimated y/z bits with actual P1–63 payload only. Exact total/header bytes are separate. No exact actual y/z split is claimed for interleaved RANS.',
            '- per_frame/frames.csv and transitions.csv retain all frames and transitions. frame_segment_summaries.csv separates frames0–14 and15–63; transition assignment uses its destination frame. Fixed images use frame15 and the predeclared region for every method/video at QP2.',
            '- The inherited four-entry QP offset description is retained in effective_config.json. effective_runtime_audit.json records the full unchanged v1 eight-frame schedule used by execution.',
            '- This is a 300-update trend experiment. Sparse interpolation is descriptive and may conceal curve shape. No convergence, significance or strict unseen-data generalization is claimed; historical checkpoint exposure is not fully established. No untested mechanism is asserted and no next-round training is launched.','']
    lines.append('![Training diagnostics](figures/training_diagnostics.png)\n')
    for k in METRICS:lines.append(f'![Actual rate versus {k}](figures/RD_{k}.png)\n')
    for seq in VIDEOS:lines.append(f'![Fixed {seq} preview](figures/fixed_preview_{seq}_frame15_q2.png)\n')
    (P/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
    (P/'completion.json').write_text(json.dumps({'status':'COMPLETE','updates':{b:300 for b in BRANCHES},'A_H00_training_reused':True,'preflight':'PASS','evaluation_tasks':len(items),'reused_evaluation_tasks':reused,'new_evaluation_tasks':len(items)-reused,'all_64_reference_checks':True},indent=2)+'\n')
    print('FINALIZED',len(items),flush=True)

if __name__=='__main__':main()
