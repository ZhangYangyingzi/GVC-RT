"""Raw point tables and dataset-specific plots; no experimental interpretation."""
from common_s2 import *
from collections import defaultdict
METRIC_DIRECTIONS={'PSNR':'↑','LPIPS_Alex':'↓','LPIPS_VGG':'↓','DISTS':'↓','FloLPIPS':'↓','Sobel_MCResidual_PtoP':'↓'}
FINAL_MODELS=['Original','B300_original_reference','B300_noLoRA_step0','G0_selected','G1_selected']
LABELS.update(G0_selected='G0 teacher spatial selected',G1_selected='G1 teacher temporal selected')

def all_points(phase):
    return [load(p) for p in sorted((ROOT/'evaluation'/phase).glob('*/step_*/points/*.json'))]

def history():
    rows=[]
    for g in GROUPS:
        for p in sorted((ROOT/g/'training/receipts').glob('*.json')):
            a=load(p);r={k:v for k,v in a.items() if not isinstance(v,(dict,list))}
            windows=[window for rank in a['ranks'] for window in rank['records']]
            for k in ['LPIPS_VGG','DISTS','RGB_L1','cosine','RAFT_motion','RGB_MCResidual','Sobel_MCResidual','latent_temporal']:
                means=[]
                for window in windows:
                    values=[x[k] for x in window['trace'] if x[k] is not None]
                    if values:means.append(sum(values)/len(values))
                r[k]=sum(means)/len(means) if means else None
            coeff=load(ROOT/'gradient_calibration.json')
            reconstructed=r['LPIPS_VGG']+.5*r['DISTS']+.1383589248954108*r['RGB_L1']+.2*r['cosine']+.05*r['RAFT_motion']
            if r['RGB_MCResidual'] is not None:reconstructed+=coeff['lambda_out']*(r['RGB_MCResidual']+.5*r['Sobel_MCResidual'])
            if g==GROUPS[1] and r['latent_temporal'] is not None:reconstructed+=coeff['lambda_latent']*r['latent_temporal']
            r['objective_from_components_abs_difference']=abs(reconstructed-r['loss']);assert r['objective_from_components_abs_difference']<1e-5
            r['component_aggregation']='valid frames per window, then equal nonempty windows'
            r['sample_keys']=';'.join(a['sample_keys']);rows.append(r)
    if rows:write_csv(ROOT/'training_history.csv',rows)
    return rows

def tables():
    h=history();dev=all_points('development');final=all_points('final');dv=[a['result'] for a in dev];fv=[]
    for a in final:
        transitions_path=next(path for path in a['record_SHA256'] if '/transitions/' in path)
        transitions=list(csv.DictReader((ROOT/transitions_path).open()))
        row=dict(a['result']);row.update(GTflow_trace_SHA256=digest([t['GT_backward_SHA256'] for t in transitions]),GTvalid_trace_SHA256=digest([t['GT_valid_SHA256'] for t in transitions]));fv.append(row)
    if dv:write_csv(ROOT/'development_metrics.csv',dv)
    if fv:write_csv(ROOT/'metrics.csv',fv)
    matched=[]
    for step in sorted({r['step'] for r in dv if r['model'] in GROUPS}):
        look={(r['model'],r['sequence'],r['QP']):r for r in dv if r['step']==step}
        for seq,q in sorted({(r['sequence'],r['QP']) for r in dv if r['step']==step}):
            a=look.get((GROUPS[0],seq,q));b=look.get((GROUPS[1],seq,q))
            if a and b:matched.append(dict(step=step,sequence=seq,QP=q,**{k+'_G1_minus_G0':b[k]-a[k] if b[k] is not None and a[k] is not None else None for k in METRIC_DIRECTIONS}))
    if matched:write_csv(ROOT/'development_G1_minus_G0_common_steps.csv',matched)
    grouped=defaultdict(list)
    for r in fv:grouped[(r['dataset'],r['model'],r['QP'])].append(r)
    agg=[]
    numeric=['bpp',*METRIC_DIRECTIONS,'RGB_MCResidual_PtoP','RGB_MCResidual_ItoP','Sobel_MCResidual_ItoP']
    for (ds,m,q),items in sorted(grouped.items()):
        out=dict(dataset=ds,model=m,QP=q,videos=len(items),aggregation='equal video weight within dataset/model/QP')
        for k in numeric:
            valid=[r[k] for r in items if r[k] is not None];out[k]=sum(valid)/len(valid) if valid else None;out[k+'_available_videos']=len(valid)
        agg.append(out)
    if agg:write_csv(ROOT/'rd_points_by_dataset.csv',agg)
    deltas=[];ids=[];look={(r['dataset'],r['sequence'],r['QP'],r['model']):r for r in fv}
    for ds,seq,q in sorted({(r['dataset'],r['sequence'],r['QP']) for r in fv}):
        rows={m:look.get((ds,seq,q,m)) for m in FINAL_MODELS}
        if not all(rows.values()):continue
        ref=rows['Original'];same=all(all(r[k]==ref[k] for k in ['real_bytes','bpp','stream_SHA256','codec_trace_SHA256','source_SHA256','source_RGB_SHA256','GTflow_trace_SHA256','GTvalid_trace_SHA256']) for r in rows.values())
        ids.append(dict(dataset=ds,sequence=seq,QP=q,status='PASS' if same else 'FAILED',real_bytes=ref['real_bytes'],bpp=ref['bpp'],stream_SHA256=ref['stream_SHA256'],codec_trace_SHA256=ref['codec_trace_SHA256'],GTflow_trace_SHA256=ref['GTflow_trace_SHA256'],GTvalid_trace_SHA256=ref['GTvalid_trace_SHA256'],source_RGB_SHA256=ref['source_RGB_SHA256'],models=len(rows)))
        for lhs,rhs in [('G1_selected','G0_selected'),('G0_selected','B300_noLoRA_step0'),('G1_selected','B300_noLoRA_step0'),('G0_selected','B300_original_reference'),('G1_selected','B300_original_reference')]:
            a,b=rows[lhs],rows[rhs];deltas.append(dict(dataset=ds,sequence=seq,QP=q,lhs=lhs,rhs=rhs,**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in METRIC_DIRECTIONS}))
    if ids:write_csv(ROOT/'stream_identity_checks.csv',ids)
    if deltas:
        write_csv(ROOT/'paired_deltas_by_video.csv',deltas);gp=defaultdict(list)
        for r in deltas:gp[(r['dataset'],r['QP'],r['lhs'],r['rhs'])].append(r)
        out=[]
        for (ds,q,lhs,rhs),r in sorted(gp.items()):
            v=dict(dataset=ds,QP=q,lhs=lhs,rhs=rhs,videos=len(r))
            for k in METRIC_DIRECTIONS:
                z=[a[k] for a in r if a[k] is not None];v[k]=sum(z)/len(z) if z else None
            out.append(v)
        write_csv(ROOT/'paired_deltas_by_dataset_qp.csv',out)
    return dict(history=h,dev=dv,final=fv,aggregate=agg,identity=ids)

def plots(data):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.figsize':(8,5),'savefig.dpi':130})
    styles=['o-','s--','^-.','D:','v-'];colors=['#111111','#7570b3','#e6ab02','#1b9e77','#d95f02'];outputs=[]
    def chart(rows,base,title):
        base.mkdir(parents=True,exist_ok=True)
        for metric,arrow in METRIC_DIRECTIONS.items():
            fig,ax=plt.subplots();seen={};overlap=[]
            for i,m in enumerate(FINAL_MODELS):
                r=sorted([a for a in rows if a['model']==m],key=lambda a:a['QP']);assert len(r)==3
                x=[a['bpp'] for a in r];y=[a[metric] for a in r]
                if any(v is None for v in y):raise RuntimeError(('MISSING_PLOT_METRIC',title,m,metric))
                xy=tuple(zip(x,y));label=LABELS[m]
                if xy in seen:overlap.append(f'{label} overlaps {seen[xy]}')
                seen[xy]=label;ax.plot(x,y,styles[i],color=colors[i],label=label,markersize=4,alpha=.9)
                for a in r:ax.annotate(f'QP{a["QP"]}',(a['bpp'],a[metric]),xytext=(3,3+i*3),textcoords='offset points',fontsize=6)
            ax.set_xlabel('Actual RANS bpp (I frame and stream headers included)');ax.set_ylabel(f'{metric} {arrow}');ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=7)
            note='Three measured QP points only; no extrapolation or BD-rate.'
            if overlap:note+='\n'+'; '.join(overlap)
            fig.text(.01,.01,note,fontsize=6);fig.tight_layout(rect=(0,.055,1,1));p=base/(metric+'.png');fig.savefig(p);plt.close(fig);outputs.append(dict(path=str(p),SHA256=sha(p)))
    for ds in sorted({r['dataset'] for r in data['aggregate']}):
        title=ds.replace('_',' ')
        if ds=='MCL_JCV_30':title+=' (historically used test set)'
        chart([r for r in data['aggregate'] if r['dataset']==ds],ROOT/'plots'/ds,title+'; equal video weights')
    for ds,seq in sorted({(r['dataset'],r['sequence']) for r in data['final']}):chart([r for r in data['final'] if (r['dataset'],r['sequence'])==(ds,seq)],ROOT/'plots/per_video'/ds/seq,f'{ds}: {seq}')
    for metric in ['LPIPS_VGG','DISTS','Sobel_MCResidual_PtoP']:
        fig,ax=plt.subplots()
        for g in GROUPS:
            for q in [0,5,9]:
                rows=[r for r in data['dev'] if r['model']==g and r['QP']==q];steps=sorted({r['step'] for r in rows});vals=[]
                for s in steps:
                    z=[r[metric] for r in rows if r['step']==s];assert len(z)==32;vals.append(sum(z)/32)
                ax.plot(steps,vals,'o-',label=f'{LABELS[g]}, QP{q}')
        ax.set_xlabel('New optimizer updates');ax.set_ylabel(metric+' ↓');ax.set_title('ULong development subset: 32 videos');ax.legend(fontsize=6);ax.grid(alpha=.25);fig.tight_layout();p=ROOT/'plots'/f'development_steps_{metric}.png';fig.savefig(p);plt.close(fig);outputs.append(dict(path=str(p),SHA256=sha(p)))
    dump(ROOT/'plots/manifest.json',dict(status='PASS',plots=outputs,raw_scores=True,no_BD_rate=True,no_cross_dataset_curves=True))

def report(data,complete=False):
    init=load(ROOT/'initialization_audit.json');sel=load(ROOT/'selection.json') if (ROOT/'selection.json').exists() else None
    text=['# Teacher temporal noLoRA A/B execution record','',f'Status: {"NUMERICAL_RESULTS_RECORDED; integrity and publication statuses are in their separate JSON records" if complete else "RUNNING"}.', '', 'Both groups inherit the B300 Bridge and TCG, which were previously trained with LoRA. Five LoRA wrappers were physically removed without merging their increments. The decoder backbone is frozen. G0 and G1 differ only by the latent temporal objective; Teacher is training-only.','',f'Common step0 SHA256: `{init["common_step0_SHA256"]}`.',f'B300 SHA256: `{B300_SHA}`.',f'Calibration: lambda_out={load(ROOT/"gradient_calibration.json")["lambda_out"]}; lambda_latent={load(ROOT/"gradient_calibration.json")["lambda_latent"]}.','', 'Training: FP32 Bridge/TCG and frozen decoder computation, fresh AdamW; global batch four, two ranks each sequentially processing two windows. GPU0/1: G0; GPU2/3: G1. Independent Original codec trajectory and real RANS rate, with no candidate image fed back to codec. Deployment follows the uniform inherited FP16 backbone/Bridge and FP32 TCG policy.','', 'Training LPIPS is VGG only. Evaluation records Alex and VGG separately, RGB [0,1], LPIPS normalize=True, v0.1 weights. Full numerical definitions and weight identities are in protocol.json, quality.py and per-point receipts. Temporal missing regions are excluded using valid denominators. No margin or extra detail loss is implemented.','', 'Teacher is the channel-compatible audited pretrain262144 EMA; its identity is not asserted to be the exact historical author checkpoint. Channel compatibility is not a claim of temporal stability.','', '| Group | Real added updates | Stop reason | Selected step | Qualification | S |','|---|---:|---|---:|---|---:|']
    for g in GROUPS:
        p=ROOT/g/'training/progress.json';n=load(p)['committed_updates'] if p.exists() else 0;choice=sel['groups'][g] if sel else {};text.append(f'| {g} | {n} | {sel.get("stop_reason") if sel else "running"} | {choice.get("step", "pending")} | {choice.get("qualification", "pending")} | {choice.get("S", "pending")} |')
    text+=['','Selection uses development quality only. Missing metrics and engineering faults do not count as non-improvements. All same-step G1−G0 measurements are in development_G1_minus_G0_common_steps.csv. Final checkpoint identities are frozen in selection.json before final testing.','','Final coverage:','', '| Dataset / split | Model | Completed | Expected |','|---|---|---:|---:|']
    for ds,n in load(ROOT/'protocol.json')['final']['datasets'].items():
        for m in FINAL_MODELS:text.append(f'| {ds} | {LABELS[m]} | {sum(r["dataset"]==ds and r["model"]==m for r in data["final"])} | {n*3} |')
    text+=['','Aggregation: arithmetic mean across video-level measurements within each dataset/split/QP; each video has equal weight. Each per-video PSNR uses pixel SSE across 64 frames; spatial metrics use all 64 frames. I→P and P→P temporal measurements are separate. The main selection temporal score uses P→P Sobel MCResidual. Figures connect three actual RANS bpp points only. Dataset splits are plotted separately; UVG contains only the two historical HoneyBee/Jockey sequences. ULong historical six and new holdout 32 are separate. MCL-JCV is a historically used test collection. No BD-rate is reported.','','Original source: independently restored GVC-RT_I.pt and GVC-RT_P.pt, commit '+ORIGINAL_COMMIT+'. It loads no adaptation weights. Per-point byte counts, stream hashes and Original codec feature traces are checked; LPIPS-VGG and RAFT residual metrics are newly computed for every state. Compatible older scalar components are reused only after exact reconstructed RGB and metric-weight checks.','','See metrics.csv for raw values; rd_points_by_dataset.csv for equal-video aggregates; paired_deltas_by_dataset_qp.csv and paired_deltas_by_video.csv for paired differences. Local lossless media and SHA256 are recorded in video_manifest.json. Preview sizes are never used as neural rates. Videos are records, with no claim of a subjective blind test.','','Integrity and publication: final_integrity.json and publication_status.json. This report records execution and measurements only; it draws no method or causal conclusion.']
    if data['aggregate']:
        text+=['','Measured dataset/QP/model aggregates (full precision in rd_points_by_dataset.csv):','','| Dataset | Model | QP | bpp | PSNR | LPIPS-Alex | LPIPS-VGG | DISTS | FloLPIPS | P→P Sobel |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['aggregate']:
            values=[format(r[k],'.8g') if r[k] is not None else 'MISSING' for k in ['bpp','PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','Sobel_MCResidual_PtoP']]
            text.append('| '+r['dataset']+' | '+LABELS[r['model']]+' | '+str(r['QP'])+' | '+' | '.join(values)+' |')
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')

def main(snapshot=False):
    env();data=tables();report(data,not snapshot)
    if snapshot:return
    assert len(data['final'])==1125 and len(data['identity'])==225 and all(x['status']=='PASS' for x in data['identity']);plots(data)
    dump(ROOT/'audits/results.json',dict(status='PASS',points=1125,stream_pairs=225,time=time.time()))
if __name__=='__main__':import sys;main('--snapshot' in sys.argv)
