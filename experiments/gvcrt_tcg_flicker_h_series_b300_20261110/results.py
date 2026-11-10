"""Measured per-video tables, equal-video/QP curves and factual execution report."""
from common_s2 import *
from collections import defaultdict
METRICS=['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','RGB_MCResidual_PtoP','Sobel_MCResidual_PtoP','RGB_MCResidual_ItoP','Sobel_MCResidual_ItoP','Sobel_edge_strength','GT_Sobel_edge_strength','edge_relative_GT_bias','local16_energy_abs_bias']

def records(phase):return [load(p) for p in sorted((ROOT/'evaluation'/phase).glob('*/step_*/points/*.json'))]
def all_tables():
 history=[];latent=[];edges=[];dev=[];final=[];equiv=[]
 for g in GROUPS:
  for p in sorted((ROOT/g/'training/receipts').glob('*.json')):
   a=load(p);row={k:v for k,v in a.items() if not isinstance(v,(dict,list))};row['sample_keys']=';'.join(a['sample_keys']);history.append(row)
   for j,w in enumerate(a['windows']):
    for t in w['trace']:latent.append(dict(phase='training',model=g,step=a['additional_step'],sample_key=a['sample_keys'][j],**t))
 for phase in ['development','final']:
  for a in records(phase):
   assert a['status']=='PASS';r=a['result'];(dev if phase=='development' else final).append(r);ids={k:r[k] for k in ['model','step','dataset','sequence','QP']};edges.append(dict(phase=phase,**ids,**{k:r[k] for k in ['Sobel_edge_strength','GT_Sobel_edge_strength','edge_relative_GT_bias','local16_energy_abs_bias','local16_energy_signed_bias']}))
   for path,h in a['record_SHA256'].items():
    if '/latent/' in path:
     for t in csv.DictReader((ROOT/path).open()):latent.append(dict(phase=phase,**ids,**t))
    if '/equiv/' in path:
     for t in csv.DictReader((ROOT/path).open()):equiv.append(dict(phase=phase,**ids,**t))
 if history:write_csv(ROOT/'training_history.csv',history)
 if latent:write_csv(ROOT/'latent_temporal_diagnostics.csv',latent)
 if edges:write_csv(ROOT/'edge_strength_by_video_qp.csv',edges)
 if dev:write_csv(ROOT/'development_metrics.csv',dev)
 if final:write_csv(ROOT/'metrics.csv',final)
 if equiv:write_csv(ROOT/'equivariance_development_controls.csv',equiv)
 base={(r['sequence'],r['QP']):r for r in dev if r['model']=='B300'};pairs=[];matched=[]
 for r in dev:
  key=(r['sequence'],r['QP'])
  if r['model'] in GROUPS and key in base:
   b=base[key];pairs.append(dict(model=r['model'],step=r['step'],sequence=r['sequence'],QP=r['QP'],reference='B300 s=0.5',**{k:r[k]-b[k] if r[k] is not None and b[k] is not None else None for k in METRICS},same_stream=r['stream_SHA256']==b['stream_SHA256'],same_codec_trace=r['codec_trace_SHA256']==b['codec_trace_SHA256'],same_bpp=r['bpp']==b['bpp']))
 look={(r['model'],r['step'],r['sequence'],r['QP']):r for r in dev}
 for step in [50,100]:
  for lhs,rhs in [(GROUPS[1],GROUPS[0]),(GROUPS[2],GROUPS[1])]:
   for seq,q in sorted(base):
    a=look.get((lhs,step,seq,q));b=look.get((rhs,step,seq,q))
    if a and b:matched.append(dict(lhs=lhs,rhs=rhs,step=step,sequence=seq,QP=q,**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in METRICS}))
 if pairs:write_csv(ROOT/'development_paired_by_video_qp.csv',pairs)
 if matched:write_csv(ROOT/'development_matched_steps.csv',matched)
 aggregates=[]
 for phase,rows in [('development',dev),('final',final)]:
  gp=defaultdict(list)
  for r in rows:gp[(r['dataset'],r['model'],r['step'],r['QP'])].append(r)
  for (dataset,model,step,q),items in sorted(gp.items()):
   row=dict(phase=phase,dataset=dataset,model=model,step=step,QP=q,videos=len(items),aggregation='equal video weight within dataset/QP')
   for k in ['bpp',*METRICS]:
    vals=[r[k] for r in items if r[k] is not None];row[k]=sum(vals)/len(vals) if vals else None;row[k+'_available_videos']=len(vals)
   aggregates.append(row)
 if aggregates:write_csv(ROOT/'rd_points_by_dataset.csv',aggregates)
 return dict(history=history,development=dev,final=final,paired=pairs,matched=matched,aggregates=aggregates)

def plots(data):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'figure.figsize':(8,5),'savefig.dpi':130});outputs=[]
 rows=data['aggregates'];dirs={'PSNR':'↑',**{k:'↓' for k in ['LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','Sobel_MCResidual_PtoP','local16_energy_abs_bias']}}
 for phase,ds in sorted({(r['phase'],r['dataset']) for r in rows}):
  sub=[r for r in rows if r['phase']==phase and r['dataset']==ds];models=sorted({(r['model'],r['step']) for r in sub})
  for metric,direction in dirs.items():
   fig,ax=plt.subplots();coincident={};note=[]
   for i,(model,step) in enumerate(models):
    z=sorted([r for r in sub if r['model']==model and r['step']==step],key=lambda r:r['QP']);assert len(z)==3
    xy=tuple((r['bpp'],r[metric]) for r in z);assert all(y is not None for x,y in xy);label=LABELS.get(model,model)+(f' @{step}' if model in GROUPS or model.endswith('_selected') else '')
    if xy in coincident:note.append(f'{label} overlaps {coincident[xy]}')
    coincident[xy]=label;ax.plot([x for x,y in xy],[y for x,y in xy],['o-','s--','^-.','D:','v-','x--'][i%6],label=label,markersize=4)
    for r in z:ax.annotate('QP'+str(r['QP']),(r['bpp'],r[metric]),xytext=(3,3),textcoords='offset points',fontsize=6)
   title='ULong development: 32 videos' if phase=='development' else ds.replace('_',' ')
   if ds=='MCL_JCV_30':title+=' (historically used test set)'
   ax.set_title(title);ax.set_xlabel('Actual RANS bpp (I frame and stream headers included)');ax.set_ylabel(metric+' '+direction);ax.legend(fontsize=7);ax.grid(alpha=.25);fig.text(.01,.01,'Only three measured QP points; no extrapolation or BD-rate.'+(' Overlap: '+'; '.join(note) if note else ''),fontsize=6);fig.tight_layout(rect=(0,.04,1,1));path=ROOT/'plots'/phase/ds/(metric+'.png');path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path);plt.close(fig);outputs.append(dict(path=str(path),SHA256=sha(path)))
 dump(ROOT/'plots/manifest.json',dict(status='PASS',files=outputs,no_cross_dataset_curves=True,actual_RANS_bpp=True,no_BD_rate=True))

def report(data):
 sel=load(ROOT/'selection.json') if (ROOT/'selection.json').exists() else None;gate=load(ROOT/'audits/H2_gate.json') if (ROOT/'audits/H2_gate.json').exists() else None;cal=load(ROOT/'gradient_calibration.json') if (ROOT/'gradient_calibration.json').exists() else None
 lines=['# Corrected H-series B300 execution record','','Status: '+load(ROOT/'status.json')['status']+'. See status.json for current stage.','','All models restore original I/P → complete pilot generator/Bridge/persistent buffers → B+1500 TCG → A2000 LoRA → B30052-key overlay last. LoRA is retained without merging at s=0.5. Full Bridge801124, TCG137760, LoRA59392 are trainable; original decoder backbone, codec, I model, entropy and metric networks are frozen. H0 is a continuation control whose difference from B300 includes widening Bridge training and further updates; this report assigns no single-factor causal effect.','','Base objective: LPIPS-Alex +0.5 DISTS +0.1383589248954108 RGB-L1 +0.05 RAFT-motion +2.603297563918803×(RGB MCResidual +0.5 Sobel MCResidual). Training LPIPS is Alex, evaluation records Alex and VGG separately ([0,1] RGB, normalize=True, v0.1 weights).','','H1 adds hard-mask-area-normalized confidence-weighted latent differences with shared differentiable channel RMS, tau0.02. Flow is estimated by frozen RAFT on GT RGB, not a known true motion field. H2 additionally proposes four stateless paired translation losses/window; both sides disable TCG and cache updates; target is no_grad. No edge-floor, Teacher, Router, GAN, SFT, history alignment or new upsampler.','','Fresh AdamW: Bridge2e-6, TCG1e-5, LoRA5e-5; betas(.9,.999), eps1e-8, weight_decay1e-4, globalclip1. Exactly four logical windows/update,64 frames/window, current-weight warmup1–48, supervised49–63 segmented4/4/4/3. Maximum100 updates; no automatic extension. GPU0 H0, GPU1 H1, GPU2 H2 if mechanically allowed, GPU3 serialized bounded cache/evaluation.','','Calibration uses10 frozen training windows, no optimizer updates or development/test choice. '+(f'lambda_lat={cal["lambda_lat"]}; lambda_equiv={cal["lambda_equiv"]}.' if cal else 'Calibration pending.'),'','H2 mechanical gate: '+(f'{gate["status"]}; median16px/8px error ratio atcrop64={gate["median_16px_to_8px_error_ratio"]}. The assistant-added0.1 launch cutoff was withdrawn by explicit user instruction. Original controls remain unchanged and diagnostic; H2 is trained with the original calibrated objective, padding and crop.' if gate else 'pending'),'', '| Group | Added updates | Dev50 | Dev100 | Numerical screen |','|---|---:|---:|---:|---|']
 for g in GROUPS:
  p=ROOT/g/'training/progress.json';u=load(p)['committed_updates'] if p.exists() else 0;count=lambda s:sum(r['model']==g and r['step']==s for r in data['development']);screen=sel['groups'][g]['status'] if sel else 'pending';lines.append(f'| {g} | {u} | {count(50)}/96 | {count(100)}/96 | {screen} |')
 lines+=['','Mechanical screen compares every QP mean DISTS, LPIPS-Alex, LPIPS-VGG and FloLPIPS to≤1.005×B300 and requires meanQP P→P Sobel ratio S<1. Eligible50/100 checkpoints are selected by lowest S; differences below0.0005 prefer earlier. NOT_QUALIFIED never substitutes step0. This is a numerical screen, not a claim about human-visible flicker.','','Aggregation: first64-frame video metrics, then equal video weights within dataset/split/QP. I→P/P→P temporal metrics are separate. Candidate RGB never enters the independent Original codec reference trajectory. Rates use real RANS bytes including actual I frame and stream headers. Byte counts, bpp, stream SHA and decoded codec-feature traces are paired checks.','','Final test status: '+(load(ROOT/'final_test_status.json')['status'] if (ROOT/'final_test_status.json').exists() else 'pending conditional numerical screen')+'. No qualification means no new final R-D points. Existing32new-holdout/6historical ULong,5HEVC-B,2historical HoneyBee/Jockey UVG and30historical MCL-JCV remain distinct test plots when measured. No cross-dataset pooled curve, extrapolation or three-point BD-rate.','','Video materials retain full-resolution single-model FFV1 and unscaled ROI, with labels, source frames, fps and SHA. Anonymous label mapping is separate. person_dog is recovered from audited original RGB frames; its MP4 is identification material only, historical exposure is retained, and it is excluded from training and model selection. No subjective experiment is claimed.','','Raw development values: development_metrics.csv; paired baseline differences: development_paired_by_video_qp.csv; same-step H1−H0 and H2−H1 where available: development_matched_steps.csv. H2 started later after the user withdrew the assistant-added cutoff; its matched-step rows are populated from actual completed evaluations. Edge/local energy values are diagnostics and not quality-improvement guarantees.','','Measured equal-video aggregates:','','| Phase | Dataset | Model | Step | QP | bpp | PSNR | Alex | VGG | DISTS | FloLPIPS | P→P Sobel |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in data['aggregates']:
  values=[format(r[k],'.8g') if r[k] is not None else 'MISSING' for k in ['bpp','PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','Sobel_MCResidual_PtoP']];lines.append('| '+r['phase']+' | '+r['dataset']+' | '+r['model']+' | '+str(r['step'])+' | '+str(r['QP'])+' | '+' | '.join(values)+' |')
 lines+=['','This report records execution, coverage and numbers only. Integrity/publication states are in final_integrity.json and publication_status.json.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main(snapshot=False):
 env();data=all_tables();report(data)
 if snapshot:return
 assert load(ROOT/'selection.json')['status']=='FROZEN';plots(data);dump(ROOT/'audits/results.json',dict(status='PASS',development_points=len(data['development']),final_points=len(data['final']),time=time.time()))
if __name__=='__main__':main('--snapshot' in sys.argv)
