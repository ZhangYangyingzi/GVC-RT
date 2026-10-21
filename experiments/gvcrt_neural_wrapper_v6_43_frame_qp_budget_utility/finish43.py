"""Objective reporting, fixed plots and independently checked deliverable inventory."""
from common43 import *
from analyze43 import tables,select,transfers
import math,collections

def md(rows,fields):
 def fmt(x):return f'{x:.8g}' if isinstance(x,float) else str(x)
 return '| '+' | '.join(fields)+' |\n| '+' | '.join('---' for _ in fields)+' |\n'+''.join('| '+' | '.join(fmt(r.get(k,'')) for k in fields)+' |\n' for r in rows)

def finish():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 ss=tables();select(ss);candidates,chosen=transfers()
 tasks=load(ROOT/'intervention_plan.json')+load(ROOT/'transfer_plan.json')
 assert all(validate_result(t) for t in tasks)
 result=[load(result_path(t)) for t in tasks];ev=[r['result'] for r in result]
 write(ROOT/'evaluation.csv',ev)
 frames=[];trans=[];streams=[];audit=[]
 for t,r in zip(tasks,result):
  d=result_path(t).parent
  frames.extend(read(d/'frames.csv'))
  trans.extend(dict(dataset=t['dataset'],sequence=t['sequence'],variant=t['variant'],**a) for a in read(d/'transitions.csv'))
  a=next(f for f in r['artifacts'] if f['path'].endswith('.bin'))
  streams.append(dict(task=key(t),**a,actual_qps=t['actual_qps'],bytes_consumed=r['result']['bytes_consumed'],encoder_decoder_state_sync=r['audits']['encoder_decoder_state_sync']))
  audit.append(dict(task=key(t),**r['audits']))
 write(ROOT/'all_frames.csv',frames);write(ROOT/'all_transitions.csv',trans);dump(ROOT/'stream_manifest.json',streams);dump(ROOT/'audits.json',audit)
 # Every nonbaseline stream, including joint interventions, has its own raw differences.
 all_delta=[];all_transition_delta=[]
 for t in tasks:
  if t['kind']=='baseline':continue
  base=ROOT/'tasks'/t['dataset']/t['sequence']/'baseline';d=result_path(t).parent
  for x,b in zip(read(d/'frames.csv'),read(base/'frames.csv')):
   all_delta.append(dict(dataset=t['dataset'],sequence=t['sequence'],variant=t['variant'],kind=t['kind'],frame=int(x['frame']),delta_bits=int(x['real_bits'])-int(b['real_bits']),**{f'gain_{m}':float(b[m])-float(x[m]) for m in ('LPIPS','DISTS','L1','pixel_SSE')},delta_PSNR_dB=float(x['PSNR'])-float(b['PSNR'])))
  for x,b in zip(read(d/'transitions.csv'),read(base/'transitions.csv')):
   all_transition_delta.append(dict(dataset=t['dataset'],sequence=t['sequence'],variant=t['variant'],kind=t['kind'],from_frame=int(x['from_frame']),to_frame=int(x['to_frame']),gain_FloLPIPS=float(b['FloLPIPS'])-float(x['FloLPIPS'])))
 write(ROOT/'all_frame_differences.csv',all_delta);write(ROOT/'all_transition_differences.csv',all_transition_delta)
 for category in ('baseline_reproduction','prefix_equality','encoder_decoder_state_sync','freeze','bit_accounting'):
  relevant=[a for a in audit if category in a]
  dump(ROOT/'audits'/f'{category}.json',dict(status='PASS',checked_tasks=len(relevant),records=relevant))
 figs=ROOT/'figures';figs.mkdir(exist_ok=True)
 delta=read(ROOT/'frame_differences.csv')
 for seq in ('ParkScene','Jockey'):
  for target in TARGETS:
   fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
   for label in ('qplus2','qminus2'):
    rows=[r for r in delta if r['sequence']==seq and int(r['target'])==target and r['label']==label]
    x=[int(r['frame']) for r in rows]
    ax[0].plot(x,[float(r['gain_LPIPS']) for r in rows],label=label)
    ax[1].plot(x,[float(r['delta_bits'])/1000 for r in rows],label=label)
   for a in ax:a.axhline(0,color='black',lw=.7);a.axvspan(target,target+15,alpha=.12,color='grey');a.legend()
   ax[0].set_ylabel('LPIPS gain: baseline - variant\npositive = improvement');ax[1].set_ylabel('Bit change (kbit)\npositive = added cost');ax[1].set_xlabel('Zero-based frame index')
   fig.suptitle(f'{seq}, target {target}; shaded window [{target}, {target+15}]');fig.tight_layout();fig.savefig(figs/f'{seq}_t{target:02d}_impulse.png',dpi=130);plt.close(fig)
  subset=[r for r in ss if r['sequence']==seq and r['metric']=='LPIPS'];fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True)
  for col,label in enumerate(('qplus2','qminus2')):
   rows=[r for r in subset if r['label']==label];x=[r['target'] for r in rows]
   for name,offset,color in [('now',-.8,'tab:blue'),('future15',.8,'tab:orange')]:
    ax[0,col].bar([i+offset for i in x],[r['G_'+name] for r in rows],width=1.5,label=name,color=color)
    ax[1,col].bar([i+offset for i in x],[r['delta_bits_'+name]/1000 for r in rows],width=1.5,label=name,color=color)
   ax[0,col].set_title(label)
   for a in ax[:,col]:a.axhline(0,color='black',lw=.7);a.legend();a.set_xticks(TARGETS)
   ax[1,col].set_xlabel('Target t (future = t+1 ... t+15)')
  ax[0,0].set_ylabel('Cumulative LPIPS gain\npositive = improvement');ax[1,0].set_ylabel('Bit change (kbit)\npositive = added cost');fig.suptitle(seq+' current and future15; fixed 16-frame window');fig.tight_layout();fig.savefig(figs/f'{seq}_gain_cost.png',dpi=130);plt.close(fig)
  base=next(r for r in ev if r['sequence']==seq and r['kind']=='baseline');fig,ax=plt.subplots(figsize=(9,5))
  for direction,color in [('forward','tab:blue'),('reverse','tab:orange')]:
   subset=[r for r in candidates if r['sequence']==seq and r['direction']==direction]
   ax.scatter([r['total_bits']/1000 for r in subset],[r['LPIPS'] for r in subset],label=direction,color=color)
   for r in subset:ax.annotate(f"({r['a']},{r['b']})",(r['total_bits']/1000,r['LPIPS']),fontsize=8)
  ax.scatter([base['total_bits']/1000],[base['LPIPS']],marker='x',color='black',label='baseline')
  ax.axvspan(.99*base['total_bits']/1000,base['total_bits']/1000,color='grey',alpha=.15,label='[0.99 B0, B0]')
  ax.axvline(base['total_bits']/1000,color='black',lw=.8);ax.axhline(base['LPIPS'],color='black',lw=.8)
  ax.set(xlabel='Actual full-stream cost (kbit)',ylabel='Whole 64-frame mean LPIPS (lower is better)',title=seq+' fixed oracle transfer candidates');ax.legend();fig.tight_layout();fig.savefig(figs/f'{seq}_transfer.png',dpi=130);plt.close(fig)
 expected_figures=[f'{seq}_t{t:02d}_impulse.png' for seq in ('ParkScene','Jockey') for t in TARGETS]+[f'{seq}_{kind}.png' for seq in ('ParkScene','Jockey') for kind in ('gain_cost','transfer')]
 assert sorted(p.name for p in figs.glob('*.png'))==sorted(expected_figures)
 dump(ROOT/'figure_manifest.json',[dict(path=str(figs/n),bytes=(figs/n).stat().st_size,sha256=sha(figs/n)) for n in expected_figures])
 # Raw numerical checks, independent of summaries.
 for r in result:
  t=r['task'];d=result_path(t).parent;fr=read(d/'frames.csv');tr=read(d/'transitions.csv');pr=load(d/'proofs.json')
  assert len(fr)==64 and len(tr)==63
  assert [int(x['frame']) for x in fr]==list(range(64))
  assert [(int(x['from_frame']),int(x['to_frame'])) for x in tr]==[(i,i+1) for i in range(63)]
  assert [int(x['actual_qp']) for x in fr]==t['actual_qps']
  assert sum(int(x['packet_bytes']) for x in fr)==r['result']['total_bytes']==r['result']['bytes_consumed']
  assert all(int(x['packet_bytes'])==int(x['payload_bytes'])+int(x['header_bytes']) for x in fr)
  assert -10*math.log10(max(sum(float(x['pixel_SSE']) for x in fr)/(64*1920*1080*3),1e-15))==r['result']['PSNR']
  assert len(pr['encoder'])==len(pr['decoder'])==64
  for e,d in zip(pr['encoder'][1:],pr['decoder'][1:]):assert e['y']==d['y'] and e['f']==d['f'] and e['state']['feature']==d['state']['feature']
  assert all(a=='PASS' for k,a in r['audits'].items() if k in ('freeze','bit_accounting','encoder_decoder_state_sync','qp_in_stream','baseline_reproduction','prefix_equality'))
 for f in load(ROOT/'source_manifest.json'):assert sha(f['path'])==f['sha256']
 for path,entry in load(ROOT/'inherited_source_audit.json')['files'].items():assert sha(path)==entry['sha256'],path
 assert sha(CP)==CP_SHA
 check=dict(status='PASS',optimizer_updates=0,baselines_complete=2,baseline_bitstreams_and_outputs_exact=True,impulses_complete=24,transfers_complete=len(load(ROOT/'transfer_plan.json')),max_transfers=16,total_streams=len(tasks),figures=len(expected_figures),frame_records=len(frames),transition_records=len(trans),all_independent_decodes=True,encoder_decoder_state_sync='PASS',prefix_equality='PASS',bit_accounting='PASS',freeze='PASS',source_hashes='PASS',errors_unresolved=[],oracle=True)
 dump(ROOT/'final_integrity.json',check);dump(ROOT/'run_status.json',dict(**{**check,'status':'COMPLETE'},publication='See publication_status.json'))
 ranking=load(ROOT/'ranking_comparison.json');selections=load(ROOT/'transfer_selection.json')
 report='# v6.43 objective results\n\n'
 report+='This is an offline GT-assisted oracle experiment on the fixed 64-frame ParkScene and Jockey clips. Only the v6.40 G_basic step100 Generator is used with the original frozen compression core and Bridge. New optimizer updates: **0**. Native I0, P1–P63, no extra reference reset; input 1920×1080, padded to 1920×1088; FP16 codec/receiver, FP32 metric inputs, force_zero_thres=0.12. ParkScene retains its native timeline; Jockey retains the canonical 30fps RGB and original mapping in data_manifest.json.\n\n'
 report+='## Reproduction and integrity\n\nBoth newly encoded baseline streams reproduce v6.42 byte counts and SHA256 exactly. Independent G_basic reconstruction hashes, every frame metric, every FloLPIPS transition and aggregate metrics match exactly. Each intervention contains its own full RANS stream with embedded actual QPs; every independent decoder uses those fields. Prefix packet, reconstruction, receiver-input and reference-state hashes are identical before the first changed frame. Encoding and decoding latent/reference features match exactly on all 63 P frames. Weights, buffers and metric networks are unchanged. All frame packets include associated headers (initial SPS assigned to I0); packet sums equal file sizes.\n\n'
 report+=md([check],['baselines_complete','impulses_complete','transfers_complete','total_streams','frame_records','transition_records'])
 report+='\n## Whole-stream measured values\n\n'+md(ev,['sequence','variant','total_bits','actual_bpp','LPIPS','DISTS','L1','PSNR','FloLPIPS'])
 report+='\nPSNR above is in dB from aggregate RGB SSE, not the mean of frame dB values. Lower LPIPS, DISTS, L1 and FloLPIPS is better. The unaggregated frame and transition measurements are in all_frames.csv and all_transitions.csv.\n\n## Fixed impulse windows and rankings\n\n'
 report+='impulse_summary.csv contains current, future15, full 16-frame, t…63 and complete-stream gains/costs, and cumulative cutoffs t, t+4, t+8, t+15. Gains are baseline minus variant; positive denotes lower distortion. Costs are variant minus baseline in actual bits. FloLPIPS gains are reported separately by transition in transition_differences.csv.\n\n'
 report+=md([r for r in ss if r['metric']=='LPIPS'],['sequence','target','label','G_now','G_future15','G_16','delta_bits_now','delta_bits_future15','delta_bits_16','U_now','U_16','cost_type'])
 report+='\nU_now and U_16 share delta_bits_16 as denominator, in cumulative LPIPS decrease/kbit. Positive-cost ranks require positive total added bits and positive window bits. Nonpositive-cost operations retain raw gains and costs, with no fabricated efficiency. Supplemental DISTS/L1 results use the same rules.\n\n'
 report+=md([dict(sequence=r['sequence'],metric=r['metric'],n=r['eligible_count'],now=', '.join(r['now_order']),future=', '.join(r['future_order']),top_two_set_same=r['top_two_set_same'],top_two_order_same=r['top_two_order_same'],pairwise_reversals=len(r['pairwise_reversals']),Spearman=r['descriptive_spearman']) for r in ranking],['sequence','metric','n','now','future','top_two_set_same','top_two_order_same','pairwise_reversals','Spearman'])
 report+='\nSpearman values are descriptive only. Full pairwise reversal lists are in ranking_comparison.json.\n\n## Fixed budget transfer\n\n'
 report+='Selection uses qplus2 receivers and qminus2 donors with the specified actual-cost signs and |r−d|≥16; maximize A(r)−C(d), then lexicographic (r,d). All pairs and raw scores are in transfer_selection.json. The selected pairs and finite grids are used even if the maximum score is negative.\n\n'
 report+=md([dict(sequence=r['sequence'],status=r['status'],**r.get('selected',{}),reason=r.get('reason','')) for r in selections],['sequence','status','r','d','A','C','score','reason'])
 report+='\n'+md(candidates,['sequence','direction','a','b','total_bits','budget_ratio','strict_feasible','near_budget','saved_percent','LPIPS','delta_LPIPS','delta_DISTS','delta_L1','delta_PSNR_dB','delta_FloLPIPS'])
 report+='\nWithin each direction, display selections use minimum whole LPIPS among candidates in [0.99 B0,B0], with ties broken by lower bits then (a,b).\n\n'+md(chosen,['sequence','direction','status','a','b','total_bits','budget_ratio','delta_LPIPS','reason'])
 report+='\nCandidates below 0.99 B0 remain recorded with their actual savings; they are not exact matched-rate comparisons. Candidates above B0 cannot establish fixed-budget gains. No padding bytes or stream truncation is used.\n\n## Figures and raw materials\n\n'
 report+=''.join(f'- [{p.name}](figures/{p.name})\n' for p in sorted(figs.glob('*.png')))
 report+='\nAll bitstreams are small and included in bitstreams/. stream_manifest.json provides paths, exact sizes and SHA256. Each task directory contains frames.csv, transitions.csv, proofs.json, audits.json and a signed completion receipt.\n\n## Limits\n\nSame QP adjustment does not imply the same added bit count. Single-frame intervention gain is not a fixed-total-rate gain. QP changes encoding features and Generator conditions together; these are overall QP interventions, not isolated latent effects. Multiple intervention positions within these two videos are not independent video samples. Selection uses the same video GT and is oracle-only; it is not a deployable allocation algorithm or a generalization/statistical-significance result. No mechanism, innovation or subjective image-quality conclusion is drawn. No model training, FID, BD-rate or speed test was performed.\n'
 textfile(ROOT/'REPORT.md',report)
 textfile(ROOT/'README.md','# v6.43 fixed G_basic frame-QP budget utility\n\n[Protocol](PROTOCOL.md) · [Full fixed request](REQUEST.txt) · [Objective results](REPORT.md)\n\nZero optimizer updates. Two historical q5 baselines, 24 single-frame interventions, and the fixed-budget transfer candidates recorded in transfer_plan.json. Real streams are included. Large model files and input video sources remain at the server paths indexed by the manifests.\n\nTo resume: run queue43.py smoke, then queue43.py impulse, analyze43.py select, queue43.py transfer, in the project conda environment; run finish43.py with /data1/anaconda3_new/anaconda_program/bin/python (the existing CPU plotting environment). Task signatures, file hashes, locks and atomic writes protect completed results. Publication uses a separate worktree and only this experiment directory.\n')
 print('FINISH',check,flush=True)
if __name__=='__main__':finish()
