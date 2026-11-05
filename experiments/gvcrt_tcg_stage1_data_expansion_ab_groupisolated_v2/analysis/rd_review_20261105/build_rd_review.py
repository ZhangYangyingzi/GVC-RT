"""Plot frozen measurements only; no inference, fitting, or training."""
from pathlib import Path
from collections import defaultdict
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / 'metrics_v3.csv'
ROWS = list(csv.DictReader(SOURCE.open(newline='')))
METRICS = ['LPIPS', 'DISTS', 'FloLPIPS', 'PSNR', 'WarpingError_GTFlow_PWC']
STATES = ['parent0', 'parent5000', 'A_common', 'A_best_dev', 'B_common']
LABELS = ['Zero-residual baseline', 'TCG parent5000', 'A +1500 (common)', 'A +500 (dev-best)', 'B +1500 (common / dev-best)']
COLORS = ['#64748b', '#2563eb', '#f59e0b', '#0d9488', '#c026d3']
STYLES = ['-', '--', ':', '-.', '-']
QS = ['0', '5', '9']
groups = defaultdict(list)
for r in ROWS:
    groups[(r['split'], r['sequence'], r['QP'])].append(r)
assert len(ROWS) == 675 and len(groups) == 135
for key, rows in groups.items():
    assert set(r['model_state'] for r in rows) == set(STATES), key
    for field in ['real_bytes', 'bpp', 'kbps', 'bitstream_sha256']:
        assert len({r[field] for r in rows}) == 1, (key, field)
aggregates, changes, audits = [], [], {}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
for split, expected in [('new_holdout_test', 32), ('historical_reference', 13)]:
    vals = {}
    for state in STATES:
        for q in QS:
            rs = [r for r in ROWS if r['split'] == split and r['model_state'] == state and r['QP'] == q]
            assert len(rs) == expected
            rec = dict(split=split, model_state=state, QP=int(q), video_count=expected, **{m:float(np.mean([float(r[m]) for r in rs])) for m in ['bpp','kbps']+METRICS})
            aggregates.append(rec); vals[(state,q)] = rec
    rates = np.array([vals[('parent0',q)]['bpp'] for q in QS])
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.3))
    for ax, metric in zip(axs.flat, METRICS):
        for state,label,color,style in zip(STATES,LABELS,COLORS,STYLES):
            ax.plot(rates, [vals[(state,q)][metric] for q in QS], label=label, color=color, linestyle=style, marker='o', markersize=5, linewidth=2 if state=='B_common' else 1.5)
        direction = 'higher is better' if metric == 'PSNR' else 'lower is better'
        title = 'GT-flow warp error' if metric == 'WarpingError_GTFlow_PWC' else metric
        ax.set_title(f'{title} ({direction})', loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel('dB' if metric == 'PSNR' else 'Metric value')
    ax = axs.flat[-1]
    for metric,color in zip(['LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC'],['#2563eb','#0d9488','#c026d3','#b45309']):
        yy = [100*(vals[('B_common',q)][metric]/vals[('parent5000',q)][metric]-1) for q in QS]
        ax.plot(rates,yy,marker='o',label='GT-flow warp' if metric=='WarpingError_GTFlow_PWC' else metric,color=color,linewidth=1.8)
    ax.axhline(0, color='#64748b', linewidth=1)
    ax.set_title('B +1500 vs parent5000: incremental change',loc='left',fontsize=11,fontweight='bold')
    ax.set_ylabel('Change (%) | negative = better'); ax.legend(fontsize=8,loc='best')
    for ax in axs.flat:
        ax.set_xscale('log'); ax.set_xticks(rates, [f'{v:.5f}\nQP {q}' for v,q in zip(rates,QS)])
        ax.minorticks_off(); ax.set_xlabel('Mean measured bpp (log scale)')
        ax.grid(True, color='#e2e8f0', linewidth=.7); ax.set_axisbelow(True)
    title = 'New held-out ULong test' if split == 'new_holdout_test' else 'Historical reference (reported separately)'
    fig.suptitle(f'{title} | {expected} videos | measured rate-quality curves',fontsize=17,fontweight='bold',y=.985)
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.946),ncol=3,frameon=False,fontsize=10)
    note = ('Each point: equal-video mean at one QP; individual video rates differ. For the same video/QP, all models have identical stream hashes and rates.\n'
            'Three measured points only; connecting segments are guides, not extra measurements or a BD-rate estimate. A/B dev-best budgets differ.\n'
            'GT-flow warp is a GT-referenced warping/reconstruction error; it is not an isolated flicker measure. Source: frozen metrics_v3.csv.')
    fig.text(.055,.015,note,fontsize=9,color='#475569',va='bottom')
    fig.tight_layout(rect=[.02,.10,.99,.88],h_pad=2.0,w_pad=2.2)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'{split}_rate_quality.{ext}',dpi=160,facecolor='white')
    plt.close(fig)
    for target,reference in [('parent5000','parent0'),('B_common','parent0'),('B_common','parent5000'),('B_common','A_common'),('B_common','A_best_dev')]:
        for q in QS:
            rec = dict(split=split, target=target, reference=reference, QP=int(q), mean_bpp=float(rates[QS.index(q)]), mean_kbps=vals[(target,q)]['kbps'])
            for metric in METRICS:
                a,b=vals[(target,q)][metric],vals[(reference,q)][metric]
                rec[metric+('_delta_dB' if metric=='PSNR' else '_change_pct')] = a-b if metric=='PSNR' else 100*(a/b-1)
            changes.append(rec)
    audits[split] = dict(videos=expected, QP=[0,5,9], unique_models=5, same_video_rate_and_stream_checks='PASS')
for name,rows in [('rd_points.csv',aggregates),('same_rate_changes.csv',changes)]:
    with (OUT/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit = dict(status='PASS', source=str(SOURCE), source_sha256=sha(SOURCE), date='2026-11-05', read_only_source=True, new_training_or_evaluation=False, aggregation='equal-video arithmetic means within each QP; percentage change is ratio of means', measured_QPs=[0,5,9], no_curve_extrapolation=True, no_BD_rate_claim=True, splits=audits, files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ['.png','.pdf','.csv','.py']})
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
