from pathlib import Path
import csv
import json
import hashlib
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path('/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_coverage_ab_20261107')
OUT = Path(__file__).resolve().parent
QPS = ['0', '5', '9']

def read(name):
    with (EXP / name).open() as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def avg(rows, metric):
    return st.mean(float(r[metric]) for r in rows)

def main():
    baseline = read('common_baseline/evaluation/development_baseline/metrics.csv')
    data = read('A_control/development_metrics.csv')
    rows = [dict(step=0, FloLPIPS_percent=0.0, DISTS_percent=0.0, LPIPS_percent=0.0, composite_equal_normalized=1.0)]
    by_qp = []
    for step in sorted({int(r['LoRA_step']) for r in data}):
        current = {'step': step}
        for metric in ['FloLPIPS', 'DISTS', 'LPIPS']:
            vals = []
            for qp in QPS:
                c = [r for r in data if int(r['LoRA_step']) == step and r['QP'] == qp]
                b = [r for r in baseline if r['QP'] == qp]
                assert len(c) == len(b) == 32
                delta = 100 * (avg(c, metric) / avg(b, metric) - 1)
                by_qp.append(dict(step=step, QP=int(qp), videos=32, metric=metric, relative_change_percent=delta))
                vals.append(delta)
            current[metric + '_percent'] = st.mean(vals)
        current['composite_equal_normalized'] = 1 + (current['FloLPIPS_percent'] + current['DISTS_percent']) / 200
        rows.append(current)
    for x in rows:
        x['pareto_FloLPIPS_DISTS'] = not any(y['FloLPIPS_percent'] <= x['FloLPIPS_percent'] and y['DISTS_percent'] <= x['DISTS_percent'] and (y['FloLPIPS_percent'] < x['FloLPIPS_percent'] or y['DISTS_percent'] < x['DISTS_percent']) for y in rows)
    write('A_development_pareto.csv', rows)
    write('A_development_per_qp.csv', by_qp)
    best = min(rows, key=lambda r: r['composite_equal_normalized'])
    allrows = read('analysis/rd_with_original_20261108/metrics_with_original.csv')
    uvg = [r for r in allrows if r['dataset'] == 'UVG_historical_HoneyBee_Jockey_2']
    lookup = {(r['model'], r['sequence'], r['QP']): r for r in uvg}
    detailed, summary = [], []
    for video in ['HoneyBee', 'Jockey']:
        for candidate, reference in [('baseline', 'GVC_RT_original'), ('A_control', 'GVC_RT_original'), ('B_expanded', 'GVC_RT_original'), ('A_control', 'baseline'), ('B_expanded', 'baseline')]:
            for metric in ['FloLPIPS', 'LPIPS', 'DISTS', 'PSNR', 'WarpingError_GTFlow_PWC']:
                changes = []
                for qp in QPS:
                    x = float(lookup[candidate, video, qp][metric])
                    y = float(lookup[reference, video, qp][metric])
                    delta = x - y if metric == 'PSNR' else 100 * (x / y - 1)
                    changes.append(delta)
                    detailed.append(dict(video=video, QP=int(qp), candidate=candidate, reference=reference, metric=metric, candidate_value=x, reference_value=y, change=delta, unit='dB' if metric == 'PSNR' else 'percent'))
                summary.append(dict(video=video, candidate=candidate, reference=reference, metric=metric, equal_QP_change=st.mean(changes), unit='dB' if metric == 'PSNR' else 'percent'))
    write('UVG_per_video_qp.csv', detailed)
    write('UVG_per_video_summary.csv', summary)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        is_p = r['pareto_FloLPIPS_DISTS']
        ax.scatter(r['FloLPIPS_percent'], r['DISTS_percent'], c='#0072B2' if is_p else '#888888', s=65 if is_p else 35, zorder=3)
        ax.annotate('TCG baseline' if r['step'] == 0 else str(r['step']), (r['FloLPIPS_percent'], r['DISTS_percent']), xytext=(5, 5), textcoords='offset points', fontsize=9)
    frontier = sorted([r for r in rows if r['pareto_FloLPIPS_DISTS']], key=lambda r:r['FloLPIPS_percent'])
    ax.plot([r['FloLPIPS_percent'] for r in frontier], [r['DISTS_percent'] for r in frontier], '--', color='#0072B2', alpha=.7)
    ax.axhline(0, color='black', linewidth=.7)
    ax.axvline(0, color='black', linewidth=.7)
    ax.set(xlabel='FloLPIPS change vs TCG baseline (%) - lower is better', ylabel='DISTS change vs TCG baseline (%) - lower is better', title='A: all saved development checkpoints, 32 videos x 3 QPs')
    ax.grid(alpha=.2)
    ax.margins(.18)
    fig.tight_layout()
    fig.savefig(OUT / 'A_development_pareto.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, video in enumerate(['HoneyBee', 'Jockey']):
        for j, metric in enumerate(['FloLPIPS', 'DISTS']):
            ax = axes[i,j]
            for model, label, color in [('baseline', 'TCG B+1500', '#0072B2'), ('A_control', 'TCG + LoRA A2000', '#D55E00'), ('B_expanded', 'TCG + LoRA B2000', '#009E73')]:
                pp = [r for r in detailed if r['video'] == video and r['metric'] == metric and r['candidate'] == model and r['reference'] == 'GVC_RT_original']
                ax.plot([r['QP'] for r in pp], [r['change'] for r in pp], marker='o', color=color, label=label)
            ax.axhline(0, linestyle='--', color='black', label='Original GVC-RT')
            ax.set_title(video + ': ' + metric)
            ax.set_xlabel('External QP (9 is higher bitrate in this experiment)')
            ax.set_ylabel('Change vs original GVC-RT (%)')
            ax.set_xticks([0, 5, 9])
            ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Existing UVG test subset: per-video changes; no new evaluation')
    fig.tight_layout(rect=(0,.05,1,.97))
    fig.savefig(OUT / 'UVG_per_video_qp.png', dpi=160)
    plt.close(fig)
    sources = ['common_baseline/evaluation/development_baseline/metrics.csv', 'A_control/development_metrics.csv', 'analysis/rd_with_original_20261108/metrics_with_original.csv', 'convlora.py', 'protocol.json']
    result = dict(status='CPU_DIAGNOSTICS_COMPLETE', inference_scale_experiment='NOT_RUN', official_selection_unchanged=True, all_saved_steps_checked=12, pareto_steps_including_baseline=[r['step'] for r in rows if r['pareto_FloLPIPS_DISTS']], illustrative_composite='0.5*mean_q(F/Fbase)+0.5*mean_q(D/Dbase); exploratory only, not a new official selection', illustrative_composite_best_step=best['step'], source_sha256={p:hashlib.sha256((EXP/p).read_bytes()).hexdigest() for p in sources})
    (OUT/'diagnostic_audit.json').write_text(json.dumps(result, indent=2))
    plan = dict(status='PROPOSED_NOT_EXECUTED', candidate='A_control/checkpoints/step_2000.pt', scope='development only; no training; no test-set tuning', first_stage=dict(external_QP=9, videos=32, new_inference_gain=0.5, existing_endpoints={'0':'TCG B+1500 matched-FP32 baseline, not original GVC-RT', '1':'A2000 existing development measurements'}, engineering_checks='Matched full-sequence smoke at gains 0 and 1 before endpoint reuse; all 5 LoRA branches scaled; original weights and TCG unchanged; reset temporal buffers and decode complete 64-frame sequence at each gain; identical streams, RGB conversion and metric weights'), scale_definition='extra gain s times existing (LoRA alpha/r)=1; keep rank=4 and intrinsic alpha=4; use s=0.5, not alpha=0.5', qp_definition='Use one gain for the full P-frame sequence at external QP9; do not alternate gain according to inherited actual frame QP9/10/11 in this first probe; deployment lookup must use decoder-known rate setting, not external hidden metadata', interpretation='An intermediate point that dominates gain1 supports excessive full-strength adaptation on the development set. Reduced DISTS cost alone, with proportional FloLPIPS loss, only shows a tradeoff. One QP cannot establish QP-specific causality.', followup_if_supported='Evaluate the same gain0.5 at QP0/5 on the same 32 development videos before preferring QP-specific gain to a common global gain; preregister selection rule and then use untouched source-group-isolated confirmation data')
    (OUT/'inference_gain_probe_proposal.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    lines=['# 选模、QP 缩放与 UVG 分视频诊断','', '仅从现有 CSV 复算；未启动 GPU 训练或缩放解码评测；正式 A2000/B2000 选模记录不变。','', '开发集 12 个 A 组训练检查点中，FloLPIPS/DISTS 双目标非支配点是 A1250 和 A2000；计入无 LoRA 基线时，基线本身也是非支配点。','', '|步数|FloLPIPS 变化|DISTS 变化|双目标非支配|','|---|---:|---:|---|']
    for r in rows:
        lines.append(f"|{r['step']}|{r['FloLPIPS_percent']:+.5f}%|{r['DISTS_percent']:+.5f}%|{r['pareto_FloLPIPS_DISTS']}|")
    lines += ['', 'A1250 DISTS 平均 +0.00966%，接近不变，但不是零退化；按 QP0/5/9 为 +0.01657%、-0.07218%、+0.08459%。A2000 的对应值为 +0.15225%、+0.09581%、+0.21937%。归一化后 50/50 的示例复合评分选择 A1250；这是探索性结果，不改写原先以 FloLPIPS 为准的正式选择。','', '## 缩放实验边界','', '提议先只测 A2000，QP9，32 个开发视频，额外旁路增益 s=0.5。s=0/1 用完整序列 smoke 验证端点后复用既有基线/A2000 指标。不能插值现有指标或混合末端 RGB 替代真正分支缩放；多层与历史缓存使整体输出不必对 s 线性。','', '当前实现 intrinsic alpha=rank=4，alpha/r=1；新乘数应单独记 s，避免把 alpha 直接设置 0.5 而得到 0.125 的错误总倍率。第一轮按实验 external QP 固定整段 P 帧的 s，不随内部 QP 调度跳变。','', '若 DISTS 改善只是退回 s=0 且丢失 FloLPIPS 增益，只说明折中。若中间 s 优于 s=1 或有更优复合收益，可支持强度过大假说；还需 QP0/5 的相同缩放对照才能确认 QP 关联。','', '## UVG','', 'HoneyBee 与 Jockey 的 A2000 平均 FloLPIPS 分别比原版高 2.7344%、1.7723%，但并非每个 QP 都差。HoneyBee QP0/5 差、QP9 好；Jockey 仅 QP5 差，QP0/9 好。DISTS 中 HoneyBee 平均 +1.5482%，Jockey -9.9945%。LoRA 相对 TCG 在这两视频的全部 QP 上均改善 FloLPIPS 与 DISTS，故不能把相对原版的剩余退化归因于 LoRA 本身。','', '两视频源文件为 120fps，本实验取帧 0,4,...,252（有效 30fps）共 64 帧，所有模型用相同采样。这不是完整 UVG 评测。本次直接检查本地源视频的第 0/128/252 帧亮度图：HoneyBee 主要呈现花丛与相近背景，Jockey 可见骑手和马的位置、尺寸及背景明显变化。静态抽帧不足以量化运动强度，不能把两个样本均认定为极端高速运动或证明运动导致退化。','', '训练已混合外部 QP0–9，并非只在一个 QP 训练；训练 UVG 视频为 Beauty/ReadySteadyGo/ShakeNDry/YachtRide，两个留出测试视频不能因表现不佳移入训练。','', '## 文件','', '- [A 组帕累托图](A_development_pareto.png)','- [开发集步数统计](A_development_pareto.csv)','- [开发集逐 QP 统计](A_development_per_qp.csv)','- [UVG 分视频逐 QP 图](UVG_per_video_qp.png)','- [UVG 明细](UVG_per_video_qp.csv)','- [未执行的缩放实验建议](inference_gain_probe_proposal.json)','']
    (OUT/'README.md').write_text(chr(10).join(lines))
    print(json.dumps({'output':str(OUT), 'pareto':result['pareto_steps_including_baseline'], 'illustrative_best':best, 'GPU_experiments_run':0}, ensure_ascii=False))

if __name__ == '__main__':
    main()

