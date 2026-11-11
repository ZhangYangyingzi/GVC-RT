"""Execution receipts, paired numerical CSVs, per-dataset plots and factual report."""
from common_s2 import *
from collections import Counter,defaultdict
from evaluate import METRICS


def csv_allow_empty(path,rows,fields):
    if rows:write_csv(path,rows)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(','.join(fields)+'\n')


def window_identity(row):
    return digest(dict(domain=row['domain'],identity=row['identity'],frames=row['source_frame_indices'],crop=[row['crop_x'],row['crop_y'],256,256]))


def coverage(rows):
    identities=[window_identity(row) for row in rows]
    return dict(windows=len(rows),unique_windows=len(set(identities)),repeat_fraction=1-len(set(identities))/len(rows) if rows else 0.,unique_source_groups=len({r['source_group'] for r in rows}),unique_clips=len({(r['domain'],r['identity']) for r in rows}),domains=dict(Counter(r['domain'] for r in rows)),QP_counts=dict(Counter(r['external_qp'] for r in rows)),real_frame_positions=sum(len(r['source_frame_indices']) for r in rows),window_definition='domain/source identity + native frame indices + spatial crop; excludes QP, slot and optimizer step')


def main():
    env();points=[]
    for path in sorted((ROOT/'evaluation').glob('*/step_*/points/*.json')):
        a=load(path);assert a['status']=='PASS';points.append(a['result'])
    csv_allow_empty(ROOT/'per_video_metrics.csv',points,['model','step','dataset','sequence','QP','bpp',*METRICS])
    base={(p['dataset'],p['sequence'],p['QP']):p for p in points if p['model']=='B300'}
    paired=[];bins=defaultdict(list)
    for p in points:
        bins[(p['dataset'],p['model'],p['step'],p['QP'])].append(p)
        b=base.get((p['dataset'],p['sequence'],p['QP']))
        if b is not None:
            assert p['stream_SHA256']==b['stream_SHA256'] and p['real_bytes']==b['real_bytes'] and p['bpp']==b['bpp'] and p['codec_trace_SHA256']==b['codec_trace_SHA256']
            row=dict(p)
            for m in METRICS:row['delta_B300_'+m]=None if p.get(m) is None or b.get(m) is None else p[m]-b[m]
            paired.append(row)
    aggregates=[]
    for (dataset,model,step,q),ps in sorted(bins.items()):
        r=dict(dataset=dataset,model=model,step=step,QP=q,videos=len(ps),bpp=sum(p['bpp'] for p in ps)/len(ps))
        bp=[(p,base[(p['dataset'],p['sequence'],p['QP'])]) for p in ps if (p['dataset'],p['sequence'],p['QP']) in base]
        r['paired_B300_videos']=len(bp)
        for metric in METRICS:
            values=[p[metric] for p in ps if p.get(metric) is not None];r[metric]=sum(values)/len(values) if values else None
            valid=[(p[metric],b[metric]) for p,b in bp if p.get(metric) is not None and b.get(metric) is not None]
            r['delta_B300_'+metric]=sum(a-b for a,b in valid)/len(valid) if valid else None
            r['relative_B300_percent_'+metric]=(sum(a for a,b in valid)/sum(b for a,b in valid)-1)*100 if valid and sum(b for a,b in valid) and metric!='PSNR' else None
        aggregates.append(r)
    csv_allow_empty(ROOT/'per_dataset_metrics.csv',aggregates,['dataset','model','step','QP','videos','bpp',*METRICS])
    csv_allow_empty(ROOT/'paired_B300_metrics.csv',paired,['model','step','dataset','sequence','QP'])
    histories=[];consumed={};plan=plan_rows()
    for g in GROUPS:
        path=ROOT/g/'training/progress.json';step=load(path)['committed_updates'] if path.exists() else 0
        consumed[g]=coverage(plan[:4*step])
        for p in sorted((ROOT/g/'training/receipts').glob('*.json')):
            a=load(p);histories.append({k:a.get(k) for k in ['group','committed_updates','effective_batch','G_loss','G_preclip_norm','G_adv_weight','D_loss','D_real_logit','D_fake_logit','seconds','time']})
    csv_allow_empty(ROOT/'training_history.csv',histories,['group','committed_updates','G_loss','seconds'])
    dump(ROOT/'training_coverage.json',dict(planned=coverage(plan),consumed=consumed,calibration_windows=24,calibration_in_optimizer_plan=False))
    plots=[]
    if aggregates:
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        for dataset in sorted({r['dataset'] for r in aggregates}):
            for metric in ['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','RGB_MCResidual','Sobel_MCResidual']:
                fig,ax=plt.subplots(figsize=(8,5))
                models=sorted({(r['model'],r['step']) for r in aggregates if r['dataset']==dataset})
                for model,step in models:
                    values=sorted([r for r in aggregates if (r['dataset'],r['model'],r['step'])==(dataset,model,step) and r.get(metric) is not None],key=lambda r:r['bpp'])
                    if not values:continue
                    ax.plot([r['bpp'] for r in values],[r[metric] for r in values],'-o',label=model if model in ['Original','B300'] else f'{model} step {step}')
                ax.set_xlabel('actual bpp (real I/P bytes included)');ax.set_ylabel(metric+(' ↑' if metric=='PSNR' else ' ↓'));ax.set_title(dataset+' development; available QP='+','.join(str(q) for q in sorted({r['QP'] for r in aggregates if r['dataset']==dataset}))+'; max n/QP='+str(max(r['videos'] for r in aggregates if r['dataset']==dataset)));ax.grid(alpha=.3);ax.legend(fontsize=7,ncol=2)
                path=ROOT/'plots'/dataset/f'RD_{metric}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig);plots.append(dict(path=str(path),SHA256=sha(path)))
    dump(ROOT/'plots/index.json',dict(plots=plots,datasets=sorted({r['dataset'] for r in aggregates}),no_BD_rate=True))
    counts={g:{str(s):sum(p['model']==g and p['step']==s for p in points) for s in [0,250,500]} for g in GROUPS}
    status=load(ROOT/'status.json');publication=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
    required={n:(ROOT/n).exists() and load(ROOT/n).get('status')=='PASS' for n in ['audits/preflight.json','audits/filter_audit.json','audits/gradient_audit.json','audits/codec_audit.json','gradient_calibration.json','evaluation/completion.json','video_manifest.json']}
    integrity=dict(status='PASS' if all(required.values()) and len(points)==1344 and all(consumed[g]['windows']==2000 for g in GROUPS) else 'IN_PROGRESS',required=required,metric_points=len(points),expected_metric_points=1344,development_counts=counts,committed_steps=status.get('committed_steps'),only_development=True,large_artifacts=[])
    for p in sorted((ROOT/'media/receipts').glob('*.json')):
        a=load(p)
        if 'path' in a:integrity['large_artifacts'].append(dict(path=a['path'],SHA256=a['SHA256'],bytes=a.get('bytes')))
    hashpath=ROOT/'audits/large_artifact_hashes.json';hashes=load(hashpath) if hashpath.exists() else {}
    for g in GROUPS:
        for p in sorted((ROOT/g/'training/checkpoints').glob('*.pt')):
            stat=p.stat();key=str(p);stamp=[stat.st_size,stat.st_mtime_ns]
            if key not in hashes or hashes[key]['stamp']!=stamp:hashes[key]=dict(stamp=stamp,SHA256=sha(p))
            integrity['large_artifacts'].append(dict(path=key,SHA256=hashes[key]['SHA256'],bytes=stat.st_size))
    dump(hashpath,hashes)
    dump(ROOT/'final_integrity.json',integrity)
    lines=['# 执行事实报告','',f"status: {status['status']}；stage: {status.get('stage')}",f'','experiment_directory: `'+str(ROOT)+'`','',f"最后心跳：{status.get('heartbeat')}；runner PID：{status.get('PID')}",'','本轮只记录执行事实和计算数值。开发清单为 ULong_development_32，32 视频 × QP0/5/9，共 96 点；未启动最终测试。','', '| 组 | committed_steps | step0 | step250 | step500 |','|---|---:|---:|---:|---:|']
    for g in GROUPS:lines.append(f"| {g} | {consumed[g]['windows']//4} | {counts[g]['0']}/96 | {counts[g]['250']}/96 | {counts[g]['500']}/96 |")
    lines+=['','训练计划：ULong/Vimeo = 1000/1000 个窗口，effective batch=4，QP0–9 各 200 个。Vimeo 为真实 7 帧，ULong 为真实连续 64 帧。实际覆盖见 `training_coverage.json`。','',f"计划 unique 内容窗口：{coverage(plan)['unique_windows']}/2000；重复率：{coverage(plan)['repeat_fraction']:.6%}。",'','| 审计 | 事实状态 |','|---|---|']
    for name,passed in required.items():lines.append(f'| {name} | '+('PASS' if passed else '尚未完成/见文件')+' |')
    lines+=['','码流、actual bpp、codec trace 仅在存在 B300 配对点时逐点核对；已配对点：'+str(len(paired))+'。原生 codec DPB 与候选生成历史分离；审计详情见 `audits/codec_audit.json`。','', '推理为共同 FP32 PyTorch 参考路径；I 帧使用共同原版 codec 的 FP16 输出。速度不代表优化 CUDA 部署性能。','']
    if aggregates:
        lines+=['| 数据集 | 模型/步数 | QP | n | bpp | PSNR | ΔPSNR vs B300 | LPIPS-Alex | ΔLPIPS | DISTS | ΔDISTS | FloLPIPS | ΔFloLPIPS |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(x):return 'NA' if x is None else f'{x:.7g}'
        for r in aggregates:lines.append('| '+' | '.join([r['dataset'],f"{r['model']}/{r['step']}",str(r['QP']),str(r['videos']),*[fmt(r.get(k)) for k in ['bpp','PSNR','delta_B300_PSNR','LPIPS_Alex','delta_B300_LPIPS_Alex','DISTS','delta_B300_DISTS','FloLPIPS','delta_B300_FloLPIPS']]])+' |')
    else:lines+=['开发指标尚未生成；相对 B300 变化目前无可报告数值。']
    lines+=['','文件：`protocol.json`、`source_manifest.json`、`per_video_metrics.csv`、`per_dataset_metrics.csv`、`paired_B300_metrics.csv`、`plots/index.json`、`video_manifest.json`、`final_integrity.json`。FFV1 视频在 `media/native`，1:1 裁剪在 `media/crops`，并排在 `media/panels`。未生成的项目以其完成标记为准。','',f"GitHub 分支：`{BRANCH}`；最近发布状态：{publication.get('status','尚未发布')}；commit URL：{publication.get('commit_URL','暂无')}。",'','四个固定开发片段及 ROI 复用 H 的 GT 选择记录：1 个 low_motion_high_texture、1 个 camera_motion、2 个 dynamic_occlusion_candidate。候选标签不等于人工逐帧确认；未依据候选模型输出重新选择案例，未另立 large_motion 标签。','', '失败/未完成项：']
    failures=list((ROOT/'audits').glob('*failure*.json'))+list((ROOT/'audits/cache_failures').glob('*.json'))+list(ROOT.glob('J*/training/failure.json'))
    lines.extend('- `'+str(p.relative_to(ROOT))+'`' for p in failures)
    lines.extend('- '+n+' 尚未完成。' for n,passed in required.items() if not passed)
    if not failures and all(required.values()):lines.append('- 无已记录执行失败。')
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
