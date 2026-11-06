"""CPU-only factual report after the user's explicit experiment stop request."""
from common_s2 import *
import collections

METRICS = ['bpp', 'kbps', 'PSNR', 'MS_SSIM', 'LPIPS', 'DISTS', 'FloLPIPS',
           'WarpingError_GTFlow_PWC', 'WarpingError_ReconFlow_PWC']


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    env()
    verify_sources()
    import torch
    audit_dir = ROOT / 'audits/user_stop_20261106'
    termination = load(audit_dir / 'process_termination.json')
    assert termination['status'] == 'STOPPED_BY_USER' and termination['runner_lock_free']
    assert not termination['remaining_verified_processes']
    protocol = load(ROOT / 'protocol.json')
    before = load(audit_dir / 'state_before_stop.json')
    stopping = before['stopping_state.json']
    cp = torch.load(ROOT / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
    n = cp['committed_joint_updates']
    assert n == 1000 and cp['joint_optimizer_step'] == n and cp['next_cursor'] == 1500 + n
    assert cp['baseline_sha256'] == BASELINE_SHA and cp['boundary_buffers_cleared']
    assert cp['protocol_sha256'] == sha(ROOT/'protocol.json')
    assert cp['source_manifest_sha256'] == sha(ROOT/'source_manifest.json')
    assert len(cp['all_rank_RNG']) == 4
    assert [(g['group_name'],g['lr']) for g in cp['optimizer']['param_groups']] == [('Router',1e-4),('TCG',1e-5)]
    assert all(int(s['step']) == n for s in cp['optimizer']['state'].values())
    assert sum(t.numel() for t in cp['router_state'].values()) == 33889
    assert sum(t.numel() for t in cp['tcg_model_state'].values()) == 137760
    assert all(torch.isfinite(t).all() for name in ['router_state','tcg_model_state'] for t in cp[name].values())
    receipts = [load(p) for p in sorted((ROOT/'receipts').glob('update_*.json.gz'))]
    assert len(receipts) == n and receipts[-1] == cp['last_committed_receipt']
    plan = plan_rows()
    training = []
    frozen = set()
    for i, receipt in enumerate(receipts, 1):
        assert receipt['step'] == i and receipt['source_plan_cursor'] == 1500+i and receipt['rank_samples'] == 4*i
        assert receipt['rank_synchronized'] and receipt['only_Router_TCG_changed'] and receipt['frozen_unchanged']
        assert [r['sample_key'] for r in receipt['ranks']] == [digest(plan[4*(i-1)+j]) for j in range(4)]
        for r in receipt['ranks']:
            assert r['frozen_byte_exact'] and r['buffers_cleared'] and r['physical_GPU'] == 4+r['rank']
            assert all(math.isfinite(float(v)) for v in r['losses'].values())
            assert all(math.isfinite(float(v)) and v > 0 for v in r['group_gradient_norms_local'].values())
            assert all(v > 0 for v in r['group_parameter_changes_L2'].values())
            frozen.add(r['frozen_sha256'])
            training.append(dict(step=i,rank=r['rank'],video=r['video'],domain=r['domain'],sample_key=r['sample_key'],
                external_QP=r['external_qp'],seconds_before_commit=r['seconds_before_commit'],
                gradient_norm_local=r['local_gradient_norm'],gradient_norm_mean=r['mean_gradient_norm'],
                Router_gradient_norm_mean=r['group_gradient_norms_mean']['Router'],TCG_gradient_norm_mean=r['group_gradient_norms_mean']['TCG'],
                Router_change_L2=r['group_parameter_changes_L2']['Router'],TCG_change_L2=r['group_parameter_changes_L2']['TCG'],**r['losses']))
    assert len(frozen) == 1 and cp['frozen_sha256'] in frozen
    write_csv(ROOT/'training_history.csv',training)
    coverage = {}
    for domain in ['uvg','ulong']:
        rows = [r for r in plan[:4*n] if r['domain'] == domain]
        ids = {r['video'] for r in rows}
        windows = {(r['video'],tuple(r['source_frame_indices'])) for r in rows}
        exact = {(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y'],r['external_qp']) for r in rows}
        coverage[domain] = dict(samples=len(rows),distinct_identities=len(ids),distinct_windows=len(windows),
            exact_combinations=len(exact),exact_repeats=len(rows)-len(exact),QP_counts=dict(collections.Counter(str(r['external_qp']) for r in rows)),
            pool_size=8470 if domain=='ulong' else None,identity_coverage=len(ids)/8470 if domain=='ulong' else None)
    dump(ROOT/'training_coverage.json',dict(status='STOPPED_BY_USER',committed_joint_updates=n,rank_samples=4*n,domains=coverage))
    checkpoints = [dict(filename=p.name,size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'checkpoints').glob('*.pt'))]
    dump(ROOT/'checkpoints_manifest.json',dict(status='STOPPED_BY_USER',committed_joint_updates=n,read_only_audit=True,files=checkpoints))
    assert sha(ROOT/'checkpoints/latest.pt') == sha(ROOT/'checkpoints/step_1000.pt')
    batches = [('B1500_matched_FP32','development_matched_FP32',None),('joint_step0','development_joint_0000',0)]
    batches += [('joint_step'+str(c['step']),'development_joint_%04d'%c['step'],c['step']) for c in stopping['checks']]
    all_metrics = []; summaries = []; by_qp = []
    baseline_q = stopping['matched_baseline_F_q']
    for label,batch,step in batches:
        root = ROOT/'evaluation'/batch
        integrity = load(root/'integrity.json')
        assert integrity['status'] == 'PASS' and integrity['points'] == 96
        rows = read_csv(root/'metrics.csv')
        assert len(rows) == 96 and len({(r['sequence'],int(r['QP'])) for r in rows}) == 96
        assert all(math.isfinite(float(r[k])) for r in rows for k in METRICS)
        for r in rows: all_metrics.append(dict(**r,candidate_label=label,committed_joint_step=step,batch_complete=True))
        values = {k:sum(float(r[k]) for r in rows)/96 for k in METRICS}
        score = sum((sum(float(r['FloLPIPS']) for r in rows if int(r['QP'])==q)/32)/baseline_q[str(q)] for q in [0,5,9])/3
        summaries.append(dict(candidate=label,committed_joint_step=step,points=96,normalized_dev_score=score,**values))
        for q in [0,5,9]:
            subset = [r for r in rows if int(r['QP']) == q]
            assert len(subset) == 32
            by_qp.append(dict(candidate=label,committed_joint_step=step,QP=q,videos=32,**{k:sum(float(r[k]) for r in subset)/32 for k in METRICS}))
    write_csv(ROOT/'development_metrics.csv',all_metrics)
    write_csv(ROOT/'development_summary_user_stop.csv',summaries)
    write_csv(ROOT/'development_by_QP_user_stop.csv',by_qp)
    partial_root = ROOT/'evaluation/development_joint_1000'
    partial = [load(p) for p in sorted((partial_root/'points').glob('*.json'))]
    assert len(partial) < 96 and not (partial_root/'integrity.json').exists()
    assert all(p['status']=='PASS' and p['joint_checkpoint_sha256']==sha(ROOT/'checkpoints/step_1000.pt') for p in partial)
    if partial:write_csv(ROOT/'development_step1000_partial.csv',[dict(**p['result'],batch_complete=False,excluded_from_selection=True) for p in partial])
    assert not (ROOT/'selection.json').exists() and not list((ROOT/'evaluation').glob('final_*'))
    now = time.time();utc = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now))
    stopped = dict(status='STOPPED_BY_USER',current_stage='user_requested_stop',stop_reason='USER_REQUEST',
        time=now,committed_joint_updates=n,next_source_cursor=1500+n,runner_running=False,training_running=False,evaluation_running=False,
        last_completed_validation_step=750,incomplete_validation=dict(step=1000,completed_points=len(partial),expected_points=96,counted_for_patience=False),
        final_test_completed_points=0,experiment_complete=False,automatic_resume_allowed=False,report_path='REPORT.md')
    dump(ROOT/'run_status.json',stopped)
    dump(ROOT/'training_status.json',dict(status='STOPPED_BY_USER',stop_reason='USER_REQUEST',committed_joint_updates=n,next_cursor=1500+n,
        latest_sha256=sha(ROOT/'checkpoints/latest.pt'),rank_samples=4*n,physical_GPUs=[4,5,6,7],time=now))
    dump(ROOT/'active_processes.json',dict(status='STOPPED_BY_USER',stage='user_requested_stop',children=[],runner_PID=None,previous_runner_PID=termination['scheduler'],time=now))
    dump(ROOT/'stopping_state.json',dict(**{k:v for k,v in stopping.items() if k!='status'},status='STOPPED_BY_USER',previous_status=stopping['status'],
        stop_reason='USER_REQUEST',stopped_at_committed_joint_step=n,incomplete_validation_not_counted=True,time=now))
    dump(ROOT/'final_integrity.json',dict(status='NOT_COMPLETED_USER_STOP',experiment_complete=False,
        stopped_artifact_verification='PASS',committed_joint_updates=n,rank_samples=4*n,only_Router_TCG_changed=True,frozen_components_unchanged=True,
        receipt_sequence_complete=True,checkpoint_and_last_receipt_aligned=True,all_rank_RNG=4,protocol_and_source_hashes_unchanged=True,
        completed_development_points=96*len(batches),incomplete_step1000_points=len(partial),final_test_completed_points=0,final_test_expected_points=135,
        final_selection_frozen=False,final_test_and_RD_figures_not_run=True,stop_reason='USER_REQUEST',time=now))
    dump(audit_dir/'artifact_checks.json',dict(status='PASS',scope='Stopped artifacts only; not full experiment acceptance',time=now,
        protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),plan_index_sha256=sha(ROOT/'plans/index.json'),
        checkpoints=checkpoints,receipt_count=n,rank_samples=4*n,groups=coverage,report_generator_sha256=sha(Path(__file__))))
    dump(ROOT/'stopped_artifacts/training_receipts.json.gz',receipts)
    for label,batch,step in batches + [('joint_step1000_partial','development_joint_1000',1000)]:
        dump(ROOT/f'stopped_artifacts/{batch}_points.json.gz',[load(p) for p in sorted((ROOT/'evaluation'/batch/'points').glob('*.json'))])
    report = [
        '# TCG 历史分支空间门控 + Router/TCG 联合训练：用户停止报告',
        f'快照时间（UTC）：{utc}。状态：**STOPPED_BY_USER**。',
        '本报告只记录配置、已完成的训练和评测数值。用户明确要求停止；未将此次停止记为自动 patience 早停，也未完成原定全部测试。',
        '## 执行状态',
        f'- 已提交联合 optimizer updates：**{n}**；有效 batch=4；已处理 rank samples：**{4*n}**。',
        '- 物理 GPU：4、5、6、7。调度器及其四个评测子进程已按核验后的进程树停止，生命周期锁已释放。没有终止其他实验。',
        '- 最后完整开发集验证节点：750 步。250/500/750 步分别完成96/96点，patience计数依次为1/2/3。',
        f'- 1000步开发集评测已完成**{len(partial)}/96**点后按用户要求停止；该不完整批次不参与完整开发分数、checkpoint选择或patience计数。',
        '- 最终测试：**0/135**。没有冻结最终模型选择，没有生成最终测试R-D图，不能将本报告视为完整实验验收通过。',
        '- 已保存的checkpoint、模型源码、原协议、数据计划及既有原始结果均保留。不会自动恢复；如未来继续，需要用户重新授权。',
        '## 固定配置',
        f'- 协议版本：`{protocol["revision"]}`。',
        f'- 底座：B+1500，初始化TCG累计6500步，SHA256：`{BASELINE_SHA}`。',
        '- 全新Router，seed=20261106，初始空间M=0.2；门控仅作用于previous_adapter输出，位于concat之前；current_adapter保留，没有外层residual乘法。',
        '- 训练参数：Router33889 + tcg_low/tcg_mid/tcg_out137760，共171649。其余codec、Bridge、generator及质量网络冻结。',
        '- Router/TCG内部FP32，TCG输出回到输入dtype；缓存为detach的pre-residual current；Router历史为detach的codec decoded feature；无BPTT。',
        '- 新AdamW：Router lr=1e-4、TCG lr=1e-5；betas=(0.9,0.999)、eps=1e-8、weight_decay=1e-4、联合global clip=1.0。',
        '- 数据沿用封存B计划，source cursor从1500开始，停止时next cursor=2500；每更新2UVG+2ULong；64帧、256crop，I0/P1–48 warmup/P49–63监督。',
        '- Loss：LPIPS + 0.5 DISTS + 0.1383589248954108 RGB_L1 + 0.01 RAFT_motion + beta(q) Rate + 0.01 proxy；沿用原VJP4096缩放及一次unscale。',
        '- 原定上限3000、每250验证、patience=4、min_delta=0.001。归一化基线为独立B+1500 matched FP32-TCG推理，不使用历史FP16数值作分母。',
        '## 已完成开发集结果',
        '每个完整批次为32条development视频×QP0/5/9，共96点。下表原始指标按视频与QP等权平均；归一化分数按每QP视频均值除以同QP基线均值后，再对三个QP等权平均。LPIPS、DISTS、FloLPIPS及两种warp误差越低越好；PSNR、MS-SSIM越高越好。',
        '|状态|点数|归一化开发FloLPIPS分数↓|bpp|PSNR↑|MS-SSIM↑|LPIPS↓|DISTS↓|FloLPIPS↓|GT-flow warp↓|Recon-flow warp↓|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        report.append('|'+ '|'.join([row['candidate'],str(row['points'])]+[f'{row[k]:.12g}' for k in ['normalized_dev_score','bpp','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']])+'|')
    report += ['\n已完成验证的训练checkpoint中，记录的raw-best为250步；该记录截至最后完整验证节点750步。最终测试选点流程未执行。',
        '## 按QP记录',
        '|状态|QP|视频数|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|GT-flow warp|Recon-flow warp|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in by_qp:
        report.append('|'+ '|'.join([row['candidate'],str(row['QP']),str(row['videos'])]+[f'{row[k]:.12g}' for k in ['bpp','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']])+'|')
    report += ['\n## 训练与工程核验',
        '- 已核对1000条连续提交回执与封存计划，包含4000个rank样本；无重复提交、跳步或漏记。latest与step_1000文件SHA一致，optimizer步数为1000，保留四rank RNG。',
        '- 全部提交回执记录Router/TCG梯度有限且非零、参数组有更新、四rank同步、冻结byte检查和buffer reset通过。完整源码/协议/计划哈希核验保持通过。',
        '- 工程检查：分支门控与FP32边界、M0匹配基线、非均匀空间mask、梯度等价、DDP smoke、真实RANS码流不变量等已有PASS记录；工程smoke未计入正式1000次更新。',
        '- 工程probe的码流一致性不替代未完成的全量最终测试。`final_integrity.json`明确标为NOT_COMPLETED_USER_STOP。',
        '## 实际数据覆盖', '```json',json.dumps(coverage,ensure_ascii=False,indent=2),'```',
        '## 文件',
        '- `development_metrics.csv`：480个已完整验收的开发集点；`development_summary_user_stop.csv`、`development_by_QP_user_stop.csv`：汇总。',
        '- `development_step1000_partial.csv`：中止批次的已落盘点，显式标记不完整及排除选点；原始逐帧、逐转移数据保留在`evaluation/`。',
        '- `training_history.csv`：4000条rank训练记录；`receipts/`：1000条原子回执；`training_coverage.json`：覆盖。',
        '- `checkpoints_manifest.json`：checkpoint文件名、大小与SHA256；权重仅保留本地。',
        '- `audits/user_stop_20261106/`：停止前状态、进程识别、定向终止、停止后的产物核验。',
        '- `protocol.json`、`source_manifest.json`、`plans/index.json`保持原始冻结内容。',
        '- GitHub代码提交已发布；本次停止报告和小型记录的上传状态由`publication_status.json`独立记录。',
        '\n本报告不包含科研效果、收敛性或后续研究方向判断。']
    import re
    report_text = re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)', r'\1\n', '\n\n'.join(report)+'\n')
    (ROOT/'REPORT.md').write_text(report_text)
    (ROOT/'REPORT_USER_STOP_20261106.md').write_text((ROOT/'REPORT.md').read_text())
    dump(audit_dir/'report_manifest.json',dict(status='WRITTEN',time=time.time(),files=[dict(path=n,sha256=sha(ROOT/n),size_bytes=(ROOT/n).stat().st_size) for n in ['REPORT.md','REPORT_USER_STOP_20261106.md','development_metrics.csv','development_summary_user_stop.csv','development_by_QP_user_stop.csv','development_step1000_partial.csv','training_history.csv','training_coverage.json','checkpoints_manifest.json','final_integrity.json']]))
    print(json.dumps(dict(status='STOPPED_BY_USER',committed_joint_updates=n,report_path=str(ROOT/'REPORT_USER_STOP_20261106.md'),complete_development_points=len(all_metrics),partial_step1000_points=len(partial),final_test_points=0),ensure_ascii=False))


if __name__ == '__main__':
    main()
