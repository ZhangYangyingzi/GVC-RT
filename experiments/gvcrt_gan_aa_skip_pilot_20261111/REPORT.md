# 执行事实报告

status: RUNNING；stage: GAN_warmup_and_gradient_calibration

experiment_directory: `/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/gvcrt_gan_aa_skip_pilot_20261111/experiments/gvcrt_gan_aa_skip_pilot_20261111`

最后心跳：1794415490.5891428；runner PID：3525009

本轮只记录执行事实和计算数值。开发清单为 ULong_development_32，32 视频 × QP0/5/9，共 96 点；未启动最终测试。

| 组 | committed_steps | step0 | step250 | step500 |
|---|---:|---:|---:|---:|
| J0 | 0 | 0/96 | 0/96 | 0/96 |
| J1 | 0 | 0/96 | 0/96 | 0/96 |
| J2 | 0 | 0/96 | 0/96 | 0/96 |
| J3 | 0 | 0/96 | 0/96 | 0/96 |

训练计划：ULong/Vimeo = 1000/1000 个窗口，effective batch=4，QP0–9 各 200 个。Vimeo 为真实 7 帧，ULong 为真实连续 64 帧。实际覆盖见 `training_coverage.json`。

计划 unique 内容窗口：2000/2000；重复率：0.000000%。

| 审计 | 事实状态 |
|---|---|
| audits/preflight.json | PASS |
| audits/filter_audit.json | PASS |
| audits/gradient_audit.json | PASS |
| audits/codec_audit.json | PASS |
| gradient_calibration.json | 尚未完成/见文件 |
| evaluation/completion.json | 尚未完成/见文件 |
| video_manifest.json | 尚未完成/见文件 |

码流、actual bpp、codec trace 仅在存在 B300 配对点时逐点核对；已配对点：0。原生 codec DPB 与候选生成历史分离；审计详情见 `audits/codec_audit.json`。

推理为共同 FP32 PyTorch 参考路径；I 帧使用共同原版 codec 的 FP16 输出。速度不代表优化 CUDA 部署性能。

开发指标尚未生成；相对 B300 变化目前无可报告数值。

文件：`protocol.json`、`source_manifest.json`、`per_video_metrics.csv`、`per_dataset_metrics.csv`、`paired_B300_metrics.csv`、`plots/index.json`、`video_manifest.json`、`final_integrity.json`。FFV1 视频在 `media/native`，1:1 裁剪在 `media/crops`，并排在 `media/panels`。未生成的项目以其完成标记为准。

GitHub 分支：`codex/gvcrt-gan-aa-skip-pilot-20261111`；最近发布状态：尚未发布；commit URL：暂无。

四个固定开发片段及 ROI 复用 H 的 GT 选择记录。其现有标签未明确标注遮挡类别；未据候选输出重新挑选案例。

失败/未完成项：
- `audits/runner_failure.json`
- gradient_calibration.json 尚未完成。
- evaluation/completion.json 尚未完成。
- video_manifest.json 尚未完成。
