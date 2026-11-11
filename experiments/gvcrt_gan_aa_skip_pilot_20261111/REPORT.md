# 执行事实报告

status: RUNNING；stage: training_and_shared_prefetch

experiment_directory: `/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/gvcrt_gan_aa_skip_pilot_20261111/experiments/gvcrt_gan_aa_skip_pilot_20261111`

最后心跳：1794437276.8312292；runner PID：3561209

本轮只记录执行事实和计算数值。开发清单为 ULong_development_32，32 视频 × QP0/5/9，共 96 点；未启动最终测试。

| 组 | committed_steps | step0 | step250 | step500 |
|---|---:|---:|---:|---:|
| J0 | 250 | 2/96 | 2/96 | 0/96 |
| J1 | 262 | 2/96 | 2/96 | 0/96 |
| J2 | 262 | 2/96 | 2/96 | 0/96 |
| J3 | 262 | 2/96 | 2/96 | 0/96 |

训练计划：ULong/Vimeo = 1000/1000 个窗口，effective batch=4，QP0–9 各 200 个。Vimeo 为真实 7 帧，ULong 为真实连续 64 帧。实际覆盖见 `training_coverage.json`。

计划 unique 内容窗口：2000/2000；重复率：0.000000%。

| 审计 | 事实状态 |
|---|---|
| audits/preflight.json | PASS |
| audits/filter_audit.json | PASS |
| audits/gradient_audit.json | PASS |
| audits/codec_audit.json | PASS |
| gradient_calibration.json | PASS |
| evaluation/completion.json | 尚未完成/见文件 |
| video_manifest.json | 尚未完成/见文件 |

码流、actual bpp、codec trace 仅在存在 B300 配对点时逐点核对；已配对点：20。原生 codec DPB 与候选生成历史分离；审计详情见 `audits/codec_audit.json`。

推理为共同 FP32 PyTorch 参考路径；I 帧使用共同原版 codec 的 FP16 输出。速度不代表优化 CUDA 部署性能。

| 数据集 | 模型/步数 | QP | n | bpp | PSNR | ΔPSNR vs B300 | LPIPS-Alex | ΔLPIPS | DISTS | ΔDISTS | FloLPIPS | ΔFloLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ULong_development_32 | B300/0 | 0 | 1 | 0.0009754171 | 22.27136 | 0 | 0.1804595 | 0 | 0.0926469 | 0 | 0.1164239 | 0 |
| ULong_development_32 | B300/0 | 5 | 1 | 0.004021026 | 24.10875 | 0 | 0.1196081 | 0 | 0.05928857 | 0 | 0.09197299 | 0 |
| ULong_development_32 | J0/0 | 0 | 1 | 0.0009754171 | 22.27136 | 0 | 0.1804595 | 0 | 0.0926469 | 0 | 0.1164239 | 0 |
| ULong_development_32 | J0/0 | 5 | 1 | 0.004021026 | 24.10875 | 0 | 0.1196081 | 0 | 0.05928857 | 0 | 0.09197299 | 0 |
| ULong_development_32 | J0/250 | 0 | 1 | 0.0009754171 | 22.36046 | 0.08910117 | 0.1818817 | 0.001422206 | 0.09349022 | 0.0008433191 | 0.1185214 | 0.002097505 |
| ULong_development_32 | J0/250 | 5 | 1 | 0.004021026 | 24.13888 | 0.03013191 | 0.1196589 | 5.077105e-05 | 0.05929338 | 4.810281e-06 | 0.09117756 | -0.0007954258 |
| ULong_development_32 | J1/0 | 0 | 1 | 0.0009754171 | 22.27136 | 0 | 0.1804595 | 0 | 0.0926469 | 0 | 0.1164239 | 0 |
| ULong_development_32 | J1/0 | 5 | 1 | 0.004021026 | 24.10875 | 0 | 0.1196081 | 0 | 0.05928857 | 0 | 0.09197299 | 0 |
| ULong_development_32 | J1/250 | 0 | 1 | 0.0009754171 | 22.29252 | 0.02116234 | 0.1870924 | 0.006632918 | 0.106489 | 0.01384206 | 0.1111683 | -0.005255626 |
| ULong_development_32 | J1/250 | 5 | 1 | 0.004021026 | 24.09697 | -0.01177799 | 0.1246937 | 0.005085621 | 0.06430718 | 0.005018615 | 0.093695 | 0.001722013 |
| ULong_development_32 | J2/0 | 0 | 1 | 0.0009754171 | 18.89731 | -3.374047 | 0.2950566 | 0.1145971 | 0.1937246 | 0.1010777 | 0.2563958 | 0.1399719 |
| ULong_development_32 | J2/0 | 5 | 1 | 0.004021026 | 20.01018 | -4.098573 | 0.25972 | 0.1401119 | 0.1877046 | 0.128416 | 0.189697 | 0.097724 |
| ULong_development_32 | J2/250 | 0 | 1 | 0.0009754171 | 20.09296 | -2.178395 | 0.2508855 | 0.07042597 | 0.1550932 | 0.06244629 | 0.209141 | 0.09271705 |
| ULong_development_32 | J2/250 | 5 | 1 | 0.004021026 | 21.00571 | -3.103043 | 0.206951 | 0.08734293 | 0.1316831 | 0.07239457 | 0.1495378 | 0.05756484 |
| ULong_development_32 | J3/0 | 0 | 1 | 0.0009754171 | 18.89731 | -3.374047 | 0.2950566 | 0.1145971 | 0.1937246 | 0.1010777 | 0.2563958 | 0.1399719 |
| ULong_development_32 | J3/0 | 5 | 1 | 0.004021026 | 20.01018 | -4.098573 | 0.25972 | 0.1401119 | 0.1877046 | 0.128416 | 0.189697 | 0.097724 |
| ULong_development_32 | J3/250 | 0 | 1 | 0.0009754171 | 19.82032 | -2.451032 | 0.2738746 | 0.09341507 | 0.1667286 | 0.07408166 | 0.226426 | 0.1100021 |
| ULong_development_32 | J3/250 | 5 | 1 | 0.004021026 | 20.94859 | -3.16016 | 0.2189729 | 0.09936477 | 0.1367399 | 0.07745136 | 0.1632304 | 0.07125738 |
| ULong_development_32 | Original/0 | 0 | 1 | 0.0009754171 | 22.14718 | -0.1241784 | 0.1924865 | 0.01202696 | 0.09886682 | 0.00621992 | 0.1434714 | 0.02704753 |
| ULong_development_32 | Original/0 | 5 | 1 | 0.004021026 | 23.79088 | -0.3178732 | 0.1334872 | 0.01387912 | 0.06651422 | 0.007225649 | 0.09827264 | 0.006299646 |

文件：`protocol.json`、`source_manifest.json`、`per_video_metrics.csv`、`per_dataset_metrics.csv`、`paired_B300_metrics.csv`、`plots/index.json`、`video_manifest.json`、`final_integrity.json`。FFV1 视频在 `media/native`，1:1 裁剪在 `media/crops`，并排在 `media/panels`。未生成的项目以其完成标记为准。

GitHub 分支：`codex/gvcrt-gan-aa-skip-pilot-20261111`；最近发布状态：REMOTE_VERIFIED；commit URL：https://github.com/ZhangYangyingzi/GVC-RT/commit/62bea9c84be404901c07e4abc08cd38baa69f109。

四个固定开发片段及 ROI 复用 H 的 GT 选择记录：1 个 low_motion_high_texture、1 个 camera_motion、2 个 dynamic_occlusion_candidate。候选标签不等于人工逐帧确认；未依据候选模型输出重新选择案例，未另立 large_motion 标签。

失败/未完成项：
- `audits/runner_failure.json`
- evaluation/completion.json 尚未完成。
- video_manifest.json 尚未完成。
