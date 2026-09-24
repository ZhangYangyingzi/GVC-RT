# GVC-RT Latent Enhancement V1.2 — Oracle Residual Codec

本轮只训练独立的轻量残差 codec。固定基线为 **V1 阶段 D 的同模型 ZERO**，不是另训 base-only C。源端允许逐帧150步 latent 优化，属于非实时研究原型。

## 一条复现命令

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_latent_enhancement_v1_2_oracle_residual_codec/reproduce.py --output reproduction_01
```

输出目录必须全新。脚本检查固定 GPU UUID 是否空闲，使用原 `gvc-rt` 环境，不下载资源，不覆盖 V1/V1.1 或原 checkpoint。完整运行包括教师恢复、直接量化真实编码、阶段A/B、门槛控制的完整Train2/Val6、指标聚合、独立接收检查和可视化。

也可按阶段执行，保留前阶段结果：

```bash
python run.py --output results_v1 --phase prepare
python run.py --output results_v1 --phase direct
python run.py --output results_v1 --phase train
```

同一阶段不得重复覆盖。阶段门槛不通过时停止扩展。A最多1000步，B最多1500步，训练/共享统计仅用固定六帧。Val6只在完整Train2真实信息门槛通过后使用。

## 关键实现

- `config.json`：运行前固定的教师选择、归一化规则、几何网格、一个瓶颈、损失和门槛。
- `support.py`：只读共享资源；阶段D ZERO、阶段C独立对照及原G严格加载/冻结；原V1指标。
- `teachers.py`：恢复旧起点+已保存delta，计算真实ell_oracle，再减本轮ell_c；只用Train6校准。必要时在评估发送端逐帧优化临时latent，完整计时，不更新网络。
- `model.py`：E(r_star/scale,ell_c)，D(u,ell_c)−D(0,ell_c)，固定反归一化。
- `bitstream.py`：ORC2真实算术码流，绑定基础/共享模型SHA256；全零P仅1byte标志，无payload；严格整数范围和CRC。
- `receiver.py`：独立正常接收进程，禁止源图、教师、oracle和基础训练缓存访问；只从基础文件重建状态。
- `experiment.py`：阶段训练、真实编码、合法解码后的wrong-source干预、完整I/P指标与预算。
- `postprocess.py`：全部实际点、逐序列预算偏差、匹配精度的真实重叠区间比较、分开的编码时间。
- `verify.py`：结果账目、原始资源不变、独立接收证据、102byte全零序列及溢出拒绝测试。
- `visualize.py`：对应视频的固定帧/裁剪和真实采样RD图；不给诊断干预画RD点。
- `protocol.md`：精确张量、基线定义、训练范围、码流格式和计时/比较口径。

## 结果位置

`results_v1/` 保存：

- `teacher_selection.json`、`teachers_train6/*.pt`：ell_base、ell_c、ell_oracle、r_star、q_recon、有效区域、参考来源和选择证据。
- `calibration.json`、`direct_shared.pt`、`learned_quantization_calibration.json`：仅Train6拟合的共享量。
- `stage_A/`、`stage_B/`：固定间隔/最终checkpoint、逐步损失和梯度、阶段门槛。
- 各方法目录：实际 `.orc`、独立 `_receiver.pt`、`frames.csv`、`summary.csv/json`、真实字节/耗时 `_coding.json`、对应 `.mkv`。
- `sender_oracles/*.pt`：评估发送端临时优化目标，不对正常接收端开放。
- `*_sender_optimization_timing.csv`：所有新逐帧优化完整耗时；历史六帧的实际优化耗时单独记录，缓存查找不冒充实时编码。
- `aggregate_metrics.csv`、`budget_deviations.csv`、`matched_rate_comparisons.csv`、`coding_timing.csv`、`verification.json`。

Train6初期是每段仅3/15个P帧有修正的**稀疏诊断码流**；完整Train2/Val6另外命名并逐帧增强。TARGET_P、P和I/P指标不能混用。教师/连续/错配是诊断，不是正式RD点。源信息利用、真实码率效率和泛化结论分别见 `report.md`。
