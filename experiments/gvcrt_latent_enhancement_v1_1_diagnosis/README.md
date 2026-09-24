# GVC-RT Latent Enhancement V1.1 Diagnosis

本实验只读取 V1 的模型与数据，执行同模型输入消融、全零码流开销拆解和固定六帧直接 latent 优化。**不训练任何增强编码器、解码器、熵模型或生成器参数。**

## 一条复现命令

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_latent_enhancement_v1_1_diagnosis/reproduce.py --output reproduction_01
```

使用预先固定的 `config.json`、`manifest.json`、`fixed_samples.json` 与 `donor_mapping.json`。启动前验证 GPU 0/固定 UUID 空闲和输入 SHA256；输出目录必须全新。不会下载模型或数据，不会调用 V1 训练入口。

## 实现与产物

- `input_audit.md`：D/G 的全部输入、A/B/C/D 来源类别、零整数反量化、参考状态和正常接收端边界。
- `input_audit.json`：原始文件、模型与 V1 文件大小/mtime 快照；`source_identity_checks.json`：128 张使用的源帧与 V1 manifest hash 对齐。
- `common.py`：只读复用 V1 的模型类、生成器与度量函数；禁止给 V1 写入 Python bytecode。
- `rate_analysis.py`：解析既有 GEC1/GLE1 文件，逐 bit 记录 V1 算术编码过程，并核对重放 payload 字节完全一致。没有写新协议或改变概率模型。
- `ablation.py`：Val6 三档 REAL/ZERO/WRONG_SOURCE/CONTINUOUS；保留原版、同精度参考、base-only；输出 P-only 和完整 I/P 指标。只对同源帧中完全相等的 RGB 张量复用度量结果，各模式的网络路径仍实际运行。
- `oracle.py`：固定 Train2 帧 1/8/15，每帧 2 个起点×2 个目标×150 Adam 更新；optimizer 只包含当前 delta_latent；保存 0/25/50/100/150 步完整指标、delta、图像与固定 crop。
- `postprocess.py`：配对差值与一致性、完整 oracle Pareto 集、曲线、质量组合图及源/基线/四种 oracle 的图像对比。
- `results_v1/source_ablation.csv`、`source_ablation_frames.csv`：逐视频/总体 P/IP 与逐帧的完整输入消融。
- `results_v1/exact_equality.csv`：u、delta_ell、G 输入、原始及有效 RGB 的 `torch.equal` 与最大差。
- `results_v1/rate_breakdown.csv`、`rate_frames.csv`、`zero_probabilities.json`：真实开销与零概率。
- `results_v1/oracle_metrics.csv`、`oracle_references.csv`、`oracle_pareto.csv`：全部 checkpoint 步数与参照质量，不只保留最佳 PSNR。
- `results_v1/ablation_videos/`：对应的 16 帧 FFV1 视频，I 帧原样保留，使用各视频真实帧率。
- `results_v1/oracle/`、`plots/`：oracle 原图、完整输出、固定 crop、delta 张量、优化曲线和对比图。
- `report.md`：实测结论；`results_v1/main_tables.md` 为机器生成主表。

ZERO、WRONG_SOURCE、CONTINUOUS 与 oracle 都是诊断干预，不能作为正式 RD 点。Oracle 是有限预算单帧可达参考，不是严格上界，也不支持时间稳定性推断。全零标志的潜在节省只是估算，没有实现码流协议修改。
