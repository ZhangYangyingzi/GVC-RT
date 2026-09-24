# V1.1 输入来源与冻结条件审计

## 实际文件与模型对应

- V1 目录：`../gvcrt_latent_enhancement_v1`，仓库 commit `d0e32bfa3e8e282f9a77437c223c605858eea637`。
- 目标仓库、上级路径及实验目录没有找到适用 `AGENTS.md`。V1 代码、配置、manifest、B/C/D 起止训练日志、完整指标和已保存文件均已核对。
- Val6 的三个实际量化点均来自 `run_v1/stage_d/checkpoint.pt`；SHA256 `238b0740cebbc9012e0f3486e1ea178e1d944e8d47801d1216fa8743163e812a`。
- Train2 的实际量化结果与本轮 ZERO_START 对照来自 `run_v1/stage_c/checkpoint.pt`；SHA256 `8c5002d627ca8130a5d317321f5e924d32b58f7110e40dde34d0bf77ed975cce`。
- 两个 checkpoint 均含 `enhancement`（32 state keys）、`base_only`（12 keys）、`entropy`（1 key），逐项严格匹配现有类。`base_qp=0`，δ=[0.125,0.5,2]，`lambda_p=0.001`。该 LPIPS 权重确实在 C/D 日志中生效。
- 保留原生有效区域 1080×1920、底部复制填充 8 行、实际 25/29.97/30 fps、QP offset、1I+15P、reset=96、RGB [0,1] 度量。基础 codec FP16；G/增强 FP32，TF32 关闭。
- 同模型消融使用 Val6；有限直接 latent 优化只用 Train2。没有重新选择视频，也没有更新任何网络参数。

## 来源分类

A：双方固定共享；B：由基础码流计算；C：增强码流额外传输；D：仅发送端/诊断监督端可用。

| 输入/量 | 使用位置 | 来源与依赖 | 是否有仅源端信息 |
|---|---|---|---|
| D 的基础 `ell`，1×18×68×120 | `Synthesis.forward(u,ell)`，与插值 u 拼接 | **B**：基础文件量化/熵解码→feature→原 Latent Adapter | 否 |
| 增强整数，1×8×17×30 | 独立算术 decoder | **C**：实际 `.gle` 文件 payload | 正常 REAL 唯一潜在源内容载荷 |
| 标量 δ | `symbols.cuda().float()*delta` | **C**：GLE1 float32 字段；档位表为 A，本实验各点固定、无源自适应 | 档位本身不随源内容变化 |
| 档位 level | 格式校验/记录 | **C**：头中 uint8；不直接进入 D 或 G | 无 |
| `u_hat` | D 的增强输入 | 由 **C** 整数和 δ 派生；无其他项 | 仅可能来自整数内容 |
| D 的卷积参数、bias、SiLU、双线性插值规则 | `Synthesis` 内部 | **A**：冻结 checkpoint 和代码；插值 `align_corners=False`，17×30→68×120 | 否 |
| D 的均值/归一化/熵 scale | D 内部 | **不存在**：D 没有 GroupNorm/BatchNorm、去均值、额外 scale 或 hyperprior 输入 | 否 |
| `delta_ell=D(u_hat,ell)` | 加到 ell，非原地 | A 网络对 B/C 输入的计算结果 | 无隐藏输入 |
| `ell_final=ell+delta_ell` | G 的首个参数 | 由 A/B/C 派生 | 无隐藏输入 |
| QP 及逐帧 offset | 基础 codec 和 q_recon 选择 | QP 是 **B**（基础 NAL），offset 规则 **A**；同模式比较固定当前帧值 | 否 |
| `q_recon`，1×320×1×1 | G 的第二个参数，stage4 通道缩放 | **A+B**：固定模型 q_scale_recon 表与基础帧 QP；不传给 D | 否 |
| G 的 Conv/GroupNorm、gamma/beta Linear 权重 | De-tokenizer | **A**：完整基础 P checkpoint，冻结 | 否 |
| AdaptiveGroupNorm 的 latent 均值/std | `improved_model_gvcrt.py:292–304` | 从本次 `ell_final` 空间均值和 `sqrt(var+eps)` 计算，再过共享 gamma/beta Linear；因此由 **B/C** 派生 | 非额外传输量 |
| G 的其他 GroupNorm 统计 | G 内部 | 当前激活计算，A 规则；无逐帧外部运行均值或 BatchNorm 状态 | 否 |
| 基础 hyperprior z、y 的概率条件 | 在得到 ell 之前的基础解码 | **B**：基础流恢复，不额外传给增强 D/G | 否 |
| 基础 DPB 的 feature/RGB、I/P/reset | 基础预测，先于增强显示路径 | **B** 和 A 规则；D/G 不直接读取 DPB | 否 |
| 增强 logistic `log_scale` 及 softplus scale | 概率/CDF 构造 | **A**：共享增强 checkpoint，仅影响熵编码，不参与反量化 | 否 |
| 增强 logistic 均值 | 概率模型 | **A**：固定零均值，无每样本 mu | 否 |
| 16bit CDF 表 | 算术 decoder | 从共享 A 概率模型和 δ(C) 生成，表本身不传输 | 否 |
| CDF CRC32 | GLE1 校验 | **C**：重复指纹，由 A/C 的 CDF 计算；不作为网络条件 | 否 |
| GEC1 基础 SHA256 | 文件绑定校验 | C 位置携带 **B** 的摘要，不是新源内容 | 否 |
| shape/帧数/长度/版本/有效区域 | 文件解析、crop | A/B 的已知结构在 C 容器重复保存；不构成源图特征 | 否 |
| 原始 RGB x | 发送端 E；CONTINUOUS、质量监督、oracle | **D**；REAL/ZERO/WRONG 的 `render(g,D,ell,q,u)` API 不接受 x | 仅诊断或发送端可用 |
| 优化后的 per-image delta_latent | 单帧 oracle | **D**：由原图重建监督求得；没有编码，不可传输 | 是，仅诊断 |

## 零整数的确切语义

实际 V1 接收路径为：

```python
symbols, delta, level = entropy.decode(packet)
u_hat = symbols.cuda().float() * delta
ell_final = ell + enhancement.decoder(u_hat, ell)
x = generator(ell_final, q_recon)
```

**没有** `+mean`、`×learned_scale`、反归一化或额外源相关参数。因此全零整数必然得到全零 u_hat，经过双线性插值仍是全零。D 可以通过其训练后的 bias 和基础 ell 产生非零修正，所以全零不等于不调用 D。

V1.1 将分别核实每个 P 帧的 u、delta_ell、G 输入 latent、完整 RGB 和有效 RGB 的 `torch.equal`/最大差。ZERO 直接经过同 D/G，没有调用 V1 `correction=None` 或无增强文件旁路。UZERO 仅在实际零反量化与零 u 不同的情况下需要；当前实现没有这种差别。

## WRONG_SOURCE 与 CONTINUOUS

- donor 依 V1 Val6 manifest 顺序固定取下一视频，最后一个轮换至第一个；帧号相同。完整映射在 `donor_mapping.json`，模型执行前保存。
- 每个 donor 的三个实际文件先针对**它自己的基础流 hash**和正确共享 CDF 独立解码，再交换整数/反量化后的表示。当前视频 ell/q/DPB 不交换。
- 没有需连同 donor 替换的源相关均值/尺度。δ 与档位相同，CDF 模型固定共享。
- CONTINUOUS 是同 E 对相同 RGB 和 ell 的量化前 u；直接按 V1 model units 输入 D，不再除/乘 δ。额外检查 `round(u/δ)` 与该帧真实文件解出的符号一致。
- ZERO/WRONG_SOURCE/CONTINUOUS 均是干预，不获得虚构 kbps/bpp，不作为可传输 RD 点。

## 新路径检查范围

复用 V1 既有基础 codec 审计。Val6 仅从真实基础文件重建正常基础状态，和 V1 逐帧 hash 对齐；不重新基础编码或跑旧训练审计。所有干预都在相同基础状态之后的显示分支进行。校验基础 DPB、ell/q、所有共享权重未变化，REAL 输出与 V1 独立接收输出精确一致。

Oracle 固定首/中/末 P 帧（1/8/15，共 6 帧），每帧两个初始化×两个目标，各最多 150 Adam 更新，仅 delta_latent 在 optimizer 中。lambda_p=0.001，lr=1e−3，零初始化；不根据 Val6 调参。0/25/50/100/150 步保存完整指标与图像。源样本和 crop 位置均预先写入 `fixed_samples.json`/`config.json`。
