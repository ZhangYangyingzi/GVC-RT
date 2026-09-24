# V1.3 接口、冻结范围与预定规则

## 三个不同模型

1. **V1阶段D旧模型**：`gvcrt_latent_enhancement_v1/run_v1/stage_d/checkpoint.pt` 的旧 enhancement.decoder，仅用于 `ell_c=ell_base+D_V1_D(0,ell_base)`。不是独立base-only C。
2. **V1.2 D_enh**：`gvcrt_latent_enhancement_v1_2_oracle_residual_codec/results_v1/stage_B/checkpoint_1500.pt` 中 `model.decoder.*`。本轮优化的是它的输入，不是它的参数。
3. **原G**：原P checkpoint严格加载的FP32生成器；基础流仍是V1原FP16codec，I保持原版。

原E、D_enh、熵模型、residual_scale、旧ZERO路径、G、基础codec全部冻结。没有新网络，也没有逐样本完整latent优化。

## 真实表示域

V1.2的 `ResidualCodec.analyze` 输出 `u:[1,8,17,30]`，4080个标量，是物理模型单位的量化前码字。E输入是 `cat(r_star/residual_scale,ell_c)`。

固定l3的delta为 **9.042649269104004**。原量化为 `s=round(u/delta)`；D_enh接收 `u_hat=s.float()*delta`。每个码率等级仅改delta/记录的等级标识，不改变D/G条件。

实际输出：

```
pred_norm = D_enh(u_hat,ell_c) - D_enh(0,ell_c)
r_hat = pred_norm * frozen_residual_scale
x_hat = G(ell_c+r_hat,q_recon)
```

`residual_scale`是V1.2由六训练帧拟合的18通道固定RMS，无去均值/加回均值。熵模型的logistic scale只影响概率，不改变反量化。因此整数零、量化域z=0及物理u_hat=0一致；r_hat精确零。D内部插值规则、网络与归一化都保持原样。

优化器前向缓存 `D_enh(0,ell_c)` 仅因模型和ell_c固定而可复用；准备检查确认缓存表达式与V1.2 `synthesize` 精确一致。

## 输入来源

| 输入 | 来源 |
|---|---|
| 旧ZERO权重、新D_enh权重、G、残差RMS、熵参数、l3固定设置 | 固定共享模型/配置 |
| ell_base、ell_c、q_recon、原DPB、I/P和reset | 基础流及共享模型可计算 |
| s整数、ORC2形状/档位/校验字段 | 增强流；ORC2不改变 |
| x源RGB | 只在发送端优化/度量；不进入独立接收端 |
| r_star/完整教师 | 仅用于ENCODER_START及独立参考；ZERO_START编码算法不需要它们 |

正常接收程序禁止打开源图片、教师、优化检查点、码字缓存和基础训练`.pt`缓存，只从真实基础文件恢复状态。显示增强不写回基础参考。

## 分组和初始化

- network_train6：两段帧1、8、15，是真正参与V1.2网络训练的六帧。
- Train2_rest24：同两段其余24个P帧，未参与增强网络训练。
- Val6_90P：固定六段90个P帧，各保留完整1I+15P。

A在物理u域优化；两种起点为当前E的真实u、全零u。学习率由六帧码字RMS决定：`lr_u=0.02*median(RMS(u))`，实测0.1377329922。

B在dimensionless `z=u/delta` 域优化，学习率`lr_z=lr_u/delta`，实测0.01523148671。前向使用 `round(z).detach()+(z-z.detach())`，确保数值严格等于整数，同时STE梯度为1。D接收其乘delta的结果。所有B运行显式传 `quantized=True`，不从阶段名称推断。

ZERO_START的第一次完整反向已经验证非零梯度；优化前向从不走ORC2的全零跳过分支。ORC2零标志仅在硬编码/评估时使用。

## Q1规则

每种初始化×六帧150次Adam更新，记录0/25/50/100/150。每次只优化4080码值；不优化146880维生成latent。

主输出为记录点中最小 `MSE+0.001LPIPS`，包含step0，平局取更早步。六帧平均选中L_image较低的全局初始化用于扩展，精确平局选ZERO。平均PSNR相对当前连续E输出达到0.2dB才扩展其余24P和Val6；LPIPS单独报告，不把PSNR门槛当联合质量标准。

## Q2规则与比特单位

Q2无条件完成六帧校准，不受Q1成败阻断。编码初始化预先固定ZERO_START，允许免去完整教师优化与E前向；这些参考仍会为评价重建，但不属于该编码算法的必需输入。

`R_est_bpp = frozen_entropy.bits(STE(s),delta)/(1080*1920)`。它是各通道零均值logistic整数区间概率的可微估计，没有重新拟合。该估计没有模拟全零跳过，因而零点仍可能估计出符号代价；实际零网格却仅1byte标志，二者差异逐点记录。

lambda参考量级由六帧当前编码器硬量化点的 `||dL_image/dz|| / ||dR_est_bpp/dz||` 中位数确定，实测 **0.1886496685**；候选固定为该值乘 **[0,1/16,1/4,1,4,16]**，共六个。没有搬用未换单位的旧lambda。

记录点重新硬量化、实际写ORC2诊断文件并round-trip。每帧选择最小 `L_image + lambda*R_actual_frame_bpp`，step0保留。分摊实际序列头时：每P帧packet bit + `(86字节序列头+1字节I标志)/15`，再除有效像素数。这使15个P的分摊和等于真实完整ORC2文件，但**六帧校准投影不是正式序列RD点**。

诊断检查点文件是16帧合法容器，仅当前帧放入被检查码字、其他P置零。它用于真实packet长度/符号检查，绝不用于宣称完整序列效率。全零文件102字节；分摊到每P是54.4bit，不是0bit。

非零lambda保留最低/最高压力，另从中间三者按六帧投影完整比特比例最接近25%者选一个，共不超过三个。该选择完成后冻结；实际预算仍必须由完整Train2/Val6文件判断。

对完全相同整数状态，同一帧/同一lambda的固定STE目标梯度可以复用；首次复用与新反向核对（atol1e−7，rtol1e−5）。所有checkpoint仍重新硬渲染和实际编码，记录缓存次数/耗时。该缓存不截断零点梯度、不改变硬前向。

## 完整预算与对照

每个固定lambda独立处理整段所有P。训练六帧若算法完全对应，可复用该lambda校准结果；其余帧执行相同150步程序。不同lambda之间不混合帧。

每个预算10/25/50%，从实际生成的整段候选中选择满足字节预算且L_image最小者，平局更少字节；固定ZERO无增强文件可退回。若选的是实际全零容器，102字节仍计费；若选择明确的无文件ZERO退回，增强bit为0，两者分别记录。

只有某一预算下两段Train2都满足真实预算，且完整I/P平均PSNR增量≥0.1dB、LPIPS恶化≤1e−4，才扩展Q2到Val6。现有E比较也对V1.2已存在l0–l3实际候选应用相同整段预算/退回规则，不产生新量化等级。

原版对比用各视频现有QP0–3匹配P精度点，在总bpp真实重叠区间内插值，不外推。诊断连续码字、ZERO干预、WRONG_SOURCE不赋予正式码率。

## 实现修复记录

启动预检发现原计划GPU0被其他任务占用，尚无实验CUDA工作时改为当时空闲的GPU6；之后只使用GPU6。

首次B的阶段字符串检测错误使 `B6_lambda0`误走连续路径；实际bit字段检查报错终止。保留全部900次无效域更新及文件，并在`invalid_attempts.json`标记排除。修复为显式量化参数，只重跑B；有效A、学习率和lambda校准均未改变。该修复不是数值尺度/超参数搜索。
