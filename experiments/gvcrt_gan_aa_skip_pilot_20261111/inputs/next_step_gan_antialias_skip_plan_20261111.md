# 下一步方案：用 GAN 保细节，用抗混叠结构稳相位

日期：2026-11-11。用途：决定下一轮诊断与受控实验，不是训练命令。本方案把 PatchGAN、AF-LDM 启发的滤波/采样结构和 1/8 特征旁路纳入研究路线；不承诺其中任何一项单独解决闪烁或提高所有指标。

## 结论先行

我同意 GAN 和滤波都应进入下一步的正式验证。它们针对不同问题：**PatchGAN 给解码器提供局部纹理分布先验，抗混叠结构降低上采样与非线性对采样相位的敏感性**。树叶既可能因为单帧生成缺少真实高频而变糊，也可能因为相邻帧的亚像素位移改变了采样相位而变形；只加 GAN 或只加平滑损失都不能同时解决这两类问题。

但现有证据不支持“此前 H 实验退化已经证明是因为没有 GAN”。H0/H1/H2 都有质量退化，且 H0 没有新增等变目标。更关键的是，v6.24 的 projected-GAN 版本在同码率比较中没有胜过对应的无 GAN 续训版本。因此，下一步应把 **GAN 作为要认真保留和校准的候选训练目标**，同时设同架构无 GAN 对照；把 **滤波作为结构因素**，先筛查具体层位，再与 GAN 做 2×2 对照。

## 我们要回答的问题

1. H 系列里的纹理变软，是否能被 PatchGAN 恢复，同时不让叶片等细纹理出现更强的帧间跳变？
2. 在现有解码器的上采样/非线性位置加入带限滤波后，8 像素及更小位移下的输出变化是否减小，代价是否只是可接受的锐度变化？
3. 1/8 `f_hat` 是否保留了经过 1/16 Bridge 投影后不易恢复的有效细节？若通过小型 Skip-Adapter 注入，细节会更好，还是把上游潜特征抖动也更直接地带进 RGB？
4. 以上结构与损失能否在相同码流、相同实际 bpp 下同时改善空间质量和时间稳定性？

## 证据与边界

### GVC-RT 为什么支持纳入 GAN

本地 GVC-RT 论文归档 `experiments/gvcrt_neural_wrapper_v6_13_gt_lfq_staged_adaptation/teacher_sources/gvcrt_paper_v1.html` 中，Stage I 的重建感知目标和 Patch-GAN 对抗目标联合训练；Stage III 使用 `R + λ(L_rec + w_adv L_adv + L_distribution)`，论文设置 `w_adv=0.1`。这说明 PatchGAN 是 GVC-RT 训练设计的一部分，GAN 能为单纯像素/感知回归提供局部真实性约束。论文还显示去掉 Stage III 联合微调会损害结果。

这些结果来自原论文的训练流程、模型和损失尺度。当前实验从 B300 继续适配，冻结了不同的 codec/生成器边界；不能直接复制论文的 `w_adv=0.1`。判别器结构、数据和损失归约方式改变后，同一个数值不代表同样的梯度强度。

### 我们已有的 GAN 对照不能略过

v6.24 将 `A_continue_1000`（无 GAN）与 `B_projected_1000`（projected adversarial）在相同实际 bpp 的共同区间比较。其 CSV 是逐视频实际 bpp 插值、每个视频等权、每条视频 100 个共同网格点；下列百分比是指标相对变化，**不是 BD-rate**：

| 数据集 | GAN 相对无 GAN 的 LPIPS | DISTS | FloLPIPS |
|---|---:|---:|---:|
| ULong，8 个视频 | +1.69% | +1.53% | +0.87% |
| HEVC-B，5 个视频 | +1.76% | +1.04% | +0.83% |
| UVG holdout，2 个视频 | +3.87% | +1.85% | +4.67% |

这些指标越低越好。UVG validation 只有 1 个视频，不足以单独推断泛化。v6.24 两种训练都相对原始 GVC-RT 有改善，但 GAN 版本弱于该轮无 GAN 版本。该实验使用 v6.22 的 B1 checkpoint，不是本方案的 B300；它证明当前 GAN 配方尚未给出增益，不证明所有 GAN 设计都无效。

报告来源：`experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/results/ip_-1/{ulong,hevc_b,uvg_holdout}/equal_rate_summary.csv`；具体训练协议见该实验 `RUNBOOK.md` 和 `config.json`。v6.24 的 adversarial 权重为 `0.014448280572707944`，校准方法和目标函数都与论文 `0.1` 不同。后续要先审计梯度和训练稳定性，再定权重。

### AF-LDM 为什么支持先改结构再加等变目标

用户提供的 CVPR 2025 AF-LDM PDF 已本地核对。作者使用理想频域重采样与 filtered nonlinearity，再训练平移等变目标。其 Table 2 中，matched SD VAE† 为 PSNR 26.45、rFID 0.63、Decoder SPSNR 21.36；只加入等变 loss 后为 24.59、4.18、30.74；结构抗混叠加等变约束的 AF-VAE 为 25.53、1.00、33.88。相对普通 VAE†，一致性大幅提高，但 PSNR/rFID 有代价；相对已做结构改造但未加等变损失的配置，等变目标提升 SPSNR，其他指标并非全都提升。

这支持“结构和训练约束应一起考察”，不支持“低通滤波一定让视频更清晰”或“等变 loss 一定提升整体画质”。AF-LDM 使用 Fourier/理想重采样；StyleGAN3 的 Kaiser 滤波不能误写成 AF-LDM 的具体滤波器。对我们的 learned Conv + PixelShuffle/DepthToSpace，应先按实际运算位置设计滤波，不应把任意 blur 放在 `PixelUnshuffle` 前。

我们的 `PixelUnshuffle(2)` 是可逆重排，不等同于丢相位的普通下采样。可能的风险点在 learned upsampling、激活和随后的离散特征处理。需要按源码标出这些位置，再决定理想重采样的离散实现。latent 到 RGB 总放大倍率为 16：RGB 的 8 px 位移是 1/16 latent 的半格，也是一格 1/8 `f_hat`；必须按这个坐标换算测试位移。

### 为什么 GAN 不能代替时序约束

逐帧 PatchGAN 只判断局部图像块是否像真实纹理，不知道相邻帧是否一直是同一片叶子、同一条轮廓。GAN 可能恢复高频纹理，也可能让纹理更锐但更跳。跨帧约束仍要比较运动补偿后的重建变化与 GT 的真实变化，并用视频播放检查拖影、纹理跳动。若空间 PatchGAN 让细节恢复但闪烁仍在，之后再评估时空判别器；不要第一轮就同时加 3D GAN、额外流网络和更多门控。

## 建议实施顺序

### 阶段 0：锁定实验对象和数据边界

1. 使用同一个已审计的 B300 checkpoint 作为全部分支初始化；记录完整 SHA、LoRA 推理缩放（如该 checkpoint 保留 `s=0.5` 则固定为 0.5）、模块哈希、优化器初值和随机种子。Original GVC-RT 和 B300 都作为独立评测基线。
2. codec、量化器、QP、码流和 DPB/refreshed reference 路径固定。各重建模型读取同一份真实 RANS 码流；训练分支的输出不得反馈到 canonical codec DPB。否则“码率不变”不能成立。
3. 使用经过同源审计的 ULong 与 Vimeo90K 训练窗口。按来源组隔离训练/开发/测试，窗口随机抽样覆盖整段视频，不重复少数固定窗口来冒充大数据。UVG、MCL-JCV 等只按既有协议作未见开发/最终测试；每个数据集单独出 R-D 曲线。
4. 固定验证集、QP 覆盖和码流。开发集上的调参不能触碰最终测试集。最终报告至少同时给 Original GVC-RT、B300、候选模型。

### 阶段 1：零训练成本结构筛查

**目标**：先判断滤波候选在哪些实际 decoder 操作上能减小相位敏感性，以及是否会直接抹掉细节；此阶段不是证明滤波训练一定有效。

1. 在 PyTorch 和实际部署推理路径中逐层审计 `improved_model_gvcrt.py` 的两次 learned upsampling、PixelShuffle/DepthToSpace、相邻激活与 TCG 注入点。确认哪些算子对应真正的采样率变化，不能把 `PixelUnshuffle` 可逆重排误判成信息丢弃。
2. 对 B300 做无训练 A/B：原结构；只在最可疑的一处上采样/非线性处加入 AF-LDM 启发的带限重采样/filtered nonlinearity。筛查实际实现可用的 filter 强度与边界处理；禁止对整幅 RGB 直接套强 blur。
3. 用同一输入作 2、4、8 px 合成平移响应测试，并对真实静态细纹理、平移树叶、遮挡和大运动片段做编码端到端重编码/解码测试。合成平移用于测模型响应；真实视频测试用于判断该变换是否符合真实特征分布。
4. 同时记录输出高频能量、边缘强度、LPIPS/DISTS、GT-flow 对齐时间误差、无损原分辨率片段。若时间误差只因高频/边缘被明显滤掉才下降，该结构不进入训练。
5. 单独做零初始化 1/8 Skip-Adapter 的恒等性和梯度审计。候选注入点在第一次 2× upsampling 后、`tcg_mid` 前；初始输出须逐像素等于 B300。检查新增支路是否真的依赖当前 1/8 `f_hat`，并验证不影响码流、DPB、CUDA 与 PyTorch 路径。

**进入训练的条件**：至少有一个滤波位置在合理边缘/纹理代价下改善平移响应；Skip-Adapter 零初始化精确等价且梯度有效。若筛查无可用的滤波位置，先修正结构假设，不启动抗混叠训练。

### 阶段 2：四组 GAN × 滤波受控 pilot

所有组都从相同 B300 初始化，都启用同一个零初始化 Skip-Adapter，并使用相同训练视频、窗口顺序、QP 采样、优化器、更新数和验证步骤。只改变 GAN 与滤波两个因素：

| 组 | Skip-Adapter | AF 启发滤波 | PatchGAN |
|---|---|---|---|
| F0 | 开 | 关 | 关 |
| F1 | 开 | 关 | 开 |
| F2 | 开 | 开 | 关 |
| F3 | 开 | 开 | 开 |

这让 `F1-F0` 估计相同结构下 GAN 的增量，`F2-F0` 估计滤波增量，`F3` 检查二者是否互补。B300 原 Bridge、TCG、codec 和 DPB 保持不变；训练边界固定为 Skip-Adapter 加上阶段 1 确认必须适配的局部 decoder 上采样块，四组可训练参数数量应一致。不要在四组中途全量解冻 Bridge 或原 decoder。

四组共用同一空间与时间目标，不能为 GAN 组单独改其他 loss。第一轮保留当前 H 空间目标的同一实现和权重，时间项使用同一个 GT-flow 对齐的 RGB/边缘变化约束；这样直接回答“给 H 这套训练加 GAN 是否能保住细节”。PatchGAN 仅用于训练，推理时移除判别器。采用稳定的预热/梯度校准和裁剪；按实际共享参数记录 `g_spatial`、`g_temporal`、`g_adv` 的范数及余弦，不直接照搬 GVC-RT 的 `0.1` 或 v6.24 的 `0.0144`。

复现实验配置时，H 损失包按原实现锁定：`LPIPS-Alex + 0.5*DISTS + 0.1383589*RGB-L1 + 0.05*RAFT_motion + 2.6032976*(RGB_MCResidual + 0.5*Sobel_MCResidual)`。若核对实际 H checkpoint/config 后发现所选起点的权重不同，以该起点的审计配置为准，并让四组完全相同。

若需要短段训练，四组统一使用同样的因果 4 帧 chunk、同样的历史状态初始化和截断边界；chunk 内是否跨帧反传必须固定，不能只给 GAN 组打开 BPTT。codec latent/码流参考按既有边界 detach；若改成可变的生成参考，需另立实验。

先做短 pilot，在 250、500、750 更新保存并评估；每次看未见开发视频的不同数据集结果和无损视频。停止条件预先登记：连续两次检查均无共同画质/时间收益；GAN 出现明显振荡、噪声或高频幻觉；滤波带来不可接受的锐度/边缘损失；码流/DPB审计失败。pilot 只判断方向，不作为充分收敛或最终论文结果。

### 阶段 3：受控核对重建损失，而非简单删 DISTS

只在阶段 2 找到可用的结构/GAN组合后开展。目标是判断 H 的纹理退化是否与空间损失包有关，而不是同时改变结构和loss。

固定同一架构、GAN、时间项、训练数据和步数，比较三个空间损失配方：

1. **H 配方**：当前 LPIPS-Alex + DISTS + RGB-L1，权重完全复现 H 协议。
2. **GVC-RT 对齐配方**：RGB-L1 + LPIPS-VGG；DISTS 保留为评测指标，不作为训练项。GVC-RT 原文 Stage I/III 使用像素-感知重建与 PatchGAN；LPIPS-VGG 的具体权重从论文配方/本地实现核实，不照搬本实验不同归约下的系数。
3. **混合配方**：RGB-L1 + LPIPS-VGG + DISTS。它区分“移除 DISTS”与“改用 GVC-RT 的 VGG 感知特征”两种因素，不把问题简化成删掉单一 loss。

所有配方保持时间损失和 PatchGAN 监督一致。GVC-RT 的 rate 与 latent-distribution 项属于其联合 tokenizer/codec 训练目标；本实验固定真实码流、冻结 codec，并不训练同一个 LFQ latent 分布，因此先不把这些项照搬进 decoder 适配。记录每项作用于同一可训练参数的梯度范数、方向、梯度冲突率及输出纹理频谱；不以 loss 的数值或名义权重大小推断主导程度。根据开发集作选择，最终测试集只评一次。

### 阶段 4：完整评测与决策

1. 每个数据集分别以实际 bpp 为横轴，分别画 LPIPS、DISTS、FloLPIPS、PSNR/SSIM 等质量纵轴；Original GVC-RT、B300、候选模型都画在同图。不要把数据集平均成一条曲线。
2. 报告相同 bpp 对齐区间、每个数据集每个视频的变化和汇总不确定性；曲线不能用不同码率均值假装等码率。每个点对应相同视频、相同 QP/真实码流条件。
3. 时间指标至少包含 GT-flow 对齐后的边缘/纹理误差、FloLPIPS 或既定时间指标，并抽查真实播放。8 px 平移一致性只能作为诊断，不能代替自然视频时间稳定性。
4. 原分辨率、无损导出的连续播放视频由不知模型版本的人做并排盲看，重点看静止树叶、缓慢平移、遮挡和高速运动。锐度提升伴随闪烁上升，或闪烁下降伴随叶片变糊，均不能算达成目标。
5. 只有开发集上空间质量与真实时间质量都有可重复改善、无明显运动拖影/纹理闪烁、原始码流和 DPB 不变时，才把配置扩展到 135 个最终测试点。

## 对结果的预期

- **GAN**：合理预期是改善局部纹理真实感、减轻回归型解码器倾向输出平均/平滑纹理；不保证 PSNR、DISTS 或视频稳定性同步提升。v6.24 说明 GAN 权重、判别器输入和更新平衡必须重新校准。
- **滤波/filtered nonlinearity**：合理预期是降低分数位移/相位变化对高频重建的扰动；可能损失可重建的细节或降低边缘能量，需通过结构筛查与 GAN 配对验证。不是在 RGB 结果上糊一层。
- **1/8 Skip-Adapter**：可能绕过 `1024→18` Bridge 投影带来的条件瓶颈；也可能把 `f_hat` 的量化/预测抖动更直接带入解码器。它是可证伪的特征通路假设，不是增加码流信息或天然防闪烁模块。
- **组合**：只有当滤波减少相位敏感、GAN补回稳定的真实高频、时间项保持运动轨迹时，才有机会同时改善两个目标。预期方向可以提出，具体提升幅度不能在实验前承诺。

## 暂时不做的事及理由

- 不同时全量解冻 Bridge、TCG、LoRA 和全部 de-tokenizer；过去 H/E/F 配方已经显示多项一起变化时难以归因，且质量有退化。
- 不把 PatchGAN 宣称为“解决闪烁的 GAN”；当前 GVC-RT 的 PatchGAN 依据主要是空间真实性，时间约束需要独立测量。
- 不在 `PixelUnshuffle` 前随意低通；该操作是可逆相位重排，可能保留旁路想利用的细节。
- 不先加 3D GAN、额外运动估计器、Bridge 时序分支和更多 Router。若 F1/F3 只提高空间纹理而时间问题未改善，再有针对性评估时空判别器或历史对应模块。
- 不用 H2 结果证明 AF-LDM 一定适用于本模型。H2 与 AF-LDM 的输入变换、边界 mask、梯度路径和可训练模块不同，H2 只是提出动机的诊断证据。

## 资料索引

- GVC-RT 原文归档：`experiments/gvcrt_neural_wrapper_v6_13_gt_lfq_staged_adaptation/teacher_sources/gvcrt_paper_v1.html`
- v6.24 GAN 对照协议：`experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/RUNBOOK.md`、`config.json`；等码率结果见 `results/ip_-1/{ulong,hevc_b,uvg_holdout}/equal_rate_summary.csv`
- AF-LDM 用户论文抽取件和 Table 2：本目录 `afldm_cvpr2025_paper.txt`、`afldm_table2_page6.png`
- 现有架构及 Skip-Adapter 位置核查：本目录 `skip_adapter_afldm_review_20261111.md`
- 前序 H 实验与 loss 复盘：`experiments/gvc_rt1111_review_20261111.md`

本计划仅定义下一轮实验逻辑；尚未修改模型、运行训练或启动 GPU 工作。
