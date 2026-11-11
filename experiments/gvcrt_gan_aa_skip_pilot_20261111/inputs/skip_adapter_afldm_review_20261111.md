# 1/8 特征旁路、AF-LDM 与训练目标：逐项核查

日期：2026-11-11。仅文献/源码分析，没有启动训练、GPU 评测或修改模型。本文补充上一份 Bridge 建议，优先级更新为先检验 1/8 解码特征旁路的增量价值；历史条件 Bridge 仍为候选，不并入同一轮以免无法归因。

## 结论

用户提出的 1/8 Skip-Adapter 是合理、可隔离的假设：给生成器提供不必通过 18 通道投影的观测条件。AF-LDM 则支持另一条独立假设：结构性抗混叠与等变训练配合，可以显著改善分数位移一致性。两者分别针对信息通路和空间响应，不应被合并为“保证同时变清晰、变稳定”的结论。

## 1. 实际架构

源码 `src/models/video_model_gvcrt.py:70`：

`f_hat [256,H/8,W/8] → PixelUnshuffle(2) [1024,H/16,W/16] → Bridge [18,H/16,W/16] → decoder`。

PixelUnshuffle 保留全部元素，是可逆重排。不是分辨率一半就丢掉 3/4 信息。后续 1024→18 的连续投影确实是潜在表示瓶颈，但自然数据可能处于低维子空间，不能由通道数算出视频细节损失比例。1/16 latent 也不只是全局语义，它承载空间和纹理条件。1/8 f_hat 并不是完整 GT 细节，而是已经量化和时间预测后的解码特征。

`src/models/improved_model_gvcrt.py:298`：

`18ch → 512ch stage1/stage2 → tcg_low → ada3 → learned 3×3 conv + depth_to_space(2) → tcg_mid → stage3(512→320) → ada4/stage4 → tcg_out → ada_final → learned head(320→192) → PixelShuffle(8) → RGB`。

两个上采样都包含可学习卷积。重排固定，不等于相位对应的输出权重不可学习；但这也不保证对分数移动等变。实际部署的 custom CUDA 与 PyTorch 路径都必须经过新增分支。

## 2. 1/8 旁路的作用与限制

候选位置：第一次 2× 上采样后、tcg_mid 前，在 512 通道的 H/8×W/8 特征上注入。保留完整 B300，其 LoRA s=.5 和原历史机制保持一致，新增分支最后投影精确零初始化。

简单残差：`h_new = h + A(f_hat)`。

若采用 SFT：`h_new = h * (1 + gamma(f_hat)) + beta(f_hat)`，gamma/beta 最后投影权重与偏置置零。它是逐位置、逐通道条件，不是旧的单通道运动 Router；beta 是加性通路，gamma 调制已有响应。初版可以从加性残差开始；若选择 SFT，则与残差应作为不同结构对照，不默认 SFT 一定更好。

直观理论：令 z=B(f)，旧输出 G(z)，新输出 G(z,A(f))。若存在细节相关信息保留在 f、但无法从 z 恢复，旁路扩大了重建端可用的条件信息。新模型包含 A=0 的旧函数；在理想优化和给定目标下最佳可达值不会因扩大函数集合而变差。但训练得到的模型、泛化、DISTS 和时序不受此保证。

线性化 `delta_x ≈ J_z delta_z + J_A delta_A`：旁路既能传有用的细节，也能传上游波动。不能承诺它单独消除静止树叶的闪烁，也不能声称新增了码流没有的信息。改变重建路径后还要确保 canonical codec DPB 与参考刷新不被影响，只有这样原 bpp/码流才真正不变。

## 3. AF-LDM 原文与数字

本轮完整提取并阅读用户 PDF。表格在 **Table 2**，不是 Table 1；Table 1 比较 StyleGAN3 与 AF-LDM 的设计。已渲染检查 Table 2 页面，保存 `afldm_table2_page6.png`。

| 方法 | Rec. PSNR | rFID | Decoder SPSNR |
|---|---:|---:|---:|
| SD VAE 原发布模型 | 26.94 | .69 | 22.31 |
| SD VAE†，作者在 ImageNet 训练 | 26.45 | .63 | 21.36 |
| + Ideal sampling | 25.76 | 1.01 | 21.75 |
| + Filtered nonlinearity | 25.42 | .86 | 24.61 |
| + Equiv. loss，AF-VAE | 25.53 | 1.00 | 33.88 |
| SD VAE† + Equiv. loss | 24.59 | 4.18 | 30.74 |

正确比较：直接加 loss 的 matched baseline 是 26.45/.63，变化为 PSNR -1.86dB、rFID +3.55。25.42/.86 已含两项结构改造。AF-VAE 对同设置普通 SD VAE†，PSNR -0.92dB，rFID .63→1.00，Decoder SPSNR +12.52dB；相对“结构改造但没有等变 loss”，PSNR +.11dB、SPSNR +9.27dB，但 rFID .86→1.00。因此支持有用的质量/一致性折中，不支持“所有重建质量均提高”。rFID 数值翻倍不是人眼画质按比例下降。

Sec.3.3 明确将训练中混叠加剧的原因表述为 hypothesis，认为不完美设计的混叠可能帮助网络学习。原文没有证明“普通 VAE 优化等变性的唯一办法是砍掉所有高频”。Filtered nonlinearities 抑制无法由当前采样率正确表达的频率，并不等于把所有高分辨率纹理抹掉。Table 2 也不是无条件定理。

Table 1 明确：**StyleGAN3 用 Kaiser，AF-LDM 用 FFT 重采样**。Filtered nonlinearity 保留 2× upsample→nonlinearity→2× downsample 思路。不能把二者混称为 AF-LDM 的逐层 Kaiser 配方。尤其不能把 PixelUnshuffle 当作丢弃相位的普通抽取，未经验证先对 f_hat 做低通可能丢掉旁路本来要保留的真实信息。

原文的 scale 是 VAE 8×；本项目 latent→RGB 16×。本项目 1/16 latent 格=1 RGB px、1/8 格=2px、1/2 格=8px、1格=16px。1/8 格与1/2格不都对应8px。已有无插值控制是 f_hat 移动1格=8 RGB px，且包含 Bridge+decoder。

## 4. 为什么不能把 AF-LDM 与 H2 视作相同实验

H2 额外损失的实际实现：`torch.roll(f_hat, delta)`，delta 为 feature 整数格；输出相应 roll 8delta，裁掉64像素边界，用 L1；目标输出在 no_grad 下，辅助路径禁用TCG历史；训练 full Bridge+TCG+LoRA，但原decoder主干冻结，TCG不从该辅助loss得到梯度。

AF-LDM Eq.(5)：`z=sg(E(x))`；比较 `D(T_(Delta/k) z)` 和 `T_Delta D(z)`，乘有效区域mask，平方L2；采用Fourier定义的分数latent移动；没有在公式中将整个decoder输出目标分支stop-gradient。它还修改结构，并在VAE中分别训练encoder/decoder的等变目标。本项目冻结codec和已有LFQ表示，迁移条件不同。

H0无新增等变loss也退化。H2相对H1的匹配增量为 PSNR约-.064dB、DISTS约+.146%，不能把H2相对B300的全部-.2234dB/+1.502%归给等变loss。H2与论文现象有启发性相似，但不是共同根因的直接证明。

## 5. 损失核查

最近末端adapter：`Lsp = LPIPS_Alex + .5*DISTS + .1383589248954108*RGB_L1`。A仅Lsp；B增加 `1.267681173415134*(RGB_MCResidual+.5*Sobel_MCResidual)`。这轮没有额外RAFT或equiv项。

H系列：上述Lsp，加 `.05*RAFT_motion + 2.603297563918803*(RGB_MCResidual+.5*Sobel_MCResidual)`；H1/H2再加latent项，系数 .2953169862992426；H2的equiv系数 .10179495298276185。各项按有效帧、像素等归约，不可把系数直接当梯度占比。

原 GVC-RT Eq.(6)/(9) 是 L1+LPIPS、latent distribution，StageIII加GAN并做联合训练。当前配方不是原文照搬；原文列DISTS评估不等于以DISTS作为训练目标。

1和.5本身不能证明感知loss过强，旧的“.01运动项所以弱150倍”不适用于当前实验。需要比较加权梯度作用于同一可训练参数的范数、方向和具体输出变化。历史初始梯度校准也不保证后续持续平衡。

两种逐帧感知损失不是原则性错误，但都不直接要求跨帧叶片身份/几何保持一致；它们的某些容忍性可与细粒度时间目标不一致。不能据此直接判定去DISTS会成功。应以同结构同样本的有/无DISTS对照检验，DISTS始终可保留作独立评价指标，避免同时换结构、去DISTS、换VGG、加GAN后无法归因。

真实视频时间损失与合成平移等变loss不同。前者处理场景运动、遮挡、上游误差等；后者只检验指定坐标变换下模型响应。不能互相替代。

## 6. 若进一步做 AF 启发的等变约束

有双路输入时，定义 `G(z,u,H,q)`，z是1/16 latent，u是1/8条件，H是TCG历史。RGB移动Delta要求同时处理 z的Delta/16、u的Delta/8以及每层历史对应移动；QP保持相同。只平移z而不平移u，会把条件错配造成的差异误惩罚为混叠。

最小结构性辅助实验可禁用历史、同时变换全部空间条件；真正的视频测试再使用正常历史。对当前表示做Fourier移动可能产生分布外latent。必须包含对实际平移视频重新编码的验证，不能仅看直接roll/warp已有latent。不同平移视频允许码流不同；同一视频不同重建模型必须使用同一码流。

采用FP32的移动/损失、正确像素单位、有效mask和归一化，并检查边界/GroupNorm全局统计影响。只修新分支的激活/重采样不能称整个冻结主干alias-free。若要改变原decoder的所有相关上采样/非线性，需要作为独立模型适配工程，评估初始输出改变、必要训练边界和推理延迟；不是插一层blur就完成。

## 7. 推荐顺序与可证伪条件

1. 优先检验1/8旁路。固定B300和一个已审计的空间+GT差分时间目标，只训零初始化小旁路。目的：绕过18ch投影是否提供有效细节证据；不是宣称解决全部闪烁。先观察多样训练片段的可学习性，再看未训练开发视频。
2. 同架构做损失对照：保留/移除训练DISTS，其余相同，以检验双感知目标是否妨碍几何/时间改善。若预算只允许两组，可让这两组都采用相同旁路、共同对照固定B300。这样没有在两组之间同时改变结构和loss。不能因短pilot失败宣布模块或目标理论无效。
3. 抗混叠作为独立路线：先固定要改变的重采样/非线性与表示变换，检查是否降低8px与更小位移敏感性及质量代价，再以“同结构有/无equiv loss”对照。不要同时加入动态offset和历史Bridge。动态offset本身不自动保证抗混叠，而且offset的帧间波动可能引入新的抖动。

最终判断需要联合画质、真实时间误差、原分辨率无损视频、码流一致性和速度；按各数据集单独给出Original/B300/候选R-D。纹理变锐但抖动加剧，或Sobel下降但叶片变糊，都不算达成目标。

## 资料边界

AF-LDM依据用户提供的CVPR2025 PDF（正文、公式和表格已直接核验）。原版GVC-RT依据本地归档论文。StableSR/CodeFormer的空间条件融合是可参考方向，本轮web工具未返回可引用内容；只读下载官方论文/源码的申请因自动审批服务502未执行，未重新核验其具体层位、初始化或定量消融，不把本提案说成原样复现这两篇。
