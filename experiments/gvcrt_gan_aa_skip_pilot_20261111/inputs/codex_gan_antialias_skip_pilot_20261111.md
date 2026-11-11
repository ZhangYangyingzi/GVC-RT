# Codex 执行指令：B300 + 1/8 Skip-Adapter，GAN × 局部抗混叠

日期：2026-11-11。用户已确认物理 GPU 0、1、2、3 可用。本文件是给执行端的指令，尚未实施代码修改、训练或 GPU 测试。执行端负责实现、运行、记录及 GitHub 发布；研究结论交回主助手分析。

## 本轮范围与对上一版方案的落实

完成四组受控 pilot，每组最多 500 次额外生成器 optimizer 更新，在 0、250、500 步评测开发集；500 步后主动结束，等待结果分析，不自动续到 750 或扩大训练。四组可独立并行，缓存按需生产，不等待全量缓存。

本轮不新增 latent/Teacher/等变 loss，不更换 LPIPS 主干、不删除 DISTS、不训练完整 Bridge，不运行最终测试集。GAN 和局部抗混叠是本轮两个实验变量。AF-LDM 等变 loss 留给后续同结构有/无等变的匹配实验；此轮不是完整 AF-VAE 复现。

两项纠正必须优先于旧方案：
- 滤波改变前向函数，不能要求滤波开启时 step0 等于 B300，也不能要求无训练画质已经提升才准许短训。无训练筛查用于检查实现与量化初始代价；只有实现、数值、数据独立性或码流问题阻止启动。
- H0 是 full Bridge + TCG + LoRA 训练；本轮训练边界不同。相同损失包只能支持本轮内部比较，不能声称复现 H0 或证明 H 退化的唯一原因。

## 1. 路径、来源、隔离

项目：`/Huang_group/zyyz/Projects/GVC-RT`。Python：`/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`。

新 worktree：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/gvcrt_gan_aa_skip_pilot_20261111`。
实验根：该 worktree 下的 `experiments/gvcrt_gan_aa_skip_pilot_20261111`。
分支：`codex/gvcrt-gan-aa-skip-pilot-20261111`。

所有新增代码、日志、缓存、检查点、临时文件、视频和图表都放在上述实验根；设置 TMPDIR、模型编译缓存等到实验目录。既有数据集和模型只读，不搬动。已有同名目录则核验实验标识并安全恢复，不能覆盖或重复启动。保留主工作区现有改动。

先阅读：
1. `experiments/gvcrt_flicker_literature_review_20261111/next_step_gan_antialias_skip_plan_20261111.md` 和 `skip_adapter_afldm_review_20261111.md`。
2. `GVC-RT_worktrees/tcg_flicker_h_series_b300_20261110/experiments/gvcrt_tcg_flicker_h_series_b300_20261110/{protocol.json,source_inputs.json}`。
3. `GVC-RT_worktrees/tcg_flicker_ef_b300_20261109/experiments/gvcrt_tcg_flicker_ef_b300_20261109/{experiment_model.py,common_s2.py}` 及其 frozen runtime_tree。
4. 本项目本地 AF-LDM 论文/文本和 GVC-RT 归档论文。

复用已审计加载器、码流、指标和调度逻辑；将本轮依赖固定为实验内代码快照并记录哈希，不能运行时隐式导入用户正在修改的代码。外部网络材料不可用时使用已核验本地材料，不把下载变成无关的阻塞。

## 2. B300 加载与真实基线

B300 增量文件：
`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_flicker_pilot_abcd_20261109/experiments/gvcrt_tcg_flicker_pilot_abcd_20261109/B/training/checkpoints/step_0300.pt`

已在本轮实际 SHA256 核验：
`7465b57f20bfe2a8d6064a87155cf45c31a488cdeaa68111a228c28f6335ea85`

按来源加载链恢复原底座 → pilot → B+1500 TCG → A2000 LoRA → 最后覆盖 B300 的 52 个状态键，包含 Bridge mlp[3]。其他依赖 SHA 从已审计 source_inputs.json 读取并核验，不手工猜哈希。LoRA 训练/评测有效缩放固定 s=0.5，本轮 LoRA 参数冻结。

Original 是项目 checkpoints/GVC-RT_I.pt、GVC-RT_P.pt 独立原版重建，无 TCG/LoRA；不得将 pilot/parent0/v6_32 作为 Original。

先复现同精度 B300 至少两个视频 × QP0/5/9 的 step0，包括完整历史预热。关闭新滤波且 Skip-Adapter 为零时要求与本轮同路径 B300 精确相同；跨 FP16/FP32 的差异不能冒充加载错误或模型增益。记录 frozen codec、DPB/refreshed reference、码流、q_scale 和参数/缓冲区哈希。

## 3. 四组架构与训练边界

命名用 J0/J1/J2/J3，避免与旧 E/F 混淆：

| 组 | GPU 默认分配 | Skip-Adapter | 局部抗混叠 | PatchGAN |
|---|---:|---|---|---|
| J0 | 0 | 开 | 关 | 关 |
| J1 | 1 | 开 | 关 | 开 |
| J2 | 2 | 开 | 开 | 关 |
| J3 | 3 | 开 | 开 | 开 |

共同新增加性旁路：f_hat 原生 [B,256,H/8,W/8] → 1×1 Conv(256,32) → 原 WSiLU → depthwise 3×3 Conv(32,32) → 原 WSiLU → 1×1 Conv(32,512)；卷积 bias=False，最后投影权重精确置零。注入位置为 decoder.upsample 后、tcg_mid 前的 512 通道特征。其他卷积同种子初始化。不要换成 SFT、运动 Router 或 RGB 后处理。

四组共同训练：新增 Skip-Adapter、decoder.upsample.conv1，以及 decoder.stage3 内原生卷积参数。将实际参数名逐项列入白名单，不训练 stage3 的 GroupNorm/其他 norm，不更新已有 LoRA 参数。冻结完整 Bridge、全部 TCG 权重、AdaGN、stage1/2/4、head、codec、量化/熵模型；TCG 仍按 B300 前向使用和更新历史，冻结权重不代表关闭历史或切断输入梯度。四组生成器可训练参数集合和初值一致，判别器参数独立列出。

在同一 FP32 重建计算路径运行四组及匹配基线。修改模块不能被 custom CUDA fused 路径静默绕过；此 pilot 使用共同 PyTorch 参考路径即可。可用部署路径另测精度和速度，不能宣称参考路径性能等于原版 CUDA 性能。

## 4. 局部抗混叠实现：明确位置和算子

本轮固定同一种局部结构，不依靠开发集反复挑滤波器：

- J2/J3 在 `upsample(feature)` 后、加入 skip(f_hat) 之前，使用零相位低通；结果为 `LP(upsample(feature)) + skip(f_hat)`，随后进入 tcg_mid。以 FFT 的离散频率定义矩形通带，1/8 特征网格的 x/y 截止频率均为 0.25 cycles/pixel；这是前一 1/16 网格奈奎斯特频率映射到 2× 网格后的工作点，是本轮迁移假设，不是 AF-LDM 原模型参数。保留截止线上共轭频率并在测试中核对。只约束从低分辨率升采样的这一路，不用该0.25截止滤掉原生1/8旁路的真实高频条件；旁路非线性的降采样仍按其自身网格的奈奎斯特频率处理。
- J2/J3 的 stage3 全部实际 WSiLU 非线性改为 `理想 2× 插值 → 原 WSiLU → 抗混叠 2× 降采样`，包括 WSiLUChunkAdd 里的 WSiLU；保留原 sigmoid(4x)*x、channel chunk/add、残差、卷积、norm 与 QP 乘法语义。不要误换成普通 SiLU。Skip-Adapter 的两处 WSiLU 同样包裹；J0/J1 保留原算子。
- 实现幅值保持的 Fourier/理想插值及低通，处理奇偶尺寸、Nyquist 分量、Hermitian 对称和原点相位。运算与梯度用 FP32；边界统一使用 16 个本层网格像素 reflection halo，滤波后裁回，尺寸不足时用 min(16,H-1,W-1) 并记录。padding/FFT 周期边界本身存在影响，不宣称严格全局等变。
- 不过滤最终 RGB，不在 PixelUnshuffle 前随意加 blur，不把 PixelShuffle 的 polyphase channel 排序改掉。其他未修改上采样、TCG/非线性仍可能混叠，因此负结果只能否定这套局部适配，不能否定完整 AF-VAE。

实现测试应包括 DC 保持、低频正弦幅相、截止外频率抑制、上/下采样坐标、有限梯度和边界。关闭所有新算子时还原原函数。零初始化最后投影在 step0 会令其上游参数梯度为零，这是正常的；检查输出投影的非零梯度，并在一次临时更新后检查上游梯度，随后丢弃临时状态。

无训练检查记录 B300、J0/J1 step0、J2/J3 step0 的差异；做 RGB 2/4/8/16 px 的 x/y 平移测试，双路输入 z、f_hat 分别按 Δ/16、Δ/8 同步变换。辅助诊断隔离历史状态、使用有效区域 mask，不污染真实推理缓存。普通视频推理保持 B300 正常历史。无训练质量下降作为初始代价报告，不自动判定架构无效；数值/实现正确后允许短训适配。

## 5. 损失与优化

共同基础损失严格复用 H0 已审计实现：
`LPIPS_Alex + 0.5*DISTS + 0.1383589248954108*RGB_L1 + 0.05*RAFT_motion + 2.603297563918803*(RGB_MCResidual + 0.5*Sobel_MCResidual)`。

这些权重是本轮控制变量，不能称为新训练边界下的最优配方。保持输入范围、各项归约、遮挡 mask、RGB/Sobel 方向、有效面积和帧数一致。训练中的 GT-flow 指冻结 RAFT 从 GT 估计的流，不是真值运动场；只能作训练监督/指标，部署不能用 GT。新分支不用 Teacher/latent self/等变 loss，不添加 PCGrad 或自动变更目标。

生成器：fresh AdamW，Skip lr=5e-5，局部原生卷积 lr=2e-6，betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-4，global grad clip=1。所有组相同，不继承 B300 优化器。

J1/J3 使用同实现空间 PatchGAN：RGB 输入，5 层 4×4 Conv，channels=64/128/256/512/1，stride=2/2/2/1/1，padding=1；前四层 LeakyReLU(0.2)，spectral normalization，无 BatchNorm、无 sigmoid。输入真实和重建都为 [-1,1]，相同 crop，前向生成输出裁剪规则与共同基线一致。训练仅用 GT 与重建帧，不给 D 喂 QP/特征条件。

G_adv=-mean(D(fake))；D_loss=0.5*(mean(relu(1-D(real)))+mean(relu(1+D(fake.detach()))))。D Adam lr=2e-4、betas=(0,0.9)。G 步固定 D 参数和 spectral-norm buffers，允许对 fake 求梯度；D 步 fake detach。按有效全局 batch/目标帧归约，1 次 D update 对应 1 次 G update，不按每个累积 microbatch 额外增加 D 更新数。D/G 使用独立 RNG。

用训练集独立校准窗口（24 个，QP0/5/9 各8）做一次共同 D 预热50更新，fake来自固定B300，生成器不更新；该 D 状态复制给J1/J3。随后在上述训练窗口测两种架构可训练G参数上的 g_base 和 g_adv，使用共同 lambda_adv=0.1*median(||g_base||/(||g_adv||+1e-12))，两架构等权，记录未裁剪FP32梯度及最终值；非有限/零梯度属于实现错误，先修复。这里0.1是预先声明的 pilot 梯度比例，不是论文权重，也不保证效果。两 GAN 组同一 lambda，前100次G更新线性ramp到全值。校准后重载初始G和fresh优化器，不能将校准更新算进训练。

每25步记录基础各项、空间/时间/对抗加权梯度范数及余弦、裁剪前范数、D logits、权重漂移、Skip输出/主特征RMS；尺度异常应修实现，不能静默改变损失。

## 6. 数据和跨帧训练

沿用已经隔离的 ULong、Vimeo90K 训练来源和 H 系列开发/测试身份清单。UVG/MCL-JCV不加入训练。来源组清单已审计且哈希/依赖一致时复用，不重复全量扫描或等候所有窗口解码。源身份不确定的候选排除并记录，不把未知同源关系直接当独立。

按来源视频/序列采样再抽时间窗口，不按缓存文件数量加权。ULong:Vimeo 采用已审计上一训练协议的混合比例并在protocol冻结；若没有适用比例，预设1:1窗口数作为本轮预算选择，四组同序列，不宣称最优。统计unique sources、windows、时间覆盖和重复率。

每次G更新有效batch=4个窗口，256×256一致时空crop。四组用同一显式训练计划，固定seed=20261111，外部QP0–9均衡采样并继承codec层级offset，记录实际QP。不用缓存到达顺序取代计划顺序。

ULong沿用64帧窗口、原生I/P轨迹、既有warmup和末15个P帧监督；使用4帧chunk，chunk内保留生成历史梯度，chunk边界detach，不跨窗口残留。帧索引必须明确0/1起算，不能把协议区间抄成多/少一帧。Vimeo按真实septuplet处理I+6P，仅监督真实P帧；不能复制7帧拼成64帧。先每窗口按有效目标帧/transition平均，再对4窗口平均，避免长片段无意主导梯度。

用固定 codec 解码特征缓存；当前可训练模块的历史 warmup 必须在线重放，不能缓存并复用 B300 的 TCG 隐状态供更新后的模型训练。梯度检查要确认本轮可训练部分通过4帧链路得到预期梯度。

## 7. 并行缓存、训练和恢复

仅用物理GPU0/1/2/3，设置CUDA_DEVICE_ORDER=PCI_BUS_ID并记录GPU UUID/物理与进程编号映射。默认一组一张卡，独立进程、优化器、D和检查点；不要误将四组用DDP合成一个模型。CPU负责视频读解码和I/O预取，线程数依据机器可用核分配，避免四组各自抢满CPU。

采用共享不可变codec/GT-flow缓存的producer-consumer队列：首批16个训练窗口通过验证后即可训练，随后只预取32–64个计划窗口，有磁盘与显存上限。缓存原子写入、键含来源/帧索引/crop/QP/权重/精度/实现版本，所有消费者验证receipt。四组同一窗口只生成一次；禁止缓存可训练G输出。

GPU缓存worker与训练根据可用显存共存；显存不足时由正在等待缓存的组暂时让出该卡给producer，然后自动恢复，不因坚持四组同时驻留导致OOM/死锁。至少保留可推进缓存的机会，不能四组一起等待一个永远拿不到GPU的producer。新评测利用空闲时段/已暂停组的卡，避免抢占训练耗尽显存。

复用有效缓存，不等待全量先完成；满batch readiness协调，缺window等同一索引，不能换样本。OOM只调整microbatch/gradient checkpointing/调度，保持有效batch和目标归约不变并记录。不得终止其他无关进程。

持久化runner单实例锁、PID/heartbeat，记录cache/training/evaluation阶段、各组真实committed_steps、最近速度和ETA。每25步保存可恢复状态：G/D参数、优化器、RNG、sampler游标、ramp进度、协议/数据哈希；边界提交，恢复不重复算step。用本轮目录内日志及start_new_session/nohup维持任务，不要求用户另开终端。

## 8. 评测、输出和停止

开发集使用H已有32视频×QP0/5/9=96点，先按manifest确认其dataset构成，不以历史报告标题推断。本轮不把最终测试视频偷偷移进开发。每组0/250/500评同一96点。Original和B300优先复用来源/精度/指标/帧完全匹配的记录，不匹配点仅补测所缺依赖。基础加载验证完成后训练可与剩余baseline评测并行。

生成：每视频/QP原始CSV、每数据集单独R-D图（actual bpp横轴，LPIPS-Alex/VGG、DISTS、FloLPIPS、PSNR等各自纵轴）、共同码率和bitstream SHA核对、GT-flow RGB/Sobel残差、对GT边缘/纹理误差、推理延时及显存。基线必须有Original GVC-RT和B300。指标低为好者明确标↓，不得将Sobel强度下降单独解释成稳定提高。

若96点实际全为ULong，本轮只有ULong开发曲线，不伪造多数据集结果；其他数据集作为后续固定最终评测，使用现存manifest准确计数。已有H最终清单含MCL-JCV为225点/模型，不能继续无条件写成135点。仅有三QP时不给不稳健BD-rate结论，不外推码率。

固定GT挑选的静态树叶/平移/遮挡/大运动开发片段，导出原分辨率原fps FFV1无损视频和1:1 crop，GT/Original/B300/J0–J3各独立文件加并排对照；至少128连续帧用于长程播放，Vimeo短段单独标注。原无损视频不缩放、不帧插值，压缩预览另存。没有合适已保留开发片段就如实列缺项，不能借最终测试选案例。

正常完成各组500更新后停止，由主助手分析再决定下一步。只有NaN/Inf、梯度/状态/数据错误、码流审计错误等工程故障提前停止受影响组，其他独立组继续；不得因为step0指标差或某一步没增益就自动跳过整组，也不得擅自续训/调loss。若发现数据泄露影响全部组，停止共同相关工作并保留可核查事实。

## 9. GitHub与交付

已授权将本轮代码、协议、审计、简要日志、CSV、图表和事实报告推送现有GitHub origin的新分支。只stage本轮路径；不上传数据集、巨型视频/权重/cache、凭据，不改旧提交、不force-push、不合并main。记录大文件路径和SHA、Git commit URL、远端HEAD核验。网络失败保留本地提交、如实报告发布状态，不因此重跑训练。

实现本轮runner并提供status/resume入口，实际启动、确认训练step增长。工具会话允许时持续监控到完成；若提前交付RUNNING，必须核验独立进程存活、队列推进并保留心跳，不能宣称COMPLETE。

最终交付REPORT.md、protocol.json、source_manifest.json、training_plan、initialization/filter/gradient/codec审计、per_video_metrics.csv、per_dataset_metrics.csv、plots和videos索引、status.json、final_integrity.json、publication_status.json。报告只记录执行事实和数值，不写研究结论、不宣称GAN/滤波有效。

最终回复列出绝对实验目录、J0–J3 committed_steps、0/250/500各自完成点数、训练窗口/来源覆盖、缓存/码流核验结果、各数据集指标差值及样本数、图/视频位置、GitHub提交URL和任何未完成项。研究分析留给用户与主助手。
