# G11：交给执行端 Codex 的完整指令

日期：2026-11-13。用户已授权执行 G11。本文固定首轮工程配置，不表示超参数已被证明最优。本文整理端没有启动训练。

你是执行端。请完成实现、训练、评测、保存和 GitHub 发布，不要只生成脚本后让我再执行终端命令。你负责记录事实；结果的科研分析由我交给分析端处理。只训练 G11，不启动 G00/G10/G01、第二种子或自动追加实验。

## 1. 目录、GPU、任务边界

PROJECT=/Huang_group/zyyz/Projects/GVC-RT
PYTHON=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
WORKTREE=PROJECT/GVC-RT_worktrees/g11_detail_temporal_pilot_20261113
EXP=WORKTREE/experiments/gvcrt_g11_detail_temporal_pilot_20261113
BRANCH=codex/g11-detail-temporal-pilot-20261113

以上是路径缩写，脚本中展开成实际绝对路径。设计文件：PROJECT/research/spatial_detail_temporal_reconstruction_20261113/NEXT_STEP.md；执行预算以本指令为准。

全部新文件、缓存、临时文件、日志留在 PROJECT 内。使用独立 worktree；主目录有已有修改，禁止 reset/clean、全量 git add、覆盖旧产物。新 worktree 的 HEAD 不一定包含当前有效运行代码：只对明确依赖建立源码快照、来源路径和 SHA，不能假定 HEAD 等于已审计运行环境。已有数据、权重、缓存只读复用。

GPU 0/1/2/3 可用，选择一张显存足够的卡，只启动一个 G11 trainer。允许一个有界缓存生产进程与训练流水重叠，不启动四组训练，不因显卡空闲扩大任务，不终止无关进程。设置 TMPDIR 到 EXP 内。

本次只做 QP5、最多 300 optimizer updates。不要自动跑多 QP、135 点最终测试、第二种子或下一轮。若同任务目录已存在，核对协议和运行锁后恢复，禁止重复启动或覆盖。

## 2. 起点、完整加载链和固定参考

PARENT=PROJECT/experiments/gvcrt_stage3_source_step_scaling_pilot_20261112
空间分支 checkpoint=PARENT/groups/D12_S_seed20261112/step_0900.pt

C_ref=Original → pilot → B1500 TCG → A2000 LoRA(s=0.5) → 旧 B300 overlay 最后加载 → D12-S@900 空间分支。

这里 B300 是旧完整模型，不是最近 flow-guided 实验 B 组的 step300。D12-S 文件不是完整模型，必须先恢复底座。复用父实验审计 loader 和来源/权重 manifests，实测 SHA 对照已有记录，不手抄预期 SHA，不忽略 missing/unexpected keys 或绕过真实不一致。

完整保留 S 前向、condition='S' 和共享 metadata/validity；S 仍含原 TCG 等历史信息，不能标为完全无历史模型。C_ref 不重新训练，与候选不得共享会被修改的参数和状态。

固定参考：独立 Original GVC-RT、旧 B300、C_ref。Original 必须用独立原版实现和 PROJECT/checkpoints 中原始权重对应的审计路径，不能用微调模型零残差输出冒充。

## 3. 精确结构

冻结 codec 输出 f_t=[256,H/8,W/8]；冻结 Bridge 输出 z_t=[18,H/16,W/16]。保留 z_t 的 conv_in/AdaGN 路径。
在 tcg_mid 后、stage3 前，完整 S 得到 h_s=[512,H/8,W/8]。新增：
h_star=h_s+v*Delta_h，再送进 stage3。

读取冻结 S 编码器：c_t=E_S(f_t)、c_prev=E_S(f_prev)，128 通道。新缓存只存原始 f_prev 或严格等价的 c_prev，不存纠正后的 h_star/RGB。

对齐器：
1. 基础流来自独立 Original 解码当前/过去 RGB 上的冻结 RAFT；当前向过去采样。推理不使用 GT，不以候选 RGB 估计流或更新 codec DPB。
2. 沿用父协议像素中心 (8x+3.5,8y+3.5)、align_corners=False、流方向与 H/8 单位转换；不能只缩放光流图而漏除以 8。
3. 新建 128→128、3×3 flow-guided modulated DCN；残差 offset=tanh(raw_offset)，限幅 ±1 个 H/8 格；9 个采样点。粗 warp 只给 predictor 提供线索，最终 DCN 从未 warp 的 c_prev 采样一次。
4. predictor 输入 concat[c_t,coarse_warp(c_prev),c_t-coarse_warp(c_prev),base_flow(2),v(1),FB_error(1)] 共388通道；3×3 Conv 388→128+SiLU→3×3 Conv 128→27。末层 weight/bias=0，18通道 offset 初始0、9通道 sigmoid modulation 初始0.5。DCN 中心权重初始化2I，其余位置0、无bias，使初始中心采样对应基础流；DCN 权重可训练。
5. v/FB_error 沿用审计过的边界和前后向一致性，正确处理实际采样点越界。v 是有效性代理，不称真实可靠度标签。可复用近期 DCN 的机制和测试，不能加载旧18通道权重到128通道模块。

融合器：P_h=1×1 Conv512→128+SiLU；
concat[P_h(h_s),c_t,a_t,c_t-a_t,v,FB_error] 共514通道
→1×1 Conv514→128+SiLU→两个父分支同定义的128通道残差块
→1×1 ZeroConv128→512(bias=False)=Delta_h。

只将该输出层零初始化，其余融合层正常初始化；不叠加一个初始为零的可学习标量。除 predictor 明确要求的 bias 外，新增卷积默认 bias=False。保存精确模块清单。不加 scalar Router、GAN、主路滤波、latent 凸组合或新 LoRA。

## 4. 训练边界、状态和精度

只训练新增对齐/投影/融合/输出参数，以及 stage3.blocks[0] 的原生卷积。按实际 wrapper 明确白名单：可以训练原生 .base，但冻结已有 LoRA 因子；不能把整个 stage3 解冻。接收块预期988800原生参数，核对实际计数。

冻结 codec/熵模型、Bridge、decoder 前缀、原 TCG、已有 LoRA(s=.5)、完整 S、其他 stage3/stage4、GN/AdaGN、head。冻结 suffix 参数不等于 no_grad：必须保留输入梯度。

四帧内保留候选 TCG_out 状态计算图，段间 detach；候选与 C_ref 独立预热和滚动，不能回放 B300/C_ref 固定状态。raw f/E 可 detach/cache。沿用 scene cut、GOP、尺寸变化的 reset，跨 reset 不算时间损失。activation checkpoint 使用无副作用 functional state，防止重算重复更新 buffer。

后缀走完整 padded 特征图，只在损失处裁 ROI。禁止用局部 patch 穿过 GroupNorm 来代替全图。分支、decoder、损失及梯度 FP32，TF32 关闭；codec 保持已审计精度。

旧整个 P-model 哈希不变规则不适用，因为接收块有意改变；分别检查冻结张量、codec、Bridge 哈希，仅训练白名单可变化。码流/bpp/codec feature hash 仍须完全一致。

## 5. 数据、采样、流水缓存

沿用 PARENT/data_manifest.json 的12个 train 来源和6个 validation 来源、源组隔离、QP5及实际帧QP映射。不要使用其另外6个 final_test 来源做本轮选模。不重新全库扫描，复用已通过的来源审计。

seed=20261112。新建显式300步 schedule：每步一个连续4帧 clip；每12步将12个 train 来源 seeded shuffle 后轮转一次；start 在合法49..60抽取，全部监督帧处于49..63。256×256 ROI 在原图有效范围按8像素网格抽取，同片段固定ROI。warmup 为 start-2/start-1 两帧候选独立无梯度滚动。不能照抄旧 schedule 前100步的两个2帧片段。

固定校准集为 schedule 最先出现的4个不同 train 来源对应的四帧片段，不按性能挑选。训练前保存 schedule、ROI、校准集和SHA。时间损失仅取片段内有效相邻帧。

缓存须验证来源、码流、特征、预处理和权重哈希。可增量生产，但训练严格按固定 sample ID 消费，不以缓存完成顺序决定采样。step0预检必须通过；正式训练不必等待全部后续缓存。

## 6. 固定损失和优化配置

沿用父实验 RGB 范围与指标实现：
L_sp=LPIPS_Alex(normalize=True)+0.5*DISTS(require_grad=True)+0.1383589248954108*RGB_L1。

GT图像估计的冻结RAFT流仅用于监督/评测：
r_phi=[phi(y_t)-W(phi(y_prev))]-[phi(GT_t)-W(phi(GT_prev))]
L_temp=valid_mean(abs(r_RGB))+0.5*valid_mean(abs(r_signed_Sobel))。

复用审计的 signed Sobel、mask、valid_mean。先整图 warp/求梯度/高通，再按 ROI 和有效性归约；不裁图后误用全图坐标。片段内时间梯度经过前后两个输出。记录有效像素数，空域不得除零。

质量软预算的 E_j 为 LPIPS、未乘0.5的DISTS、对GT的signed 5×5 binomial highpass MSE；高通沿用父 evaluate.py。独立冻结 C_ref 在相同片段/预热/ROI产生参考误差，不以 C_ref RGB 作为蒸馏目标。
m_j=4个校准训练片段的 C_ref 平均误差，正式训练前固定。
g_j=(E_j(candidate)-1.01*E_j(C_ref))/max(m_j,1e-8)
L_budget=sum(mu_j*relu(g_j))；mu初始0.1。
每次update后用该训练步 detached g：mu_j=clip(mu_j+0.01*g_j,0,1)。不使用validation更新mu。1%是工程容忍度，软罚项不保证泛化或逐样本硬满足。

步骤1..50：L=L_sp+L_budget。
step50保存后，对固定4个训练校准片段，仅计算接收块原生参数上的梯度，不更新权重：
lambda_star=0.2*median(||grad L_sp||/(||grad L_temp||+1e-12))。
梯度不有限或有效范数近零则停止报告校准错误；不得随意给常数。保存逐片段范数、比值、梯度余弦和lambda，随后固定。校准选择接收块，避免零输出层阻断新增分支上游梯度而误估。
步骤51..100：lambda(k)=lambda_star*(k-50)/50；101..300固定lambda_star。
总目标L=L_sp+lambda(k)*L_temp+L_budget。

AdamW：新分支peak_lr=5e-5、接收块peak_lr=5e-6；betas=(.9,.999)、eps=1e-8、weight_decay=1e-4；前20步线性warmup至peak，其后恒定；训练参数整体clip_norm=1。一个四帧clip一个update，空间项按帧平均、时间项按有效相邻对平均。

以上LR、比例、mu和阈值均为首轮工程默认值，不声称最优；禁止依据dev/test中途重调。记录grid_sample/DCN等非确定性，不承诺训练逐位复现。

## 7. 必要预检

完成小型smoke后恢复初始模型、optimizer、RNG和状态，不计入正式步数。
1. G11 step0对完整C_ref：相同输入和独立滚动RGB max_abs_diff≤1e-6；不是对B300相等。
2. DCN恒等/已知平移小张量检查通过，核实offset次序、单位、padding。
3. step0输出层/接收块有有限梯度；新增分支上游初始零梯度可正常，但smoke更新后在有效样本上offset/fusion上游须可获有效梯度。
4. 四帧梯度链、段间/reset断开、checkpoint无重复状态更新；冻结/白名单、独立参考无别名；码流/bpp/feature hash一致。
预检不过先修实现，未解决则BLOCKED并保存具体证据，禁止带错训练。

## 8. 训练观察点与停止

保存和验证0/50/100/200/300。每个检查点评测完成后再进入下一段；不跳过观测一直跑完。最多300更新，不自动追加。日志持续写cache/train/eval阶段、样本进度、step、时间。
立即停止：非有限值、冻结参数/码流违规变化、状态/坐标错误、不可恢复资源故障。
预设性能停止：从step100起，连续两个检查点（100/200或200/300），同一预算指标相对C_ref的视频等权均值恶化>3%，且在至少4/6个validation来源恶化>1%，则结束本轮并记录逐源证据。单次波动或小幅PSNR下降不自动停止。不事后改阈值。
主要终点step300；若提前停止则报告最后完成检查点。所有中间点保留，不从final_test选权重。

## 9. 评测和媒体

六个validation来源分别评测Original、旧B300、C_ref、G11；同码流、帧、padding/native mask、指标实现。历史结果仅在完全匹配时复用，否则重评。各模型独立rollout。父协议47/48预热、49..63计分：15帧、14相邻对；prefix仍按父协议因果滚动。不要调用父Evaluator.all()顺带读final_test，显式过滤validation。

记录实际bytes/bpp、PSNR、LPIPS、DISTS、FloLPIPS、MC_RGB、MC_Sobel、signed highpass MSE、signed Sobel L1、输出/GT高频能量、高频余弦、valid fraction。保存逐帧、逐源、逐数据集CSV；整体均值只补充。分别给出相对C_ref和B300差值。

各数据集画bpp—质量散点和训练轨迹，标明只有QP5；不得把单码率点伪称完整R-D曲线或混合数据集。Original必须出现在比较里；同bpp用marker/表格区分。未覆盖的数据集注明未评估，不造数据。

导出六源对应短段的原分辨率无损RGB/FFV1视频，GT/Original/B300/C_ref/G11对齐；G11至少step100和终点（若到达）。统一帧率、帧号、颜色范围和float→8bit转换。预览可另外拼接/循环，但不得替代原始无损短段，15帧结果不得声称长时稳定。匿名预览可附映射表。

用户树叶样例若有可核验同协议缓存，可另做diagnostic_only；不进训练、校准或提前停止，不称新盲测，不因此阻塞主交付。

报告只记录事实：不将MC_Sobel降低x%解释为肉眼闪烁减少x%，不把高频能量增加当细节更真实，单组结果不独立归因到历史/DCN。记录Original+RAFT参考路径及新增模块耗时，不宣称实时。

## 10. 恢复、交付、GitHub

实现并自行启动持久化runner，单实例锁、原子checkpoint和阶段状态。恢复保存新分支、训练过的原生接收块、optimizer、步数/LR、RNG、mu、lambda校准、schedule位置、协议SHA；在clip完成边界提交。不能只存branch漏掉接收块。父权重只读。

至少交付PROTOCOL.json、权重/来源/代码/缓存SHA清单、sample_schedule.json、实现代码、预检/梯度/冻结审计、训练日志、checkpoint清单、逐帧/源/数据集CSV、图和无损视频路径、REPORT.md、final_integrity.json、publication_status.json。给出可靠resume_command；已有进程运行时不能重复启动。

按用户既有授权，将本任务代码、配置、审计、报告、CSV、小图提交并push origin独立分支；不push upstream/main，不force push，不提交其他改动。大checkpoint/cache/video/数据集和凭据不上传，以本地路径+SHA清单交付。验证远端ref/commit及关键文件blob；发布失败单独记录，不混同训练失败。

最终返回事实JSON：status、complete、experiment_directory、C_ref链及SHA、训练参数白名单/数量、GPU、committed_updates、stop_reason、各检查点validation完成度、指标/图/视频/报告路径、冻结/码流核验、GitHub branch/commit_url/remote_verified。仍运行则诚实返回RUNNING、PID、日志与阶段，不称完成。

不要擅自开展下一轮或给出已解决闪烁等科研结论。完成后把产物留给分析端分析。
