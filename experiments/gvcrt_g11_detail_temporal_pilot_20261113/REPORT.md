# G11单组先导试验：事实记录

本轮已完成300个正式optimizer updates，停止原因`BUDGET_300`。只执行G11、seed20261112，未启动G00/G10/G01、第二种子、多QP或下一轮。科研解释由用户与分析端处理。

## 固定参考与参数边界

C_ref加载链为Original→pilot→B1500 TCG→A2000 LoRA(s=0.5)→旧B300最后overlay→D12-S@900 seed20261112。旧B300不是flow-guided B组step300。D12-S完整S前向、condition=S及共享metadata/validity保持原样，S仍有codec/TCG历史。

D12-S增量文件SHA256：`fbfdb0c52daa65465a77a804613082622e70117189a5bd0d5d3a8123713cc856`；完整C_ref(I+P+S、按命名张量顺序)SHA256：`f415f4f386b6be2786819a52b5a58c460cb6661efa6d7620cf516d41932ff675`。权威预期来自父checkpoint_integrity/weight_manifest，实际文件逐项核验。本轮重算C_ref六源的LPIPS/DISTS/RGB_MSE/signedHF_MSE/SobelL1与父同协议结果逐项差值均为0，见audits/C_ref_parent_replay.json。父已审计P-model状态SHA与本次实测一致：`90d64a3325d23ffa950da7007828d081c75e3641c0b9a6169ef56fb20e9a2fad`。这里状态哈希与其他文档仅参数哈希口径不同，不混作同一数值。

只训练新增对齐/投影/融合/输出1,412,251参数，及stage3.blocks[0]原生卷积988,800参数，共2,401,051。精确12个接收块张量名和模块结构见audits/runtime_inspection.json；已有LoRA因子、TCG、S、GN/AdaGN、Bridge、codec、其他stage3/stage4与head均冻结。冻结hash排除有意训练的接收块；suffix保留输入梯度。

## 实现、数据和精度

G11在tcg_mid后先经过完整S，再以h_star=h_s+v*Delta_h进入stage3。读取冻结S.E的128通道当前/原始过去内容；128通道3×3 modulated DCN从raw过去一次采样。粗warp仅给388→128→27 predictor提供线索；残余dy/dx offset=tanh限±1个H/8格，mask初始0.5，DCN中心2I、其他0、无bias。P_h512→128；514→128+SiLU、两个父定义残差块、128→512零输出头无bias，无额外零scalar。

v是父协议FB/native-center及实际采样点几何有效性代理，不是真实遮挡或可靠度标签。采用像素中心8x+3.5、align_corners=False，向量单位除8。推理只读独立Original当前/过去解码RGB上RAFT的父metadata；GT-flow仅进入监督/评价，不进候选forward。raw c_prev不包含G11输出；每片段候选和参考独立2帧suffix预热、独立TCG_out状态，四帧内保留前后输出梯度，段间/reset断开。stage3/4 checkpoint只包纯计算块。

严格使用父12个train与6个validation来源，均ULong。final_test六源没有加入本地数据清单或被本轮评测。固定300步schedule每12步seeded shuffle覆盖12源各一次，start49..60、连续4帧全部49..63、256×256 ROI在native有效范围的8格上，同clip同ROI；预热start-2/start-1。首4个不同训练源对应片段用于全部校准。来源组隔离和全部ROI/schedule/SHA已冻结。

完整padded后缀执行GroupNorm，只有损失裁ROI；高通、Sobel与warp在native整图计算再裁ROI。分支/decoder/loss/grad为FP32，TF32关闭，codec沿用FP16。对照Original用独立原版代码和原始I/P，FP32重建与本轮评分相配；父对齐metadata仍保持其已审计Original FP16参考，不把二者混成同一缓存。缓存仅复用父码流/codec feature/GT/独立Original，另缓存本任务冻结prefix/S/E及固定GT-RAFT监督。所有新写入留在独立worktree。

## 固定目标和正式训练

L_sp=LPIPS_Alex(normalize=True)+0.5*DISTS(require_grad=True)+0.1383589248954108*RGB_L1。L_temp=GT参考固定运动补偿RGB变化误差+0.5*有符号/8 Sobel变化误差；只纳有效同reset段相邻对，双端梯度保留。

预算E_j为LPIPS、未加权DISTS、对GT的signed5×5 binomial highpass MSE。m_j为首4个训练校准clip的C_ref平均误差；g_j=(E_j(candidate)-1.01*E_j(C_ref))/max(m_j,1e-8)。L_budget=sum(mu_j*relu(g_j))；mu初始0.1，正式更新后mu=clip(mu+0.01*detached(g),0,1)，validation不更新mu。C_ref只提供误差基准，不是RGB蒸馏目标；这是软预算，没有逐样本或泛化保证。

训练固定m_j：`{'LPIPS': 0.0768924355506897, 'DISTS': 0.0990268811583519, 'highpass_signed_error_MSE': 0.00030260753010225017}`。

step1..50仅L_sp+L_budget；step50已提交后，在固定4训练clip上对接收块原生参数计算L_sp/L_temp梯度范数、比值和余弦，无权重更新。lambda_star=0.2*median(norm_sp/(norm_temp+1e-12))，51..100线性升至lambda_star，101..300固定。

实际lambda_star=`1.782992434577951`；逐片段范数/比值/余弦见audits/lambda_calibration.json。

AdamW新分支peak_lr5e-5、接收块5e-6，前20步线性warmup、其后恒定，betas0.9/0.999、eps1e-8、weight_decay1e-4、全局clip1。每步一个4帧clip。训练完整loss/误差/乘子/LR/梯度/激活/ROI见metrics/training_updates.csv及本地logs/train.jsonl。checkpoint含branch、native_receiver、optimizer、RNG、mu、lambda、schedule游标和协议SHA，未遗漏接收块。

纯训练更新计算累计1556.220339秒，缓存和验证等待不计入该数。固定观察点实际完成[0, 50, 100, 200, 300]，每点六源验证全部结束后才推进。预设预算停止规则与逐源证据见evaluation/decision_*.json，未因单次指标波动或PSNR下降自行改方向。

## 验证与原始指标

共48个模型×checkpoint×validation-source点，每点47/48独立suffix预热、49..63计分15帧。跨reset时间对被排除，最大14相邻对。所有候选共享对应64帧真实码流、实际QP、feature、native/padding约定。bytes/bpp口径是完整64帧真实码流，未拿15帧长度重新冒充码流。

|dataset|model|step|sources|bpp|PSNR|LPIPS|DISTS|FloLPIPS|MC_RGB|MC_Sobel|signed HF MSE|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ulong|B300|0|6|0.00607236007|28.9948204|0.0779267581|0.0570457962|0.0719674498|0.00880140749|0.00306416433|0.0002322011|
|ulong|C_ref|0|6|0.00607236007|32.3947221|0.0683311671|0.0484556423|0.0621914189|0.0084537026|0.00299514934|0.000211257842|
|ulong|G11|0|6|0.00607236007|32.3947221|0.0683311671|0.0484556423|0.0621914189|0.0084537026|0.00299514934|0.000211257842|
|ulong|G11|50|6|0.00607236007|32.3869215|0.0682722204|0.0482494063|0.0621627502|0.00846246274|0.00299907789|0.000212049528|
|ulong|G11|100|6|0.00607236007|32.4320525|0.0680881429|0.0482596536|0.0619937247|0.00841048959|0.00296951799|0.000208618188|
|ulong|G11|200|6|0.00607236007|32.489073|0.0680476989|0.0485291051|0.061982107|0.00834651602|0.00294130526|0.000203979191|
|ulong|G11|300|6|0.00607236007|32.5204512|0.0680754774|0.0485302316|0.0619807179|0.00828751753|0.00291475811|0.00019985846|
|ulong|Original|0|6|0.00607236007|28.7067176|0.0894934773|0.0717365781|0.0826676005|0.00908573684|0.00323051744|0.000282895807|

逐帧、逐时间对、逐源、逐数据集等权汇总与分别相对C_ref/B300差值见metrics/。整体均值没有替代逐数据集结果。本轮合法来源只有ULong，其他数据集未评估；QP5图为bpp—质量散点及更新轨迹，不是完整R-D曲线。保留Original，未做码率外推或选择最终测试权重。

## 预检、性能、媒体及限制

step0对完整C_ref的RGB max_abs_diff=0.0，阈值1e-6。DCN恒等与正负/分数平移、越界回退、独立参考无可变别名、四帧状态梯度、双端时间梯度、冻结白名单检查均通过。smoke3步不计正式预算，完成后已恢复初始branch/receiver/optimizer/RNG及状态。见audits/preflight.json。

首次正式更新在backward阶段被Original初始化恢复的strict deterministic模式阻断，未提交optimizer更新。已保存当时step0/initial与原始凭据，修复setup_torch在Original初始化后恢复协议声明的warn_only，新增该路径的反向预检，并从同一初始seed/权重/schedule重做预检与step0。结构、loss、预算、数据和阈值未改变，详见audits/pre_first_update_determinism_fix。

新增分支/接收块、codec/Bridge冻结审计、相同bits/bpp/feature、非有限值与点数完整性见final_integrity.json。grid_sample/DCN反向采用deterministic warn_only，固定seed不承诺CUDA训练bitwise复现。

原分辨率FFV1为full-range RGB8，float→8bit使用round/clamp，逐字节无损回读；各视频统一source frame49..63及各自真实FPS。GT/Original/B300/C_ref/G11明确标识，G11保存step100及终点（达到时），附MP4播放副本。15帧短段不代表长时稳定性；循环预览不作长序列证据。完整媒体路径/哈希/身份见VIDEO_INDEX.md与audits/video_delivery.json。

独立Original参考+双向RAFT及G11分支/后缀耗时范围与峰值见audits/performance.json，分别标明实际父FP16参考路径（与缓存RGB核对）和额外FP32重建计时；输入codec feature来自已审计缓存，此计时不冒充完整entropy端到端解码速度，不宣称实时。

step0验证期间GPU2被其他进程占用导致临时OOM，未提交正式更新。已保留凭据，在用户授权GPU0–3范围按28000MiB可用显存排队/迁移，只有一个trainer，未终止其他进程、缩小后缀或更换数据/loss。实际GPU、PID、恢复次数和日志见completion.device、logs/runner_events.jsonl、audits/resource_recoveries。验证阶段恢复RNG，0步运行元数据迁移时已核对branch/native/optimizer未改变。

单seed、单训练候选、12训练源、6开发源、QP5和短验证段是本轮范围。没有把MC_Sobel百分比改写成人眼闪烁百分比，也没有把高频能量增加称为真实细节增加；单组结果不能独立归因于历史/DCN/接收块/软预算。用户树叶样例未纳入主训练/校准/停止/评价。

所有大型缓存、权重、数据和视频只在本地项目内，以large_artifacts.json路径/大小/SHA清单交付，不上传普通Git；代码、协议、审计、CSV、小图、报告和索引提交origin独立分支，远端核验见publication_status.json。
