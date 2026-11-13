# Flow-guided deformable A/B：事实执行报告

状态：本轮训练、150个64帧评测点、强度扫描、图表、媒体和本地完整性检查已完成。仅执行预定实验，不作方法有效性、统计显著性或收敛结论。GitHub发布状态另见 publication_status.json。

## 基座与训练边界

起点为PREV已提交HEAD `71ed7e74c75cc21d1007dd63876ca08db01ddf27`。完整B300恢复链：Original→pilot→B1500 TCG→A2000 LoRA(s=0.5)→最后覆盖B300增量。合成参数SHA256：`743218b24af1efe631d71e3f0626a6ffdf423cef2175f33fc41470fb064c8552`。codec/entropy/Bridge/de-tokenizer/现有TCG/LoRA权重全部冻结，输入梯度和4帧段内TCG跨帧梯度保留。没有从旧T100/T300初始化，也没有重训旧S/U/T。

A_zero_base基础位移为0，局部offset/mask/DCN/q/fusion仍可学习；B_flow_guided基础位移来自冻结RAFT在独立Original解码参考RGB上的预测。A/B证据张量相同，A仍包含运动证据，不是完全无对齐或无运动信息的组。每组新增258,372个参数，同seed20261113与相同初始化、样本顺序、crop、QP及四帧短段。

## 结构和状态

Bridge18通道输出后接单deform-group的3×3 modulated DCN，18输入/18输出。offset输入58通道，Conv3×3 58→64→64→27；18个offset采用1.0*tanh，单位latent网格；其余9个logit经sigmoid。offset/mask末层全零；DCN中心核2*I、其余核与bias0，使初始mask0.5时中心内容恒等。这是本实验初始化，不是论文标准初始化声明。

最终DCN只采样原始未warp历史latent；粗warp只供offset预测。每点独立几何越界屏蔽，全局v采用中心采样有效性，全部无效时填回当前latent。q接口59→32→1，残差融合54→64→两个残差块→18零输出；l_prime=l+beta*q*delta。训练beta1，输出同时进入原conv_in和AdaGN。buffer只保存上一帧原始Bridge latent及元数据，修正latent与候选RGB不进入codec DPB。各候选独立reset与TCG滚动。

## 数据、缓存和目标

沿用32个隔离ULong训练来源组、128个不重叠16帧窗口、固定256×256原始RGB裁剪、QP0/5/9共384个实例；先RGB裁剪再冻结codec/Bridge。两组沿用前轮前300个joint计划，每更新4个4帧段等权，microbatch1；按各自当前权重从窗口I帧reset因果预热。段内保留跨帧梯度，段外detach。来源、完整计划与重复次数见data_manifest.json、training_plan.json及metrics/training_coverage.csv。600条正式更新的loss、分组梯度和激活记录见metrics/training_updates.csv；零初始化及后续固定样本诊断见checkpoints/各组/diagnostic_*.json。

旧训练缓存original仅有I帧。本轮补齐所有P帧FP32 Original参考RGB，并对每种QP抽样逐帧与真实bitstream decompress比对，结果严格相等，见audits/original_reference_replay_q*.json。参考流来自已审计RAFT权重/transform、双向各12次迭代，只读取解码参考，不读取GT、GT-flow、候选输出或未来帧。photometric与FB残差在首8个训练实例上固定RMS归一化，latent/QP固定统计沿用合法训练统计。

磁盘压力下使用可逆字节重排压缩；后续流/证据保存为新分支实际消费的latent网格，向量按真实宽高比缩单位，Original参考RGB仍完整FP32保存。旧RGB网格条目保持只读兼容，已核对同尺寸二次resize数值完全相同，详见audits/cache_storage_revision.json。

L_sp=LPIPS_Alex(normalize=True)+0.5*DISTS+0.1383589248954108*RGB_L1；L_temp为GT固定流/有效区下的MC-RGB变化残差+0.5*有符号归一化/8 Sobel变化残差。L_total=L_sp+4.63349060748*L_temp。明确不含旧0.1*L_align，没有新增GAN、滤波训练目标、teacher-latent或HF旁路。AdamW lr5e-5, betas0.9/0.999, eps1e-8, weight_decay1e-4, global_clip1.0，无调度。重建、分支、loss为FP32参考路径；codec沿用FP16，Original P参考FP32、I沿用冻结image codec精度。

## 实际步数与停止规则

|组|已提交更新|停止原因|更新计算秒数|
|---|---:|---|---:|
|A_zero_base|300|BUDGET_300|950.041815|
|B_flow_guided|300|BUDGET_300|973.078031|

保存0/1/10/50/100/200/300及latest，包含optimizer/RNG/计划游标/协议及实现哈希。checkpoint0/50/100/200/300均等待自己的64帧评测，实际领先评测0个间隔。按首评测前冻结的规则，仅连续两个检查点被同一早期点在PSNR↑、LPIPS/DISTS/FloLPIPS/MC-Sobel↓五指标Pareto支配且至少一项严格更好才停，容差为1e-6相对。所有检查点机械判定均为继续，两组最终达到300上限，没有自动续训。

## 评测范围与原值

150个视频×模型×step×QP×beta/blur点，每点64连续原分辨率帧，总9600条逐帧记录。4个ULong合法开发源，UVG仅Bosphorus合法validation单序列，Bosphorus原生120fps；2个可视案例ULong原生30fps。全序列从相同I帧/GOP起点独立reset/rollout，无重复帧补齐。所有候选共享对应64帧真实码流，未混用旧16帧bpp/指标，也未使用最终135点测试。

固定多QP对照点为[100, 300]；没有按质量选择其他step。两个固定案例分别beta0/0.25/0.5/1独立滚动，beta0经64帧与B300精确相等后显示去重。B300轻微Gaussian sigma0.25/0.5/0.75仅诊断，核半径ceil(3sigma)，离散归一化核，replicate边界，不进入训练。

VideoGigaGAN RWE：已核对[原文Eq4与Sec4.1](https://arxiv.org/html/2404.12388v2#S4.SS1)、[官方项目](https://videogigagan.github.io/)及[补充材料](https://videogigagan.github.io/assets/supp.pdf)。其核心为用候选估计的forward RAFT运动warp GT，再计算有效像素上的平方RGB向量误差。官方材料未给出可复核的精确non-occlusion mask实现/阈值与通道归约代码，本轮按任务允许的回退标记RWE_NOT_VERIFIED，不生成公式变体冒充原RWE或填0。停止规则首评测前已固定为其余5个指标；取舍图用已核实MC-Sobel轴，不重命名为RWE。

|ULong四源等权，QP5，beta1|step|PSNR|LPIPS|DISTS|FloLPIPS|MC-RGB|MC-Sobel|边缘强度|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A_zero_base|0|27.9155857|0.0996360616|0.0574182933|0.0874121785|0.0106683749|0.00373769292|0.00734078853|
|A_zero_base|50|28.0914598|0.0991979675|0.0567935512|0.0867761905|0.0104591279|0.00364982531|0.00718146633|
|A_zero_base|100|28.2056688|0.0993364991|0.0568215854|0.0867780602|0.010305639|0.00358596975|0.00706747646|
|A_zero_base|200|28.1900011|0.100152993|0.0574249583|0.0878415198|0.0102403367|0.00354836607|0.00695261253|
|A_zero_base|300|28.2674965|0.100372674|0.0573277744|0.0880266648|0.0102197726|0.00353267009|0.00682408109|
|B300|0|27.9155857|0.0996360616|0.0574182933|0.0874121785|0.0106683749|0.00373769292|0.00734078853|
|B_flow_guided|0|27.9155857|0.0996360616|0.0574182933|0.0874121785|0.0106683749|0.00373769292|0.00734078853|
|B_flow_guided|50|28.0581193|0.099253476|0.0568535659|0.0868219817|0.0105027576|0.00366655477|0.00720274992|
|B_flow_guided|100|28.1392535|0.09934824|0.0567664709|0.0867392421|0.0103836981|0.00361378002|0.0071021672|
|B_flow_guided|200|28.1413916|0.100369941|0.0572865151|0.0878353011|0.0103002637|0.00355591804|0.00694885619|
|B_flow_guided|300|28.203899|0.100543299|0.0572934481|0.0879860973|0.0102987345|0.00355461462|0.00686330979|
|Original|0|27.7091243|0.110888185|0.0681697235|0.0965851851|0.0109080336|0.00386102447|0.00790296957|

最终复核发现3个新GT监督缓存源的有效掩码在1088高补齐域计算后，只裁回1080而未与原生采样域相交。已对相关72个评测点修正固定掩码并仅重算MC-RGB/MC-Sobel；训练、模型、参考流、候选输出和其他指标均未改变。原值保留在audits/native_validity_correction/before，逐帧差异和输出MSE核对见其points目录。四项未变的停止指标已排除可能的连续支配，修正后重新判定仍全部继续；没有基于结果更换结构、loss或权重。

逐帧/逐视频×QP×step×beta原始数值、逐数据集等权汇总与B300/A配对差值分别见metrics/all_per_frame.csv、per_video_annotated.csv、aggregate_by_dataset.csv、paired_differences.csv。原始per_video_metrics.csv中的RWE_status因通用数值归约为空，annotated表只将该显示字段明确为RWE_NOT_VERIFIED，未改任何数值。

取舍图分别逐视频绘制LPIPS/DISTS—MC-Sobel，明确step、beta和模糊参照；黑圈仅表示当前显示二维指标的非支配点，不代表最佳模型。R-D分别绘制ULong和UVG单源，包含独立Original、冻结B300、A/B100与A/B300，不跨数据集合并、不外推3个QP点。HEVC/MCL未虚构结果。

## 性能与资源

|组|codec|独立参考RGB|双向RAFT及证据|DCN/offset|fusion/q|完整最终decoder|非缓存端到端 ms/P帧|
|---|---:|---:|---:|---:|---:|---:|---:|
|A_zero_base|24.2701645|126.583051|814.556946|2.74071086|1.12621167|131.957261|1127.75012|
|B_flow_guided|23.9590242|125.53018|822.079803|2.9115166|0.99401992|132.311727|1135.19081|

实际RAM码流非缓存测量完整P1..P63，包含独立Original参考重建、双向RAFT、Bridge、DCN、融合及完整最终decoder，哈希传输也包含在总计时，GT/loss/离线质量指标不在该计时。分支命中缓存速度不是部署速度；零新增码流不代表实时。完整样本计时与累计峰值在audits/performance.json；逐组件峰值/驻留与增量显存见audits/performance_components.json。主性能报告峰值为进程内累计，未伪装成隔离组件峰值。

只使用物理GPU0–3，未终止其他任务。GPU2按计划生产缓存，A/B按自身下个样本等待；GPU0/1在组等待评测时运行相应评测，GPU3处理队列。早期CUDA OOM进入等待/重试，所有最终评测点均从reset完成并PASS，没有降分辨率、删帧、改loss或填缺失指标。缓存格式调整只涉及本任务自己的生产器重启；训练源/计划/数值目标未变。CUDA grid_sample与DCN反向使用deterministic warn_only，固定seed不等于bitwise确定性声明。

## 媒体、完整性与限制

30个原分辨率FFV1与12个5栏原生ROI FFV1，均连续64帧，附MP4播放副本；每栏ROI256×256，标题独立48像素，不遮挡或缩放内容。模型步数、beta、QP、真实码率、帧范围、FPS与SHA256见VIDEO_INDEX.md及audits/video_delivery.json。新增beta1视频在1/32/63固定帧的浮点MSE与对应已完成评测逐值一致。

B300加载/冻结、step0/beta0连续64帧等价、DCN坐标/梯度/初始化、独立TCG状态、codec bits/features、全部预期点/帧数/重复行/非有限值、checkpoint与媒体完整性在final_integrity.json复核。未增加第二seed、第三训练组或下一轮；未让执行端替用户做科学结论。限制包括单seed、32训练来源、固定开发来源、UVG单源、仅3个QP、RWE未完成精确核验，以及crop/full-frame分布差异。

最终指标修正期间，旧ULong源目录变为不可访问，相关任务立即停止。随后从既有只读ULong ZIP定向恢复相同4个开发视频，逐文件SHA256与冻结协议全部相同，写入本任务cache/recovered_sources并记录source_path_migrations.json；未替换样本或改写旧路径。已完成修正点幂等保留后恢复任务。

所有新增代码、缓存、日志和临时文件均在本独立worktree目录。大模型/数据/缓存/码流/大型视频不进普通Git，large_artifacts.json给出本地绝对路径、大小、SHA256和复现入口；发布范围与远端核验见publication_status.json。
