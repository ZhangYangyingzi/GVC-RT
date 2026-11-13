# 可直接交给执行端Codex的完整指令

你是本任务的执行端。请直接完成实现、检查、运行、评测、事实报告和GitHub发布；科学结论由用户和分析端讨论。不要只给操作建议，不要求用户另开终端启动，不擅自扩大模型/数据/loss。下列指令是本次执行准则；旧文档冲突处以此为准。

## 1. 路径、基座与资源

- PROJECT=/Huang_group/zyyz/Projects/GVC-RT
- PREV=$PROJECT/GVC-RT_worktrees/bridge_aligned_latent_pilot_20261112/experiments/gvcrt_bridge_aligned_latent_pilot_20261112
- WORKTREE=$PROJECT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113
- EXP=$WORKTREE/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113
- 分支：codex/bridge-flowguided-deformable-ab-20261113
- Python：/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
- GPU 0、1、2、3均已授权，可并行充分使用；不要使用其他GPU，不终止不属于本任务的进程。时间统一Asia/Shanghai。

先读PREV/analysis/review_20261112/EXPERIMENT_STRUCTURE_20261113.md及REVIEW.md，再检查PREV的restore.py、model.py、train.py、cache_data.py、evaluate.py、protocol.json、preflight/final_verification审计。原主仓库有其他未提交改动，不reset、不clean、不覆盖、不混入提交。以PREV所属worktree的已提交HEAD为起点建立独立worktree，记录commit；未跟踪的必要说明文件复制到EXP。若目标已存在，核对协议后幂等恢复，不另起重复任务。

一切新代码、日志、缓存、编译缓存、临时文件和输出都写到PROJECT下面；原数据集/预训练权重可在已有位置只读。不要把新实验放到home_dir/tmp或/tmp。引用旧缓存只读，禁止改写旧实验或误导入会向旧ROOT写入的util。

本轮从完整B300组合权重开始：复用已审计的Original→pilot→B1500 TCG→A2000 LoRA(s=0.5)→B300 overlay恢复链，并从PREV已验证manifest读取各源SHA，实测核对。不要把partial B300 checkpoint当完整模型，不从T100/T300续训。冻结codec/entropy/Bridge/de-tokenizer/现有TCG/LoRA全部权重，参数hash训练前后不变；保留生成器输入梯度。若发现SHA文字记录冲突，先查既有已审计凭据和实际字节，不凭猜测覆盖，也不因为非关键展示字段的笔误卡住整个实验。

## 2. 两组定义

A_zero_base：基础取样位移u=0，仍训练局部offset/mask、可变形卷积和融合分支。
B_flow_guided：基础位移来自冻结RAFT对独立Original解码参考RGB的估计，其余与A完全相同。

A并非完全无对齐；两组比较基础运动引导的价值。两组从同一B300、同一新增初始化、同一seed、同一训练计划启动。原TCG/LoRA均保留。不要增加第三个训练组，不重新训练旧S/U/T。旧T100/T300的beta扫描不是启动A/B的前置条件。

## 3. 基础光流与新增缓存

复用既有审计RAFT权重/transform及12次迭代，冻结网络，输入float32独立Original重建的相邻RGB x_ref_t、x_ref_prev，取当前→上一帧backward flow及反向flow；不输入GT，不输入候选生成结果，不使用未来帧。参考解码精度在训练/评测一致，采用已审计Original FP32重建路径，codec保留原精度。

重要：旧训练缓存original字段只可能有I帧，不可当作完整参考视频！检查帧数/帧号/精度，从已验证码流独立decompress/replay补齐所有P帧参考RGB，结果写新缓存sidecar。不要用GT替补缺帧。若直接用已解features重建Original参考，须与独立bitstream decode抽样逐帧比对，证明等价后才能复用。

新字段明确命名decoded_ref_backward/decoded_ref_forward/decoded_ref_evidence；旧GT flow/valid仅供监督和评价，禁止作为推理输入。缓存key含source、时间窗口、crop、QP、帧号、精度、参考权重/实现hash和流配置。各segment按需生产，完成后原子rename+凭据，消费者只读已提交条目。不要为检查依赖重扫完整数据集，不等待全量缓存才启动训练。

## 4. 可变形历史取样

插在Bridge输出18通道latent之后、de-tokenizer之前。H/W指padding后的尺寸。沿用训练集固定RMS统计，归一化l_t与raw l_prev。基础流缩到latent尺寸时按实际宽高比例分别缩放向量；不能只resize不缩单位。

offset网络输入为[current18,rough_history18,abs_difference18,base_flow2,normalized_QPs2]，共58通道；候选实现Conv3x3 58→64→64→27，隐藏SiLU。输出18个offset和9个modulation logit，单deform group、3x3取样、18输入/18输出。offset=1.0*tanh(raw_offset)，单位latent格；该上限是本轮预声明默认。offset/mask头最后一层全零初始化，初始offset0、sigmoid mask0.5。

最终采样位置p+p_k+u_base+delta_p_k，最终DCN采样的输入必须是未经warp的raw历史；粗warp只用于offset预测。每个采样点独立几何越界mask，乘到modulation上。DCN卷积初始中心核2*I、其余0、bias0，使初始mask0.5时历史内容不被无故缩半；明确记录此为本实验初始化，不声称论文标准。最终融合输出为零所以总体step0仍是B300。

优先使用本环境已有可用的torchvision deform_conv2d。若不可用，允许实现数学等价的9点grid_sample+可学习卷积权重聚合，需用已知位移、边界和梯度测试核对；不能偷偷退化成普通warp或改环境全局包。记录所用算子及性能区别。

## 5. 融合与状态

保留旧融合54→64、两个残差块、64→18零输出头：输入当前latent、对齐历史、绝对差；固定RMS归一化并将delta缩回原尺度。q网络59→32→1，与旧接口匹配：54内容+2证据+2QP+1有效性，末层零初始化。

两个证据通道为解码参考RGB在同一RAFT流下的光度残差、前后向一致性残差；做固定训练集尺度归一化。A/B使用完全相同的证据张量；仅在粗warp、offset输入及最终采样中，A用零base flow而B用RAFT base flow。这样对照的是位移引导路径，不应把A称为没有任何运动线索。不得将GT遮挡掩码用于q或采样有效性。

l_prime=l+beta*q*delta；训练beta=1；LoRA gain始终0.5。沿用旧生成器的conv_in/AdaGN条件接口，不同时新增AdaGN拆分或高频旁路。

buffer保存上一帧原始Bridge latent、必要参考/flow/帧元数据，不保存l_prime作为下一帧raw历史。TCG状态属于各候选独立维护。首帧/I/GOP边界、分辨率变化、历史缺失按原协议reset，缺历史时新增修正严格为0。不写codec DPB。四帧段内TCG保持跨帧梯度，段外detach，warmup无梯度。

## 6. 训练及loss

复用PREV合法训练源划分、32源/128窗口/256crop、QP0/5/9和前300步joint计划；不新增UVG/Vimeo训练，不改变已隔离开发/测试身份。A/B看到相同样本顺序、crop、QP与四帧segment。train-only统计，不使用开发集统计归一化。

两组均只训练新增offset/DCN/q/fusion，AdamW lr=5e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-4，grad_clip=1；有效batch=4个四帧segment，支持梯度累积，OOM时减microbatch但不改有效batch。模型前向、loss、新分支FP32；codec采用既有冻结精度。两组同seed20261113，记录非确定性算子的实际设置，不宣称CUDA grid_sample反向bitwise确定。

L_sp=LPIPS_Alex+0.5*DISTS+0.1383589248954108*RGB_L1
L_temp=原定义的有效区域GT参考MC-RGB变化残差+0.5*归一化Sobel变化残差
L_total=L_sp+4.63349060748*L_temp

明确删除旧0.1*L_align：基础RAFT冻结，9个取样offset不是9份同一GT-flow，不能把它们全部拉向同一向量。新offset由重建目标训练。lambda沿用只作A/B一致的实验起点，不是理论最优。不加GAN/滤波/额外新loss，不擅自加大时序权重。

每组最多300个已提交optimizer updates，保存0/1/10/50/100/200/300及latest，保存优化器/RNG/采样游标/协议/实现hash用于准确恢复。记录分项loss、参数组梯度、offset分布、mask/q分布、几何有效率、latent修正RMS、冻结审计。首次零输出上游梯度为0属于预期，输出头更新后必须核对上游梯度非零且有限。

## 7. 必要的启动检查

用小型合成和真实片段完成：正负/分数位移方向与向量单位；grid_sample/DCN坐标/分量顺序；全无效不NaN；中心恒等初始化；帧和分辨率reset；beta=0和训练step0与B300连续输出一致；训练模块梯度通过冻结decoder和TCG传回；A/B参数量和初始化一致；codec码流/feature hash一致；推理forward不接受GT字段。

这些检查要落实，不做漫无目的全仓审计。可复用旧源分组和基座审计，只检查复用条件/新模块变更。加载/维度/缓存键的普通实现错误自行修复；真实源文件缺失、权重损坏、信息泄漏或码流不一致时停止相关阶段并如实记录。

## 8. GPU并行与持久执行

默认GPU0训练A，GPU1训练B，GPU2生产当前训练计划所需参考RGB/flow缓存，GPU3做baseline/开发评测。任务结束后的空闲GPU可接管队列任务。先按训练顺序缓存最早batch，A/B只在自身下一batch所需缓存齐备后推进，不按缓存完成快慢改样本顺序。缓存、训练、评价分进程，进程间锁+原子checkpoint，禁止读半写文件。

沿用runner方式实现持续运行、日志、heartbeat、进程锁和退出码；不得仅启动后台进程就宣称完成。训练每到50/100/200/300产生评估任务，评估可异步；最多领先一个评估间隔，防止停止结果到达时已多训很久。读取GPU实际占用再分配，不kill其他任务。

结构或数值失败立即停相关组；质量停止只使用明确机械规则：相同4个ULong开发源、QP5、beta=1，连续两个已完成检查点被同一早期检查点在PSNR↑及LPIPS/DISTS/FloLPIPS/MC-Sobel/RWE↓全指标Pareto支配（至少一项严格更好，浮点相等容差1e-6相对），则该组停止并记录证据。若指标有取舍则不得只因某一项回弹停止，执行至300步上限。缺少未完成指标时不假设失败或0；暂停等待该评估即可。主观视频判断留给用户，不让执行端替用户决定。

## 9. 开发评测与可视化

沿用4个ULong开发源，扩展其既有合法窗口到64连续原分辨率帧，窗口起点/帧率/色彩转换/GOP一致。UVG仅用既有合法Bosphorus开发源并明确单序列；不扩充训练，不使用135个最终测试点。若64帧不足，选择原开发窗口中预先声明的共同合法长度并标清，不能重复帧补齐。所有候选共享相同真实码流，不能把旧16帧bpp与新64帧混用。

阶段点0/50/100/200/300（或提前停止的最后点）：A/B在4个ULong源QP5做beta=1、64帧连续评测；Original和B300同配置独立评测可验证后复用。

固定对照点100及A/B最后共同完成检查点：在4个ULong和Bosphorus的QP0/5/9补齐beta=1的R–D，不根据测试结果选择step。若两组均未到100，标明不完整，只用已完成的共同点。

在同两个固定可视化源24747b35、de12efc2、QP5，对上述固定点做beta=0/0.25/0.5/1扫描；beta0只需验证等于B300后去重。旧T100/T300在相同配置已有指标可复用，不把它们的旧16帧统计混进新图。无需另设大规模旧模型扫描作为前置。每个beta必须独立reset/rollout，不能插值最终RGB。

输出PSNR、LPIPS-Alex、DISTS、FloLPIPS、既有MC-RGB/MC-Sobel、边缘强度及GT值、高频能量/高频对GT误差；逐帧CSV、逐视频×QP×step×beta CSV、逐数据集等权汇总和基座配对差值。

增加VideoGigaGAN RWE：核对原文/官方实现，明确输入/光流方向/有效区和量纲；使用候选输出估计的运动在GT上计算相应误差，不能重命名旧MC-Sobel。不把RWE当完全无模糊偏置。若无法取得/核实精确定义，标记RWE_NOT_VERIFIED，不编造公式/数值，不阻断其余训练交付；停训规则对应退回其余已核实指标并在首个评估前冻结规则。

给B300添加仅诊断用的轻微RGB空间高斯模糊参照，sigma=0.25/0.5/0.75像素，滤波与边界记录；不用作正式候选或训练目标。画各视频LPIPS/DISTS—RWE/MC-Sobel取舍图，明确beta、step、模糊参照，展示非支配点但不擅自宣布最佳模型。

分别画ULong与UVG(Bosphorus单源)的码率横轴R–D，必须包括独立Original GVC-RT、B300、A/B固定点；不要把B300称原版，不把不同数据集混成一条线，不虚构HEVC/MCL结果。

固定两个64帧案例导出GT/Original/B300/A/B的原分辨率FFV1无损视频和MP4预览，以及带标签原生ROI对比；报告步数、beta、QP、真实码率、帧范围、FPS和校验和。原始树叶案例如复用，仅标历史定性回访，不用于训练或选模。

## 10. 性能、验收和GitHub

分别记录codec、独立参考RGB重建、双向RAFT、DCN、融合、最终decoder耗时/峰值显存及完整非缓存端到端耗时；区分训练缓存命中与部署延迟，不宣称零码流意味着实时。原始bits/features哈希、冻结参数、step0恒等、独立状态、预期评估点/帧数必须最终复核，检查NaN/缺失/重复行。

保存RUN_MANIFEST.json、PROTOCOL.json、status.json、REPORT.md（只写事实，不写研究结论）、全部指标/图、VIDEO_INDEX.md、checkpoint_manifest.json、large_artifacts.json、final_integrity.json、publication_status.json。用户要由分析端分析，执行端只计算和陈述。

已授权推送GitHub：仅向origin新分支codex/bridge-flowguided-deformable-ab-20261113正常commit/push，不force push、不合并主分支。显式暂存本任务代码、协议、依赖/复用凭据、报告、小型指标图表和清单；不要git add -A混入无关内容。大权重、数据、缓存、视频留PROJECT内，以路径/大小/SHA清单交付，不声称已上传大文件。核验remote branch commit及关键报告/CSV blob。网络/认证失败保留本地成果如实记录，不伪造发布成功。

## 11. 完成时的回复

持续执行到训练停止、约定评测、视频、最终检查和发布阶段都结束。中途只能报告RUNNING/部分完成，不能把后台pid当COMPLETE；不要要求用户再执行服务器终端命令。真正阻塞则准确说明阶段、已完成项、失败日志及恢复条件。

最终只返回简短JSON和必要路径：status、experiment_directory、A/B committed_updates/stop_reason、完成评估点与帧数、baseline_step0/frozen/bitstream检查状态、report/metrics/figures/video_index路径、github branch/commit_url/remote_verified/large_artifacts_uploaded、未完成项、以及仅在进程已退出时才适用的resume_command。不要替用户下科学结论，不自动启动下一轮实验。
