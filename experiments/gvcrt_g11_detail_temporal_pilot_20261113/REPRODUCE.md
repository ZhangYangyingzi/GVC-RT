# G11复现与恢复

实验目录为本文件目录。Python固定为 `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`，在此目录执行 `-B manage.py status`查看，`-B manage.py resume`自动接管存活runner或恢复原子checkpoint。完成后恢复为空操作，不开启下一轮。用户无需手动分阶段启动。

只运行一个G11 trainer，初始物理GPU2，临时外部显存OOM后按0–3空闲显存排队/迁移，实际卡见completion.device与runner日志；最多一个有界18源缓存生产进程用GPU0。GPU0–3有授权，不使用4–7，不终止无关进程。所有缓存、CUPY/CUDA/编译/临时文件都在实验内。ROOT、PROJECT明确固定，不导入旧实验会向旧ROOT写入的util。

完整读取sources/TASK.md和PROTOCOL.json。来源只含父协议12train、6validation，QP5；本地data_manifest不含final_test六源。sample_schedule是全新300个连续四帧clip，每12步seeded shuffle所有训练来源，固定ROI、start及首4训练校准clip，不复用父两个2帧clip的schedule。

dependency_snapshot记录父实验实际源码SHA与frozen_sources权威核对；runtime_tree/original_tree/vendor都从实际父运行环境隔离快照，未假定Git HEAD等于有效代码。权重按weight_manifest及父checkpoint_integrity核验：Original→pilot→B1500 TCG→A2000 LoRA(s=.5)→旧B300→D12-S@900。后者不是完整模型，也不是flow-guided B300。模型整体与S的实际状态SHA见audits/preflight。

父冻结codec/Original/metadata缓存只读，来源、流向、像素中心8x+3.5、FB规则保持一致。新缓存保存冻结prefix、完整S输出、冻结S.E的原始当前/过去内容及固定GT监督流，不缓存候选decoder激活。缓存按计划生成并原子提交；checkpoint和缓存都带SHA凭据。旧源路径缺失时六个validation源仅从既有只读ZIP定向恢复相同SHA，以核验FPS，见sources/validation_source_recovery.json；不改变样本身份。

G11的精确模块和988800个原生接收块白名单见model.py及audits/runtime_inspection.json。S完整前向condition=S不拆开，TCG/LoRA/S/其他主干均冻结，后缀走整张padded图并保留输入梯度，GN不在局部ROI上前向。functional TCG_out状态在四帧内保留图，clip外/reset断开；checkpoint stage3/stage4不更新buffer。新历史只读原始c_prev，不读GT/GT-flow或候选RGB。

objective.py保存空间/固定GT参考时序误差/软预算的精确定义。正式训练前固定m_j；step50保存后训练clip梯度范数定标lambda，不更新权重。LR、mu、lambda渐进和机械停止规则不依据validation中途调整。每个milestone完整六源验证后才续训。

checkpoint包含branch、native_receiver、optimizer、RNG、mu、lambda_calibration、LR、schedule位置、协议/实现SHA。latest在完整clip更新后原子保存；0/50/100/200/300另保存。初始smoke3步后恢复模型/optimizer/RNG，不算正式更新。日志原文件在logs/train.jsonl，提交用CSV在metrics/training_updates.csv。

仅显式validation6，47/48 suffix预热，49..63计分。Original基线用独立原版I/P+原版实现并FP32重建；父推理对齐metadata仍为其已审计Original FP16参考，二者明确区分。各模型独立TCG_out滚动。同一64帧码流bytes/bpp用于所有15帧计分结果。指标/图为QP5单点，不是完整R-D。15帧视频不作长时稳定性结论。

大型checkpoint/cache/data/FFV1/MP4只在本地项目内，large_artifacts.json记录绝对路径/大小/SHA与本入口。代码/协议/审计/CSV/小图/报告/索引提交独立origin分支；发布失败独立记录，不与训练失败混同。未经授权不启动下一轮。
