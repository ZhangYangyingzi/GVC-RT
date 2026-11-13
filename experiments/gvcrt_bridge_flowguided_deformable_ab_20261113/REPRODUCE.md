# 本轮复现与状态入口

实验目录为本文件所在目录，所有新临时文件、缓存、权重、日志、媒体均留在此目录下。Python固定为 `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`，物理GPU只允许0/1/2/3。

在本目录用上述Python执行 `-B manage.py status` 查询，`-B manage.py verify`只读验证已提交检查点与点数；`-B manage.py resume`接管本任务存活PID并仅消费原子提交条目，任务已完成时只读核验并空操作，不重新训练、不启动下一轮。实际runner调用cache_train.py、train.py、evaluate.py、finish.py；用户不需要分阶段开终端。恢复拒绝不匹配协议/实现/数据计划或无效hash的checkpoint，不猜测模型身份。

来源见RUN_MANIFEST.json、PROTOCOL.json、sources/TASK.md、sources/下旧审计与未跟踪说明的本轮快照。起点为PREV所属worktree已提交HEAD71ed7e74，未改旧实验或主工作区。restore.py和vendor/model_context.py严格恢复Original→pilot→B1500 TCG→A2000 LoRA(s=0.5)→B300最后增量，源权重SHA由既有协议/凭据读取后实测，基座参数完整SHA见audits/preflight.json。

训练源、128个窗口、先RGB裁剪的256×256空间域、QP0/5/9和300步joint计划完整复用data_manifest.json/training_plan.json，32来源组隔离不变。旧cache/train仅提供冻结codec feature、GT监督流和I帧参考；Original P帧参考在本轮重新独立重建并做真实码流逐帧等价验证，冻结RAFT只读独立解码参考RGB，绝不以GT补齐。新sidecar字段明确为decoded_ref_RGB/backward/forward/evidence，与evaluation_supervision内GT字段分离。字节重排压缩可逆；latent-grid字段已完成位移单位换算，同尺寸第二次resize数值相同。

A/B均为258,372个新增参数，seed20261113，AdamW lr5e-5，固定lambda_temp4.63349060748，只有空间与GT参考时序重建损失，没有L_align。checkpoint保存model/optimizer/RNG/计划游标/协议/实现/统计SHA；0/1/10/50/100/200/300均保存。source_implementation_final.json保存最终代码身份；运行中实现hash仍保留在各checkpoint内部。grid_sample/DCN CUDA反向warn_only，固定seed不等于反向逐位确定性。

64帧评测从每视频I帧reset独立流式滚动，保持原分辨率和FPS。每个QP只有一个真实codec码流，源/时间窗口/实际QP/精度/权重/flow配置进入缓存身份。eval_gpu0/1仅在训练组等待评测时消费该组任务；GPU3消费基线与队列。模型0步及beta0连续64帧与B300精确核验；beta0去重只影响显示，不替换因果重放。

所有科学指标使用量化前FP32 RGB[0,1]；LPIPS_Alex normalize=True；DISTS冻结；FloLPIPS为已审计官方PWC+Alex兼容实现；MC-RGB/MC-Sobel使用固定GT-RAFT与有效掩码，Sobel有符号/8。RWE_NOT_VERIFIED不造数；机械停止采用首评测前固定的其余5指标。RWE相关原文/官方链接与核验缺项见PROTOCOL.json和REPORT.md。

固定R-D展示100与最后共同300点，ULong四开发源与UVG Bosphorus单validation源分别绘图。两例beta0/0.25/0.5/1独立rollout；B300 Gaussian sigma0.25/0.5/0.75仅诊断。没有最终135点评测或第二seed/第三训练组。

媒体在videos/；render_missing.py只补未导出的beta1资产，并核对1/32/63浮点MSE；media_finalize.py做RGB8字节级无损回读、MP4和原生ROI，标题栏独立、不缩放内容。VIDEO_INDEX.md给出完整身份、绝对路径、QP、码率、步数、beta、帧号、FPS、哈希。MP4为有损播放副本，FFV1才是无损交付。

performance.py实测非缓存RAM码流完整P1..P63：codec、独立Original RGB、双向RAFT、DCN/offset、融合、完整最终decoder及总量；profile_parts.py补充独立组件峰值。不要把训练缓存命中或分支耗时当部署端到端速度。

普通Git只提交本任务代码、协议、依赖/复用凭据、JSON/CSV、小型图表、报告与索引。数据、码流、缓存、权重、大型FFV1/MP4和运行二进制不上传；large_artifacts.json给出本地绝对路径/大小/SHA与复现方式。GitHub发布及远端报告/CSV blob核验见publication_status.json。

最终离线MC有效域修正见audits/native_validity_correction/plan.json及complete.json。原始数值完整保留在before/；fix_native_metrics.py只修正原生RGB采样有效域并重算72点的MC-RGB/MC-Sobel，其他指标和训练权重保持不变。若该最终整理进程意外退出，可在无本任务存活进程后执行reconcile_finalize.py；它跳过已提交的修正点并继续finalize.py，不启动训练。

源路径迁移：旧ULong目录在最终离线指标修正期间消失，4个开发源从既有只读ZIP以完全相同SHA恢复到cache/recovered_sources。raw_frames只在旧路径缺失时按source_path_migrations.json查找同哈希替代路径，不改变冻结development.json/协议身份，不覆盖旧目录。
