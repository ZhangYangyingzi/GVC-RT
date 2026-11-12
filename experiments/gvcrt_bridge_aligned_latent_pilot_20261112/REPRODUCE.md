本轮是固定预算先导实验，不运行最终135点评测，不进行效果选模或配置搜索。

项目内入口：使用 `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B manage.py status|verify|resume`。所有命令均在本实验目录执行。`resume` 在完整交付后只读核验并空操作；运行中的锁会阻止重复启动。

`protocol.json` 固定结构、损失和更新预算；`training_plan.json` 固定样本、QP实例与监督短段。`data_manifest.json` 记录32个训练来源组的128个原图先裁剪窗口，384个QP实例；统计只取该训练清单中预定8个实例，冻结在 `cache/statistics.pt` 与 `audits/statistics.json`。ULong快速开发评测是16帧原分辨率连续片段，从其初始I帧完整滚动；额外UVG validation Bosphorus同为16帧，使用原生120fps连续帧。16帧短片不等于完整原视频，时间覆盖有限。

B300严格复用 `restore.py` 的已审计恢复函数和 `vendor/model_context.py`，Original→pilot→B1500 TCG→A2000 LoRA→B300最后覆盖。预期权重身份从旧协议读取，见 `vendor/common_s2.py` 末尾和 `audits/preflight.json`；路径迁移见 `audits/path_migrations.json`。训练仅新模块，LoRA gain固定0.5。Original独立代码在 `original_tree`，不加载pilot/TCG/LoRA/B300。

重建、新模块、损失是FP32参考路径；codec保持已审计FP16。所有对照同精度。CUDA grid_sample反向按原有协议设置 deterministic warn_only；固定种子不保证CUDA原子累加反向逐位确定性。不会下载新权重；数据集与旧检查点只读引用。系统ffmpeg用于正确色彩转换和视频导出。

大缓存、模型权重与视频留在项目内；普通Git仅提交源码、清单、CSV/JSON、报告、图表和视频索引。`large_artifacts.json` 列明未上传大文件的绝对路径、字节数和SHA256。历史运行前实现错误及修复记录在 `audits/implementation_corrections.json`，不删除旧错误日志。

重建缓存：`cache_data.py --shard {0,1,2}` 对应三个固定训练分片；`--shard 0 --dev` 包含开发QP5/0/9；`--dev --extra` 只生成预固定的额外UVG开发源。每个进程须设置一个获准物理GPU的 `CUDA_VISIBLE_DEVICES`。runner会原子提交/校验缓存后消费，不改变样本次序。

训练入口：`train.py --phase aligner` 最多500步，`train.py --phase joint --group S_current|U_unaligned|T_aligned` 各最多300步。调用者无需分阶段手动运行；`runner.py` 调度。`delivery_runner.py` 管理真实码流计时及最终评测后的图表/视频/报告。两者只管理本任务自己的进程。

评测入口：`evaluate.py` 调度三个单卡Python worker。每个模型/视频/检查点独立复位新增缓存与TCG。GT/教师流只进入损失和离线指标；推理模块输入签名在 `model.py`，不接受GT/教师流/未来帧。未使用缓存B300隐藏状态替代候选滚动。

精确损失归约：空间项按4帧均值、再按4短段等权；时间项是每段内部3个有效帧对的均值，再按4段等权；无有效像素的帧对不进入均值。匹配Charbonnier为各位移分量 `sqrt(d^2+1e-6)` 有效均值，尺度权重1/0.5/0.25；空有效尺度贡献图连接的零，EPE记空值。QP使用实际逐帧QP，训练统计固定归一化。任何中途图像质量波动都不更改结构或loss。

两个预先固定的视频诊断案例另按已有64帧开发片段完整输出，见 video_manifest.json、media64.py 和 media64_runner.py。定量快速开发协议仍为16帧；该延长仅用于可视检查，不改变训练或选模。
