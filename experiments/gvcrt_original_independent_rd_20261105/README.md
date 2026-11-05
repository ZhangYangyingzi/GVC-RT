# 原版 GVC-RT 独立测试与 R-D 曲线补充

目标：为已有 45 个视频、QP 0/5/9 的 TCG 对比补充 135 个原版实测点。在全部实测和完整性检查通过后，生成 4 张按数据集/划分分别绘制的图和 45 张逐视频图，新增 `GVC-RT original (independent)` 曲线。

原版实现固定为 `b2a0895c0cf31cf2be3ab1c2571ca07c05866c03`，独立源码目录为项目内的 `GVC-RT_worktrees/gvcrt_original_independent_20261105`。直接读取项目 `checkpoints/GVC-RT_I.pt` 和 `GVC-RT_P.pt`；不加载 v6、TCG 或 A/B 检查点。所有导入的 `src.*` 模块必须来自此原版目录并与固定源码哈希相符。

每点独立产生真实 RANS 码流，再用新建解码器读取持久化码流。编码时同步解码与独立解码的逐帧输出哈希必须相等。图中原版点来自实际测量；不会复制或改名 parent0 数据。与 parent0 码流或质量不同不会被强行改为相等。

已有 TCG 数据保持原样。原版测试沿用相同视频、64 帧窗口、RGB 预处理、QP 调度、FP16 编解码、FP32 质量指标及已有 GT-flow 定义。共享的是数据和评分标准，原版编解码器本身不含 TCG。这里不声称复现论文全部默认测试设置，也不把子集称为完整数据集。

所有新输出都在本目录。程序不训练。GPU 使用此前授权的物理卡 4、5、6、7，每卡一进程，不终止其他任务。

运行入口（可断点恢复，完成后校验、作图并尝试上传既有 GitHub 分支）：

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_original_independent_rd_20261105/runner.py
```

`run_status.json` 为实际状态。只有 `final_integrity.json` 为 PASS 且 `metrics_original.csv` 有 135 点，才代表完整评测完成。`REPORT.md` 和 `plots/` 仅在完整实测校验通过后生成。GitHub 上传结果另记 `publication.json`，权重、视频和码流不上传。
