# 当前执行状态

准备与 CPU 核验已完成；原版完整 GPU 评测尚未开始，0/135 点，没有新增原版 R-D 曲线。

45 个视频的 2880 帧 RGB 校验通过。原版代码固定在独立目录，I/P 权重加载成功且无 TCG 参数。PWC/FloLPIPS 权重哈希与原评测一致。

启动 GPU 4/5/6/7 的自动审批服务返回 `stream disconnected before completion: Upstream request failed`，命令未执行。这是审批服务故障，不是安全判定。没有绕过审批启动 GPU 任务。

入口与恢复方式见 README.md。恢复运行后才会产生 metrics_original.csv、REPORT.md、plots/ 和 final_integrity.json。不要将当前准备文件视为已完成评测。
