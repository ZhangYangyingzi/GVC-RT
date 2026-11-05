# 原版独立评测：评分转换修复后重测中

此前提交 ee83979dc838743ae6570d1a24c07ef6ee3b81fc 的135点评分及49张图已失效，不能用于当前对比。原因和原始记录见 audits/precision_alignment_v2/repair.json 及本地 invalidated_v1/。

原版固定源码与I/P权重未改动；修复仅复用既有评分的FP32范围转换函数。当前将完整重新编码并独立解码135点。实际进度见 run_status.json；最终验收见 final_integrity.json。
