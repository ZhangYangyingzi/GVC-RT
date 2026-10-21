# Protocol

G_L1 从指定原始 step0 开始，只训练 Generator 100 次；复用 v6.40 的不可变 first100 计划、优化器设置与评测码流。目标为 `0.1383589248954108 * reconstruction_L1 + beta_q(schedule_s1p0) * rate_bpp + 0.01 * proxy_L1`，不含 LPIPS 或 DISTS。O_original 与 G_basic 固定结果来自经哈希核验的 v6.40。
