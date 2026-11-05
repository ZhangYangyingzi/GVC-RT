# parent0 基线身份核查

依据用户于 2026-11-05 确认的原版目录 `/Huang_group/zyyz/Projects/GVC-RT/checkpoints`，本次直接读取检查点张量，而不是仅根据实验协议的文字或文件名判断初始化。

结论：parent0 的非 TCG Bridge 和生成器权重，与用户确认的 `GVC-RT_P.pt` 在原有加载精度下完全一致。它经由 v6_34 的 step 0 保存，但不包含 v6_34 后续训练的权重更新。这里只指实际使用的这一份 step 0，不对 v6_34 其他检查点作此判断。

## 权重链与证据

`checkpoints/GVC-RT_P.pt` → 加载器 FP16 转换 → Bridge/生成器转回 FP32 → `v6_34/N_noP_BG_FD/adaptation_step_0.pt` → TCG pilot step 0 → expanded step 0 的 TCG 状态 → 评测中的 parent0。

v3 评测在 `v3_evaluate.py` 中先读取 pilot step 0 的 `models`，再用所选检查点的 `tcg_model_state` 覆盖生成器中的 TCG 参数。parent0 选用 expanded step 0；该 TCG 状态与 pilot step 0 完全一致。

| 比较项 | 张量数 | 结果 |
|---|---:|---|
| v6_34 step 0 Bridge vs 原版 P 权重经 `half().float()` | 28 | 全部逐元素相等，最大误差 0 |
| v6_34 step 0 生成器 vs 原版 P 权重经 `half().float()` | 212 | 全部逐元素相等，最大误差 0 |
| pilot step 0 的上述参数 vs v6_34 step 0 | 240 | 全部逐元素相等 |
| expanded step 0 TCG vs pilot step 0 TCG | 全部 TCG 状态 | 全部逐元素相等 |
| 三个 TCG 输出层的权重和偏置 | 6 | 非零元素数均为 0 |

v6_34 该文件内的 `adaptation_step`、`committed_optimizer_updates` 均为 0，优化器 `state` 为空。张量比较为主要证据，这些元数据只作补充。

直接对比原版保存的 FP32 张量时，Bridge 最大差异约 0.00024402，生成器约 0.00042212。这些差异全部由加载器的 FP16 舍入解释；经过同样转换后误差为零。不能将这种精度转换误判为训练更新，也不应声称原始 FP32 文件字节完全相同。

## 输出检查及边界

从仓库 `b2a0895c0`（Initial public release）读取原始 receiver 模块，与当前零残差 receiver 做 CPU FP16 检查：QP 0/5/9，各连续 3 个合成压缩特征输入，共 9 次输出逐元素相等，最大误差均为 0。当前 TCG 缓存在连续帧中启用，因此也覆盖了有历史缓存时的零残差行为。

该检查使用小尺寸合成特征和当前共享层实现，只检查 Bridge/生成器 receiver；没有独立重跑完整原版编码器、算术编解码器及全视频测试集。因此新图称 `Original-weight baseline (parent0)`，准确表示原版权重基线；不称作原版代码全测试集独立运行结果。

以该目录为原版权重参照，现有曲线可以用于分析相对原版权重基线的 TCG 增益。如果要报告严格的原版实现复现，应再用固定原版代码、相同视频/帧范围/码率协议独立评测。

## 可复核产物

- [逐张量核查与输出检查](baseline_identity_audit.json)：检查点 SHA256、比较数量、精度差异、9 次输出结果、代码来源。
- [核查脚本](audit_baseline.py)：仅 CPU 读取与小规模 receiver 推理，不训练、不修改历史检查点。
- [数据集及逐视频曲线](README.md)：4 张分组曲线和 45 张逐视频曲线。

全部新增产物位于 GVC-RT 目录内。核查只读访问旧 expanded step 0 的既有位置，没有向临时目录写入文件。
