# GVC-RT V1.5 — Frame Budget Allocation

冻结全部网络，在 l3 的 `{Z,E,O1,O2}` 有限候选集上执行显式逐帧跳过和真实 ORC2 GOP 字节预算动态规划。Train2 复用已核验候选；Val6 实际新增 O1/O2 共 180 次码字优化。

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_latent_enhancement_v1_5_frame_budget_allocation/run.py --output reproduction_01 --phase all
```

输出目录必须不存在。固定候选、lambda、预算、并列规则和 ORC2 字节模型见 `config.json`。发送端允许整 GOP 前视与缓存，属于离线诊断；接收端仍按帧因果执行。
