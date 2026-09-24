# GVC-RT V1.4 — Rate-Gap Diagnosis

本实验冻结全部历史网络，分析 V1.3 整数码字的码率跳变，实现独立 ORS1 无损稀疏格式，并按门槛补充有限中间 lambda。网络参数更新次数固定为 0。

## 复现

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_latent_enhancement_v1_4_rate_gap_diagnosis/run.py --output reproduction_01 --phase all
```

输出名必须是不存在的同目录 basename。运行前需确认配置指定的 GPU4–7 之一具有至少 12 GiB 可用显存。历史目录、权重和码流均只读。

ORS1 对全零帧跳过；非零帧在原密集算术编码和“位置差分 canonical LEB128 + 有符号值 zigzag canonical LEB128”之间按实际 packet 字节选择更短模式，平局退回密集模式。模式、数量、长度、CRC、身份绑定均进入真实码流。
