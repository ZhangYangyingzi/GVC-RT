# Generator Conv-LoRA 执行记录

状态：**BLOCKED — PILOT_SHA_MISMATCH**。正式训练更新0；开发集0/96；最终测试0/135。未实例化完整模型，也没有运行后台训练。

## 基线阻塞

本轮指定pilot SHA：

```
06cf8f53f432d1f1f0fed7e8a72e911af70f5ee1d2a0b869ceaabef94b1f8744
```

指定文件实测SHA：

```
06cf8f53f432d1d1f0fed7e8a72e911af70f5ee1d2a0b869ceaabef94b1f8744
```

实测值与上一实验protocol.json一致，但与本轮消息指定值不一致。按用户第18条停止正式训练，不自行更改预期SHA、不更换文件。B+1500和原始I/P权重SHA均通过。

## 已完成

- 基于6c3747d0e960f2fed656b509d4537e1226045cbe建立独立worktree及指定新分支，原有worktree与用户修改未动。
- 只读核验继承的split/source manifests及120个计划分片SHA，不重新划分数据。
- 源码确认stage3四个目标均为320通道depthwise3×3；head是320→192的dense3×3。
- 准备5个并行LoRA支路：stage3为320→4→320，head按实际输出为320→4→192，rank4、alpha/r=1、无新增bias、第二层零初始化，共59392参数。stage3是depthwise层旁的并行更新，不是dense层权重的直接分解。
- 准备stage3和head的PyTorch执行适配，baseline与candidate必须使用同一适配，防止CUDA proxy绕开LoRA；真实GPU/CUDA执行一致性尚未验证。
- 合成CPU独立算子检查通过：零初始化严格一致、LoRA梯度及参数变化、冻结base无梯度且不变、half转换保持LoRA FP32。3次临时合成线性VJP更新全部丢弃，不计正式训练，没有使用实验模型或质量loss。
- 历史loss.py按原SHA复制，loss权重和VJP4096处理未修改。

## 未执行

严格完整模型重建、真实模型零初始化等价、真实各loss项对LoRA梯度审计、GPU/CUDA执行检查、正式训练、96点开发集验证、135点测试、真实bitstream/bpp核验及R-D作图均未执行。合成CPU检查不能代替这些阶段。没有实验质量数值，也没有科研结论。

## 恢复前提与命令

用户确认将本轮预期pilot SHA更正为实测及上一协议的SHA，或提供与本轮指定SHA完全匹配的文件。确认前不绕过检查。下面命令只复核预检，不启动训练；正式持久化runner尚未完成或启动：

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_generator_convlora_20261106/experiments/gvcrt_tcg_stage2_generator_convlora_20261106/preflight.py
```

## 产物

`preflight.json`记录各权重SHA；`audits/baseline_sha_mismatch.json`记录差异；`audits/inherited_manifests.json`记录封存来源核验；`audits/cpu_lora_operator_probe.json`只记录独立CPU算子检查；`protocol.json`是未冻结草案；`status.json`与`final_integrity.json`均明确BLOCKED；`metrics.csv`只有表头；发布状态由`publication_status.json`独立记录。阻塞记录发布不代表实验完成。
