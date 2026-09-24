# GVC-RT Latent Enhancement V1

独立的 P 帧显示层 latent 增强实验。结果与结论见 `report.md`；计算图、状态隔离、量化格式和度量口径见 `audit_notes.md`。

## 复现实验

在已核实的本地 `gvc-rt` 环境中执行一条命令：

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_latent_enhancement_v1/reproduce.py --run-name reproduction_01
```

复现前校验固定源帧 SHA256、完整基础 checkpoint 和阶段 A 门槛，并检查配置的唯一 GPU 是否空闲。已有输出目录会报错，应使用新的 `--run-name`。本地原视频/模型缺失时不会下载替代资源。阶段门槛失败时停止后续训练。

配置中 B/C/D 分别为 500/1000/1500 个配对训练迭代，总计最多 3000；增强模型和无码流修正对照各自得到相同更新次数。D 使用固定 Train8 而非全训练集。增强量化网格固定为 8×17×30，delta=0.125/0.5/2.0。

## 文件

- `config.json`、`manifest.json`：预先固定的参数、Train2/Train8/Val6 视频与帧/时间戳/有效区域、源 PNG hash。
- `prepare_data.py`：原生 1080p 检查和固定视频级划分；`data/` 为选定连续帧。
- `codec.py`：严格权重加载、原版真实基础编码/解码、只读 ell 提取；不改变原测试入口。
- `enhancement.py`：轻量 E/D、base-only 对照、均匀量化、因子化概率模型、真实算术编码。
- `receiver.py`：源数据访问受阻的独立接收进程；省略 `--enhancement` 时完全旁路增强网络。
- `run_audit.py`、`stage_a_v2/`：真实码流、参考状态、I/P/reset、旁路、梯度、权重 hash、整数编码审计。
- `stage_a/`：保留的首次中断调试记录；这里的增强 stream 是已淘汰的 rANS 格式，不是最终评价点。
- `run_experiment.py`、`run_v1/`：阶段 B/C/D、检查点、逐步日志、完整基础流、真实增强流、独立解码输出、质量/码率 CSV、FFV1 对应视频。
- `evaluation.py`：有效区域质量指标、可视化输出、带全部元数据的增强序列容器。
- `benchmark_e2e.py`：包含发送端获得基础 ell 的真实端到端计时。
- `supplemental.py`：补齐 Train2 原版 QP 对照和配对可视化。
- `summarize.py`、`analyze_streams.py`：实际字节汇总、重叠区间插值、传输整数符号利用率。

## 单独的接收端

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python receiver.py \
  --base /absolute/path/sequence.bin \
  --enhancement /absolute/path/sequence.gle \
  --checkpoint /absolute/path/checkpoint.pt \
  --output /absolute/path/new_receiver_output.pt
```

接收端不接受原视频/源帧参数；`--enhancement` 与 `--checkpoint` 同时省略时输出原版基础重建。共享基础 checkpoint 和固定配置由本实验目录解析。

连续增强的质量值只作优化诊断，不能视为不付增强码率的可传输结果。本版也没有可截断的渐进码流。计费包含 `.bin` 基础 SPS/NAL 和整个 `.gle` 文件；性能工具明确区分缓存评估与真正端到端开销。
