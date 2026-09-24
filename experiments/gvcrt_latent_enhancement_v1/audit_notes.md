# 本地代码审计与实验契约

## 资源与指令

- 仓库：`/Huang_group/zyyz/Projects/GVC-RT`，commit `d0e32bfa3e8e282f9a77437c223c605858eea637`。
- 在 `/`、`/Huang_group`、`/Huang_group/zyyz`、`Projects` 及目标仓库/子目录未发现适用的 `AGENTS.md`；仓库内又以 `rg --files --hidden --no-ignore -g AGENTS.md` 核实。
- 原始工作树已有 `real_data`、smoke 及 VIRAT 结果；实验只新增 `experiments/gvcrt_latent_enhancement_v1`。未修改任何原始 checkpoint 或测试入口。
- 环境 `.../home_dir/.conda/envs/gvc-rt/bin/python`：Python 3.12、PyTorch 2.6.0+cu124；完整版本在 `stage_a_v2/pip_freeze.txt`。
- 唯一 GPU：物理 0 / `GPU-9f8e23af-ce47-d03e-07a6-2e89cc8ef3bb`，A100 PCIe 40GB；启动前 0% 利用率、无计算任务。其他 GPU/进程未处理。
- `GVC-RT_I.pt` 596/596 键、`GVC-RT_P.pt` 537/537 键均 `strict=True` 加载，键及形状完整；SHA256 在 `stage_a_v2/audit.json`。
- 本地基础 C++ 熵编码扩展可用；`inference_extensions_cuda` 不存在，官方入口实际使用 PyTorch fallback。没有下载模型/数据，LPIPS AlexNet 权重来自已有 torch hub 缓存。

## P 帧的真实计算图

`src/models/video_model_gvcrt.py`：

1. `compress`：基础参考适配/上下文 → `enc(x, ctx, q_encoder)` → **y**（128 通道）→ hyperprior z → round/熵编码；`compress_prior_2x` 给出真正量化后的 `y_hat`。
2. `dec(y_hat, ctx, q_decoder)` → **feature_base**。`compress` 将该 feature 写入 DPB，不生成本帧显示图。
3. `decompress` 从基础字节流恢复 z/y，调用 `get_recon_and_feature`。
4. `PretrainedReconWrapper.forward`：`pixel_unshuffle(feature,2)` → GroupNorm → DepthConvBlock → GroupNorm → swish → 最后一个 DepthConvBlock，得到 **ell_base/codeword**。
5. 原版 `self.decoder(codeword, quant_step)` 是本轮 G。增强仅使用 `ell + D(u_hat,ell)`，不改 ell 空间布局。

原生 1080×1920 图像复制填充底部 8 行为 1088×1920，真实观测：

| 张量 | NCHW |
|---|---|
| y_hat | 1×128×68×120 |
| feature_base | 1×256×136×240 |
| ell_base | 1×18×68×120 |
| q_recon | 1×320×1×1 |
| G 输出 | 1×3×1088×1920 |
| 有效 RGB | 1×3×1080×1920 |
| 增强 u 固定网格 | 1×8×17×30 |

`feature_base`、`ell_base` 未限制在 [−1,1]；例如被审计 P 帧 feature 范围 [−8.484375,8.0078125]、ell 范围 [−1.267578125,1.2666015625]。不能与 y 或图像直接相减。RGB 输入按官方入口从 [0,1] 转为 [−1,1]；G 输出裁剪到 [−1,1]，质量评价转换为 [0,1]。

逐帧 `index_map=[0,1,0,2,0,2,0,2]`，`qp_shift=[0,2,1]`，因此 I 后 P 帧的偏移为 +2,+0,+1,+0,+1,+0,+1,+0…；每帧 q_recon 取**偏移后的 QP** 对应切片。q_recon 是 320 通道学习向量，可含负值，并非增强均匀量化的标量 delta。

## 基础状态与梯度

- I 帧清空 P DPB，写入原版 I 重建；P 解码更新原版 feature/RGB。
- `prepare_feature_adaptor_i(last_qp)` 在发送端需要时以原版 G 重建参考 RGB、清除 feature；接收端对应 `reset_ref_feature()`。本轮常规间隔 96，另用独立 12 帧审计流、重置间隔 4/I 周期 8 覆盖这些分支。
- 增强接收端先完成基础解码及其 DPB 更新，再生成显示用增强 RGB。增强结果不回写 DPB。生成 latent 由只读 pre-hook clone 获得，增强使用非原地加法。
- 发送端的 ell 从**实际序列码流的本地解码**获得；不是从未量化编码特征估算。训练缓存只是这种已解码 ell/q 的副本，含对应帧 QP、SPS/重置、参考张量 hash。
- 接收端独立进程只读共享 checkpoint、基础/增强字节流；审计钩子禁止访问源图/源视频/基础训练 `.pt` 缓存。
- 可微路径为纯 PyTorch G，参数冻结但输入不置于 `no_grad`。源网络与训练 G 各自严格从 checkpoint 构造，不把可能原地融合后的实例转回训练。原版 encoder 的 `forward_cuda` 确有 in-place 权重融合，CUDA proxy 不用于反传。
- 同精度 FP16 的官方 fallback 与独立 PyTorch 重建精确一致。FP32 G 对相同 FP16 ell/q 的 RGB 差异：最大绝对值 0.010620698，MSE 1.5030581e−7。增强收益一律首先对比同 FP32 G 的参考输出，原版 FP16 结果另外列出。

## 增强编码与计费

- E：4 次 stride-2 RGB stem、与 ell 融合、轻量深度可分离残差块，固定池化到 17×30；D：双输入融合、2 个残差块、零初始化输出头；没有教师或其他 tokenizer。
- 训练 STE，正式编码真实 round。增强模型只接受 int8 范围的整数，超范围**抛错**，不裁剪。
- 因子化零均值 logistic，每通道一个可学 scale；复用原库的 PMF→16bit CDF 转换，实际字节由独立 32bit 算术编码器产生。
- 原生 rANS 的输出缓冲仅按符号数量分配（`src/cpp/py_rans/rans.cpp:221`），不安全地支持任意增强数据。首轮边界测试挂起/超时已保留在 `stage_a`；修复后的完整阶段 A 证据在 `stage_a_v2`。基础流仍使用原生 coder。
- 单帧 GLE1 头 24 字节，含版本/通道/档位/网格/delta/CDF指纹/长度；算术终止与字节对齐也计费。
- 序列 GEC1 头 44 字节，含有效尺寸、帧数、基础流 SHA256；每个 I/P 帧另有 4 字节长度。I 帧增强 payload 长度为零。统计总 `.gle` 文件全部字节。
- 无增强字节流时，`receiver.py` 完全不构建增强网络，输出基础 codec 解码图；“全零增强符号”不被视作无增强。
- 量化档位 [0.125,0.5,2] 是独立多码率点，不是可截断渐进码流。

## 数据与指标

用户在执行中指定 `.../tianyi/UltraVideo-Long`。使用 `clips_long_1920` 中经 ffprobe 验证的原生 1920×1080 MP4；未使用先前 VIRAT/Cosmos 实验的分辨率/帧率配置。

未找到适用 UltraVideo 既有划分，固定 SHA256/seed=91401 的 UUID 视频级划分；Train2、Train8、Val6 在任何模型结果之前写入 manifest，前 16 连续帧从源 clip 的第 0 帧开始。每段包含完整 1 I+15 P，无片段中途重置。原始父视频映射未知，不能排除不同 UUID 来自同一长视频；不把这组 Val6 当作最终测试集。

Train2 第一段为游行/人群、小人物与树叶纹理，第二段为有相机运动的室内穹顶；源帧运动诊断非零。所选实际 fps 包含 25、30000/1001、30，使用 ffprobe 帧时间戳计算 16 帧片段时长。

- RGB PSNR、仓库 `calc_msssim_rgb` **MS-SSIM**、LPIPS Alex v0.1（`normalize=True`，输入 RGB [0,1]）。排除 8 行 padding，所有方案一致。
- PSNR/MS-SSIM/LPIPS 先逐帧再等权平均，包含 I 帧；所有片段均 16 帧，所以视频等权与帧等权相同。
- 总 bpp = 全部基础 `.bin` 与增强 `.gle` 实际比特 / 原始有效像素总数；基础 SPS/NAL 头已包含。
- 汇总 kbps 使用各片段比特之和 / 各片段真实时长之和 /1000；不硬编码 30 fps，不用 padded 面积。
- 连续增强只列诊断质量，不能当作零增强比特的可传输 RD 点。
- 同码率比较按每个视频的真实采样点，在重叠范围内做 PSNR 对 log(bpp) 的分段线性插值；不外推。
- FFV1 review 视频是有效区域的 8bit 无损可视化，指标在写视频前浮点 RGB 上计算。未采用 FID。
- 未配准的相邻重建误差差分 MSE/RGB 偏差随帧波动仅是时间稳定性诊断，不足以证明没有闪烁。16 帧短片段不足以推断长序列稳定性。
