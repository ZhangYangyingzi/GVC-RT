# GVC-RT 抗闪烁与画质保持下一步完整执行手册（H 系列实验方案）

本文档基于前期全部实验证据（`A/B/C/D`、`E/F`、`REPORT_2.md` 原文复核、`REPORT_3.md` 零训练闪烁源头诊断、以及 `REVIEW.md` 的 `G0/G1` 100 步核查），给出接下来唯一聚焦、可直接落地的工程与算法执行方案。

---

## 一、 前期实验核心定论：为什么之前的方法走不通？（四条铁律）

在启动下一步训练前，必须明确前期实验已经用真实数据证明的四条物理边界，避免重复踩坑：

1. **铁律一：绝不能在无稳定分布约束下全量解冻 `de-tokenizer` 主干（来自 `E/F` 与 `G0/G1`）**
   * 原版 `GVC-RT` 的 `de-tokenizer` 主干存储着预训练的高频生成纹理字典。
   * 在 `E/F` 和 `G0/G1` 中，一旦同时解冻 `Bridge` 和整个 `de-tokenizer` 主干，模型在短短 $50 \sim 100$ 步内就会破坏预训练纹理流形，导致对纹理结构最敏感的 `DISTS` 恶化 $+1.62\% \sim +8.69\%$。
   * **结论**：现阶段必须 **100% 冻结 `de-tokenizer` 原始主干权重**，只在 `Bridge`、`TCG` 和 `LoRA` 上做时序与相位适配，这是守住单帧画质不退化的安全底线。

2. **铁律二：绝不能移除 `LoRA`，且必须以 `B300 (s=0.5)` 为统一起点（来自 `REVIEW.md`）**
   * `REVIEW.md` 实测证明，仅移除 `LoRA` 一步未训（`B300_noLoRA_step0@0`），96 点开发集上的 `Sobel` 时序误差立刻恶化 $+5.338\%$，`PSNR` 暴跌 $-0.2204\text{ dB}$。
   * **结论**：`B300` 中的 `Bridge.mlp[3]`、`TCG` 与 `LoRA (s=0.5)` 已经形成紧密协同的整体，下一步实验必须**完整继承 `B300` 的全部权重并保留 `LoRA (s=0.5)`**。

3. **铁律三：彻底弃用 `v6_13` 连续 Teacher 潜变量（来自 `REVIEW.md` 第 76–100 步记录）**
   * 实测数据显示：`G0` 中 Student 潜变量的运动补偿帧间变化量仅为 $0.095950$（幅度 $\text{RMS} = 0.421150$），而 `v6_13` Teacher 潜变量的帧间变化量高达 $0.554976$（是 Student 的 $5.78$ 倍，幅度 $\text{RMS} = 1.161835$，是 Student 的 $2.76$ 倍）。
   * **结论**：`v6_13` Teacher 本身是一个没有时序约束的 2D 编码器，其潜变量在帧间剧烈跳动；在 `G1` 中用 Teacher 潜变量做时序差分匹配，反而把剧烈跳变传染给了 Student（导致 Student 潜变量变化增加 $+3.8406\%$，`FloLPIPS` 恶化 $+4.100\%$）。下一步必须**彻底移除 Teacher 潜变量损失**。

4. **铁律四：必须直接打击 `REPORT_3.md` 锁定的两大真实物理闪烁源头，并堵死“降对比度”捷径**
   * **源头 A（近静止区上游潜变量漂移）**：近静止树叶（位移仅 $0.0342\text{ px/frame}$）的 `Bridge` 输出潜变量 $\hat{l}_t$ 仍存在 $0.05810$ 的逐帧变化；反事实实验证明，只要沿 GT 光流稳住 $\hat{l}_t$（`oracle_latent`），`B300` 的 `Sobel` 时序误差即可暴降 $-23.165\%$ 且边缘不软（$-0.037\%$）。
   * **源头 B（8 像素亚网格相位重绘）**：对解码特征 $\hat{f}_t$ 做整数平移（`torch.roll`），平移 2 格（对应 RGB $16$ 像素，即 `Bridge` 内 `pixel_unshuffle(2)` 的 1 个完整周期）等变误差为 $0$；但平移 1 格（对应 RGB $8$ 像素，即半个周期，导致特征在奇偶列通道间互换）时，等变误差高达 $0.0193 \sim 0.0778$。物体每移动 $8$ 像素就会触发纹理重新绘制。
   * **源头 C（振幅压低捷径）**：`B300` 虽然将 `Sobel` 时序误差降低了 $-8.856\%$，但单帧 `Sobel` 边缘强度也比 `A2000` 下降了约 $8\%$（低于 GT 均值），存在靠压低高频对比度换取时序分数的倾向。

---

## 二、 第一步收尾动作：按协议终止 `G0/G1`

1. 等待正在运行的 `G0`（71/96）与 `G1`（70/96）完成 100 步的 96 点开发集验证。
2. 由于 50 步和 100 步子集的 `DISTS` 恶化均在 $+5.4\% \sim +6.6\%$ 之间（远超每 QP $+1.0\%$ 的守门红线），严格依照预注册的 `patience=2` 规则**正式终止 `G0/G1` 训练**，不扩展至 200–300 步。
3. 归档 `REVIEW.md` 数据，将“移除 LoRA 导致时序恶化 $+5.34\%$”与“2D Teacher 潜变量帧间方差过大（$0.555$ vs. $0.096$）不适合作时序锚点”作为正式消融结论记录在案。

---

## 三、 下一步核心方案：`H0 / H1 / H2` 受控递进实验矩阵

为了严格遵循控制变量原则（能清晰分辨每一项改动的独立贡献），我们以 **`B300 (LoRA s=0.5)`** 为唯一起点，在 **100% 冻结 `codec` 与 `de-tokenizer` 原始主干** 的安全前提下，设计 `H0`、`H1`、`H2` 三组并行小规模实验（每组训练 300 步，每 100 步验证一次）。

### 1. 实验矩阵总览

| 实验组 | 起步权重 | 可训练模块 (`codec` 与 `de-tokenizer` 主干全冻结) | 损失函数配置（在 `B300` 原损失基础上增量叠加） | 核心验证目的（严格控制变量） |
| :--- | :--- | :--- | :--- | :--- |
| **`H0`** | `B300` (`s=0.5`) | 完整 `Bridge` + `TCG` + `LoRA` | `B300` 原损失 + **单帧边缘能量保底项 $\mathcal{L}_{\text{edge\_floor}}$** | 验证放开完整 `Bridge`（此前 `B300` 仅放开 `mlp[3]`）并加入边缘保底项后，能否在不伤 `DISTS` 的前提下追回 `B300` 损失的 $8\%$ 边缘锐度。 |
| **`H1`** | `B300` (`s=0.5`) | 完整 `Bridge` + `TCG` + `LoRA` | `H0` 全部损失 + **自适应门控潜变量时序平滑损失 $\mathcal{L}_{\text{lat\_self}}$** | **专治源头 A**：直接约束 `Bridge` 输出的连续潜变量 $\hat{l}_t$ 在近静止区的无依据逐帧漂移，复刻 `oracle_latent`（$-23.2\%$ 时序误差）的净收益。 |
| **`H2`** | `B300` (`s=0.5`) | 完整 `Bridge` + `TCG` + `LoRA` | `H1` 全部损失 + **8 像素亚网格整数平移等变损失 $\mathcal{L}_{\text{equiv}}$** | **专治源头 B**：强制 `Bridge` 的 `pixel_unshuffle(2)` 奇偶列通道输出一致的特征表示，消除缓慢移动树叶每跨过 $8$ 像素网格时的周期性形状重绘。 |

*(注：若当时空闲 GPU 仅支持跑两组，则直接并行跑 **`H1`** 与 **`H2`**；若支持三组，则 **`H0 / H1 / H2`** 同跑，归因链条最完美。)*

---

## 四、 模型加载链、冻结边界与优化器精确配置

### 1. 权重加载链（与 `REPORT_3.md` 中 `B300` 恢复链严格一致）

1. 加载原版基础检查点 `checkpoints/GVC-RT_P.pt`。
2. 挂载 `B+1500` 的 `TCG` 模块权重（137,760 参数）与 `A2000` 的 `LoRA` 权重（59,392 参数，固定缩放系数 $s = 0.5$）。
3. 覆盖加载 `B300` 的 52 个已训练状态键（包含 `Bridge.mlp[3]`、`TCG` 与 `LoRA` 在 `B300` 阶段的更新结果）。
4. **Step 0 自检要求**：在未添加随机平移分支时，`H0 / H1 / H2` 在 Step 0 的前向推理输出必须与 `B300` 逐像素完全一致（最大绝对误差为 $0$）。

### 2. 为什么这次要放开“完整 `Bridge`”而不是只放开 `Bridge.mlp[3]`？

* 在 `GVC-RT` 中，`Bridge` 的结构依次为：`pixel_unshuffle(2)` $\to$ `input_proj` $\to$ `mlp[0..3]` $\to$ `output_proj`。
* `REPORT_3.md` 发现的 **8 像素奇偶列相位敏感性**，直接产生于 `Bridge` 最前端的 `pixel_unshuffle(2)` 与紧随其后的 `input_proj` / `mlp[0..2]`（因为 `pixel_unshuffle(2)` 把偶数列和奇数列拆分到了不同的输入通道上）。
* 如果像 `B300` 那样只放开最末端的 `mlp[3]`（7,524 参数），前面的层已经把奇偶列特征映射成了截然不同的响应，末端一层根本扭转不过来。
* 反之，**放开完整 `Bridge` + 保持 `de-tokenizer` 主干 100% 冻结** 是绝佳的黄金组合：既赋予了 `Bridge` 消除潜变量漂移和对齐奇偶列通道的充足自由度，又因为下游 `de-tokenizer` 的生成卷积核完全锁死，模型绝对无法把生成纹理抹糊！

### 3. 参数冻结与分组学习率代码规范

```python
def configure_h_series_trainable_params(model):
    # 1. 默认冻结全部参数（确保 codec 与 de-tokenizer 原始主干 100% 锁死）
    for name, param in model.named_parameters():
        param.requires_grad = False

    bridge_params = []
    tcg_params = []
    lora_params = []

    # 2. 精确解冻完整 Bridge、TCG 和 LoRA
    for name, param in model.named_parameters():
        if "bridge" in name.lower():
            param.requires_grad = True
            bridge_params.append(param)
        elif "tcg" in name.lower():
            param.requires_grad = True
            tcg_params.append(param)
        elif "lora" in name.lower():
            param.requires_grad = True
            lora_params.append(param)

    # 3. 配置分组学习率：Bridge 与 TCG 使用 1e-5 稳步适配，LoRA 保持 5e-5
    optimizer = torch.optim.AdamW(
        [
            {"params": bridge_params, "lr": 1e-5, "name": "bridge"},
            {"params": tcg_params, "lr": 1e-5, "name": "tcg"},
            {"params": lora_params, "lr": 5e-5, "name": "lora"},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    return optimizer
```

---

## 五、 新增三项核心损失函数的数学公式与 PyTorch 完整实现

在 `H0 / H1 / H2` 中，我们保留 `B300` 原有的基础损失项 $\mathcal{L}_{\text{B300}}$（包含 `1.0 LPIPS-Alex`、`0.5 DISTS`、`0.1383589 RGB L1`、`0.05 RAFT_motion` 以及原有的 `GT 运动补偿 RGB/Sobel 时序残差项`），并按实验组叠加以下三项专门设计的物理约束。

### 1. 单帧边缘能量保底损失 $\mathcal{L}_{\text{edge\_floor}}$（用于 `H0 / H1 / H2`）

#### （1）设计原理（为什么不会重蹈 `F` 组的覆辙？）
* 在 `F` 组中，计算的是**逐像素对齐的 Sobel 差** $\Vert \nabla \hat{I}_t - \nabla I_t^{\text{GT}} \Vert_1$。当生成树叶与 GT 存在 $1 \sim 2$ 像素相位偏移时，逐像素相减会产生“双倍惩罚”，反而逼着模型把树叶抹平。
* 在 `H` 系列中，我们不要求生成边缘与 GT 边缘在逐个像素点上死板重合，而是将图像划分为 $16 \times 16$ 的局部块（Patch），只要求**每个局部块内生成图像的平均 Sobel 梯度强度不得低于 GT 局部块的平均梯度强度**！
* 只要模型试图通过“压低 $8\%$ 边缘对比度”来偷拿时序分，$\operatorname{ReLU}$ 门控就会立刻激活并给以梯度惩罚；而只要局部锐度达标，该项损失即为 $0$，绝不干扰正常纹理生成。

#### （2）数学表达式
设 $S(\cdot)$ 为提取水平与垂直 Sobel 梯度幅值的算子，$\operatorname{Pool}_{16}(\cdot)$ 为 $16 \times 16$ 局部平均池化算子，则：

$$
\mathcal{L}_{\text{edge\_floor}} = \operatorname{mean}\left( \operatorname{ReLU}\Big( \operatorname{Pool}_{16}\big(S(I_t^{\text{GT}})\big) - \operatorname{Pool}_{16}\big(S(\hat{I}_t)\big) \Big) \right)
$$

#### （3）PyTorch 实现代码
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sobel_mag(img):
    """
    计算 RGB 图像的 Sobel 边缘梯度幅值
    img: [B, 3, H, W], 范围 [0, 1]
    返回: [B, 3, H, W]
    """
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    kx = kx.repeat(3, 1, 1, 1)
    ky = ky.repeat(3, 1, 1, 1)

    gx = F.conv2d(img, kx, padding=1, groups=3)
    gy = F.conv2d(img, ky, padding=1, groups=3)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def loss_edge_energy_floor(recon_rgb, gt_rgb, patch_size=16):
    """
    单帧局部边缘能量保底损失：防止模型靠压低 8% 边缘强度来刷低时序误差，同时避免逐像素相位惩罚
    """
    sobel_recon = compute_sobel_mag(recon_rgb)
    with torch.no_grad():
        sobel_gt = compute_sobel_mag(gt_rgb)
        pool_gt = F.avg_pool2d(sobel_gt, kernel_size=patch_size, stride=patch_size)

    pool_recon = F.avg_pool2d(sobel_recon, kernel_size=patch_size, stride=patch_size)
    # 仅当重建局部锐度低于 GT 局部锐度时产生惩罚
    deficit = F.relu(pool_gt - pool_recon)
    return deficit.mean()
```

---

### 2. 自适应门控潜变量时序平滑损失 $\mathcal{L}_{\text{lat\_self}}$（用于 `H1 / H2`）

#### （1）设计原理（如何精准复刻 `oracle_latent` 在树叶上的 $-23.2\%$ 收益，同时避开灌木区的退化？）
* `REPORT_3.md` 显示，在近静止树叶上保持 $\hat{l}_t$ 与上一帧对齐可减少 $23.165\%$ 的闪烁，但在快速运动灌木上强行复用旧 $\hat{l}_{t-1}$ 会导致误差上升 $+16.037\%$。
* 因此，我们直接利用 **GT 视频自身的运动补偿帧间残差** $r_t^{\text{GT}} = \Vert I_t^{\text{GT}} - \operatorname{Warp}(I_{t-1}^{\text{GT}}, F_{t \to t-1}^{\text{GT}}) \Vert_1$ 下采样到潜空间分辨率（$\frac{1}{16}$），构造一个连续自适应权重掩码 $W_t \in [0, 1]$：
  * 当某区域是**近静止树叶或刚性平移背景**时，GT 光流对齐残差 $r_t^{\text{GT}} \approx 0 \Rightarrow W_t \approx 1$，此时强力约束当前帧潜变量 $\hat{l}_t$ 贴合对齐后的上一帧潜变量 $\tilde{l}_{t-1}$。
  * 当某区域发生**遮挡、剧烈形变或光流估计不准**（如快速移动灌木的边缘）时，$r_t^{\text{GT}}$ 较大 $\Rightarrow W_t \to 0$，自动关闭潜变量时序约束，完全放行当前帧新特征！
* 同时，为了防止模型通过“把潜变量 $\hat{l}_t$ 的整体数值幅度缩小”来减小帧间差，我们在计算差值前对 $\hat{l}_t$ 做**逐通道 RMS 归一化**（与 `REPORT_3.md` 的测量口径完全一致）。

#### （2）数学表达式
设 $\hat{l}_t, \hat{l}_{t-1} \in \mathbb{R}^{B \times C \times H_l \times W_l}$ 为 `Bridge` 输出的连续潜变量（其中 $H_l = \frac{H}{16}, W_l = \frac{W}{16}$），$F_{t \to t-1}^{\downarrow 16}$ 为下采样并按比例缩放 $\frac{1}{16}$ 后的 GT 光流：

$$
\bar{l}_t = \frac{\hat{l}_t}{\operatorname{RMS}(\hat{l}_t.\text{detach}()) + \epsilon}, \quad \tilde{l}_{t-1} = \operatorname{Warp}\left( \frac{\hat{l}_{t-1}}{\operatorname{RMS}(\hat{l}_{t-1}.\text{detach}()) + \epsilon}, \, F_{t \to t-1}^{\downarrow 16} \right)
$$

$$
W_t = O_t^{\downarrow 16} \odot \exp\left( -\frac{\operatorname{Pool}_{16}\big(\Vert I_t^{\text{GT}} - \operatorname{Warp}(I_{t-1}^{\text{GT}}, F_{t \to t-1}^{\text{GT}}) \Vert_1\big)}{\tau} \right), \quad (\tau = 0.02)
$$

$$
\mathcal{L}_{\text{lat\_self}} = \frac{\sum \left\Vert W_t \odot (\bar{l}_t - \tilde{l}_{t-1}) \right\Vert_1}{\sum W_t + \epsilon}
$$

#### （3）PyTorch 实现代码
```python
def warp_tensor(x, flow):
    """
    使用光流对特征或图像进行双线性采样对齐
    x: [B, C, H, W], flow: [B, 2, H, W] (单位：像素位移 dx, dy)
    """
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid_x = xx.unsqueeze(0) + flow[:, 0, :, :]
    grid_y = yy.unsqueeze(0) + flow[:, 1, :, :]

    norm_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    norm_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)

    warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # 生成越界有效性掩码 [B, 1, H, W]
    valid_mask = ((norm_x >= -1.0) & (norm_x <= 1.0) & (norm_y >= -1.0) & (norm_y <= 1.0)).to(x.dtype).unsqueeze(1)
    return warped, valid_mask

def loss_latent_self_temporal(l_curr, l_prev, gt_curr, gt_prev, gt_flow_rgb, tau=0.02):
    """
    自适应门控的 Bridge 连续潜变量时序平滑损失
    l_curr, l_prev: [B, C_l, H/16, W/16] (注意：在 4 帧 BPTT 段内，l_prev 保留梯度，不 detach！)
    gt_curr, gt_prev: [B, 3, H, W]
    gt_flow_rgb: [B, 2, H, W] (从 t 到 t-1 的 GT 光流)
    """
    B, C_l, H_l, W_l = l_curr.shape
    _, _, H, W = gt_curr.shape

    with torch.no_grad():
        # 1. 计算 RGB 分辨率下的 GT 光流对齐残差，衡量该位置是否真正静止/可对齐
        gt_prev_warped, valid_rgb = warp_tensor(gt_prev, gt_flow_rgb)
        gt_diff_l1 = (gt_curr - gt_prev_warped).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 下采样到 1/16 潜变量分辨率
        gt_diff_low = F.adaptive_avg_pool2d(gt_diff_l1, (H_l, W_l))
        valid_low = F.adaptive_avg_pool2d(valid_rgb, (H_l, W_l))

        # 2. 将 GT 光流下采样并按比例缩放 1/16
        flow_low = F.interpolate(gt_flow_rgb, size=(H_l, W_l), mode="bilinear", align_corners=True)
        flow_low[:, 0] *= (float(W_l) / float(W))
        flow_low[:, 1] *= (float(H_l) / float(H))

        # 3. 构造近静止/高置信度自适应门控权重 W_t
        weight_mask = (valid_low > 0.99).to(l_curr.dtype) * torch.exp(-gt_diff_low / tau)

        # 4. 计算逐通道 RMS 归一化分母（加 detach 防止模型靠缩小潜变量幅度来刷低损失）
        rms_curr = torch.sqrt(torch.mean(l_curr.detach() ** 2, dim=(2, 3), keepdim=True) + 1e-8)
        rms_prev = torch.sqrt(torch.mean(l_prev.detach() ** 2, dim=(2, 3), keepdim=True) + 1e-8)

    # 5. 对归一化后的潜变量做光流对齐并计算加权 L1 差值
    l_curr_norm = l_curr / rms_curr
    l_prev_norm = l_prev / rms_prev
    l_prev_warped, valid_lat = warp_tensor(l_prev_norm, flow_low)

    final_mask = weight_mask * valid_lat
    diff = (l_curr_norm - l_prev_warped).abs().mean(dim=1, keepdim=True)

    loss = (diff * final_mask).sum() / (final_mask.sum() + 1e-6)
    return loss
```

---

### 3. 8 像素亚网格整数平移等变损失 $\mathcal{L}_{\text{equiv}}$（仅用于 `H2`）

#### （1）设计原理（零插值消除 `pixel_unshuffle(2)` 奇偶相位偏差）
* `REPORT_3.md` 的整数 `torch.roll` 控制实验证明，当输入特征 $\hat{f}_t$ 平移 $1$ 个像素（对应 RGB $8$ 像素）时，原本位于偶数列的特征进入了奇数列，经过 `Bridge + de-tokenizer` 后的图像与原图平移 $8$ 像素的结果相差 $0.02 \sim 0.08$。
* 我们在训练时对当前帧的 $\hat{f}_t$ 随机抽取一个 $1$ 像素位移方向 $\delta = (\delta_y, \delta_x) \in \{(0, 1), (1, 0), (1, 1)\}$，执行 `torch.roll`。
* **关键工程细节（防止污染 `TCG` 历史缓存）**：在计算平移分支的辅助前向传播时，我们**对传入 `TCG` 的历史缓存同步执行相同的 `torch.roll`（且仅读取缓存、不覆盖写回主路径的历史缓存状态）**，这样主训练路径的 4 帧跨帧梯度链不会受到任何干扰，同时 `Bridge`、`TCG` 和 `LoRA` 都能直接学习到 $8$ 像素半格平移等变性！

#### （2）数学表达式
设 $\delta \in \{(0, 1), (1, 0), (1, 1)\}$ 为特征域的 $1$ 像素整数位移（对应潜空间缓存的位移为 $0.5\delta$ 或在进入生成器前对特征与历史做同步平移，对应输出 RGB 域的整数位移为 $8\delta$），$M_{\text{inner}}$ 为裁除四周 $16$ 像素边界后的内部有效区域掩码：

$$
\hat{I}_t^{\text{shift}} = \operatorname{Decoder}\Big( \operatorname{Bridge}\big( \operatorname{Roll}(\hat{f}_t, \delta) \big) \Big)
$$

$$
\mathcal{L}_{\text{equiv}} = \left\Vert M_{\text{inner}} \odot \Big( \hat{I}_t^{\text{shift}} - \operatorname{Roll}\big(\hat{I}_t.\text{detach}(), \, 8\delta\big) \Big) \right\Vert_1
$$

#### （3）PyTorch 实现代码
```python
import random

def loss_subgrid_shift_equivariance(model, f_hat_t, qp_t, recon_rgb_t, border_crop=16):
    """
    8 像素亚网格平移等变损失：消除 Bridge pixel_unshuffle(2) 奇偶列通道不一致引发的周期性重绘
    f_hat_t: [B, C_f, H/8, W/8]
    recon_rgb_t: [B, 3, H, W] (主前向路径已算出的当前帧重建图像)
    """
    # 1. 随机选择一个 1 特征像素平移方向（水平 1 格、垂直 1 格、或对角 1 格，对应 RGB 8 像素）
    dy, dx = random.choice([(0, 1), (1, 0), (1, 1)])

    # 2. 对输入解码特征执行零开销整数平移
    f_hat_shifted = torch.roll(f_hat_t, shifts=(dy, dx), dims=(-2, -1))

    # 3. 备份当前 TCG 历史缓存，避免辅助前向传播污染主路径的跨帧状态
    cached_state_backup = model.backup_tcg_cache()
    try:
        # 若 TCG 缓存分辨率与 Bridge 输出一致（1/16），为纯粹训练 Bridge+LoRA 的奇偶通道等变性，
        # 可在辅助分支中暂时固定/关闭历史或使用备份状态推理：
        recon_shifted, _ = model.forward_generator_only(
            f_hat_shifted, qp_t, update_cache=False
        )
    finally:
        # 恢复主路径的 TCG 历史缓存
        model.restore_tcg_cache(cached_state_backup)

    # 4. 将正常重建图像向相同方向整数平移 8 个 RGB 像素作为对齐目标
    target_shifted = torch.roll(recon_rgb_t.detach(), shifts=(8 * dy, 8 * dx), dims=(-2, -1))

    # 5. 裁除四周 16 像素边界（排除 circular wrap-around 边界伪影），计算内区 RGB L1 误差
    diff = (recon_shifted - target_shifted).abs()
    inner_diff = diff[:, :, border_crop:-border_crop, border_crop:-border_crop]
    return inner_diff.mean()
```
*(注：如果在 `forward_generator_only` 中将 `TCG` 设为使用自身当前帧特征或同步平移状态，还能省去缓存备份步骤；为了节省显存与耗时，$\mathcal{L}_{\text{equiv}}$ 在每个 4 帧微段中**只需对最后 1 帧（或随机抽 1 帧）计算一次**即可，无需帧帧重复计算！)*

---

## 六、 训练前必做步骤：10 步分项梯度范数校准（遵循 `REPORT_2.md` 规范）

`REPORT_2.md` 明确要求：“**时序项重新测量各参数组梯度后设定，不按损失数字大小猜测。**”
在正式启动 `H0 / H1 / H2` 的 300 步训练前，先运行一个仅包含 **10 个训练 Batch、不执行 `optimizer.step()`** 的梯度测量脚本，记录各损失分项对 `Bridge` 和 `LoRA` 参数产生的真实梯度 $L_2$ 范数：

1. **测量目标**：
   * 设基础感知与重建损失 $\mathcal{L}_{\text{spatial}} = \mathcal{L}_{\text{LPIPS}} + 0.5\mathcal{L}_{\text{DISTS}} + 0.1383589\mathcal{L}_{L_1}$ 在 `Bridge` 参数上的梯度范数中位数为 $G_{\text{spatial}}^{\text{Bridge}}$。
   * 分别单独对 $\mathcal{L}_{\text{edge\_floor}}$、$\mathcal{L}_{\text{lat\_self}}$、$\mathcal{L}_{\text{equiv}}$ 执行 `.backward(retain_graph=True)`，测量它们在 `Bridge` 参数上的单位权重梯度范数 $G_{\text{floor}}^{\text{Bridge}}$、$G_{\text{lat}}^{\text{Bridge}}$、$G_{\text{equiv}}^{\text{Bridge}}$。
2. **权重校准公式（定标准则）**：
   * **边缘保底项权重 $\lambda_{\text{floor}}$**：校准使其在激活时的梯度范数约为空间主损失的 $15\% \sim 20\%$（推荐初始候选值 $\lambda_{\text{floor}} = 0.2$）。
   * **潜变量时序项权重 $\lambda_{\text{lat}}$**：校准使得 $\lambda_{\text{lat}} \cdot G_{\text{lat}}^{\text{Bridge}} \approx 0.25 \cdot G_{\text{spatial}}^{\text{Bridge}}$（即潜变量时序梯度占空间主梯度的 $25\%$ 左右，推荐初始候选值 $\lambda_{\text{lat}} = 0.15 \sim 0.25$）。
   * **亚网格平移等变项权重 $\lambda_{\text{equiv}}$**：校准使得 $\lambda_{\text{equiv}} \cdot G_{\text{equiv}}^{\text{Bridge}} \approx 0.20 \cdot G_{\text{spatial}}^{\text{Bridge}}$（推荐初始候选值 $\lambda_{\text{equiv}} = 0.10 \sim 0.20$）。
   * *为什么遵循 $20\% \sim 25\%$ 梯度比例法则？* 当新增正则项的梯度范数控制在空间感知主损失的 $20\% \sim 25\%$ 时，既能持续推动 `Bridge` 收敛到时序稳定且平移等变的解空间，又绝不会压倒单帧 `LPIPS + DISTS` 的主导地位。

---

## 七、 15 帧监督期内的 4 帧分段反传（Truncated BPTT）完整训练循环

在 `ULong` 训练集上，每个样本序列包含 64 帧（`I0` + `P1..P48` 共 49 帧无梯度预热建立历史缓存 + `P49..P63` 共 15 帧监督训练）。以下为单个训练 Step 的标准执行流程：

```python
# 1. 前 49 帧（Frame 0 ~ 48）无梯度预热，建立真实的 TCG 历史缓存与上一帧潜变量 l_prev
model.reset_tcg_cache()
with torch.no_grad():
    for t in range(49):
        recon_t, l_hat_t = model.forward_step(f_hat_seq[:, t], qp_seq[:, t], detach_cache=True)
        if t == 48:
            l_prev = l_hat_t.detach()
            recon_prev = recon_t.detach()

# 2. 后 15 帧（Frame 49 ~ 63）划分为 4 个微段：[49..52], [53..56], [57..60], [61..63]
optimizer.zero_grad()
segments = [range(49, 53), range(53, 57), range(57, 61), range(61, 64)]

for seg in segments:
    seg_loss = 0.0
    for idx_in_seg, t in enumerate(seg):
        # 段内保持 TCG 缓存计算图不断开 (detach_cache=False)
        recon_t, l_hat_t = model.forward_step(f_hat_seq[:, t], qp_seq[:, t], detach_cache=False)

        # (a) 计算 B300 原有基础损失（单帧 LPIPS/DISTS/L1 + RAFT 0.05 + RGB/Sobel 运动补偿残差）
        loss_t = compute_b300_base_loss(recon_t, recon_prev, gt_seq[:, t], gt_seq[:, t-1], gt_flow_seq[:, t])

        # (b) [H0 / H1 / H2 共有] 加入单帧局部边缘能量保底损失
        loss_t = loss_t + lambda_floor * loss_edge_energy_floor(recon_t, gt_seq[:, t])

        # (c) [H1 / H2 共有] 加入自适应门控的 Bridge 潜变量时序平滑损失
        if exp_name in ["H1", "H2"]:
            loss_t = loss_t + lambda_lat * loss_latent_self_temporal(
                l_hat_t, l_prev, gt_seq[:, t], gt_seq[:, t-1], gt_flow_seq[:, t]
            )

        # (d) [仅 H2] 在每个 4 帧微段的最后 1 帧计算 8 像素亚网格平移等变损失（省显存、高效率）
        if exp_name == "H2" and (idx_in_seg == len(seg) - 1):
            loss_t = loss_t + lambda_equiv * loss_subgrid_shift_equivariance(
                model, f_hat_seq[:, t], qp_seq[:, t], recon_t
            )

        seg_loss = seg_loss + (loss_t / 15.0)  # 按总监督帧数 15 归一化

        # 段内传递状态（保留计算图梯度链 t-1 -> t）
        l_prev = l_hat_t
        recon_prev = recon_t

    # 3. 微段结束：反向传播累积梯度，随后截断传往下一微段的缓存与状态梯度（数值保留！）
    seg_loss.backward()
    model.detach_tcg_cache_in_place()
    l_prev = l_prev.detach()
    recon_prev = recon_prev.detach()

# 4. 攒够全部 15 帧梯度后，执行全局梯度裁剪并更新一次参数
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 八、 验证协议、早停红线与四维验收标准（100 / 200 / 300 步）

每训练 100 步，在固定的开发集（36 点快速子集与 96 点完整开发集）以及 `REPORT_3.md` 的两个原尺寸无损 ROI（近静止树叶 `24747b35…` 与移动灌木/岩石 `28a7c428…`）上执行全面核验：

### 1. 四维硬性验收指标（相对于 `B300 s=0.5` 基准）

1. **第一维：空间感知画质守门红线（绝不允许靠变糊换稳定）**
   * 同时计算并报告 `LPIPS-Alex`（保持历史可比性）、`LPIPS-VGG`（对齐原论文口径）与 `DISTS`。
   * **合格标准**：每个 QP 下的 `DISTS` 和 `LPIPS` 相对 `B300` **恶化不得超过 $+0.5\%$**（理想情况应持平或微降）；若任一模型在连续两个检查点（如 100 步和 200 步）出现 `DISTS` 恶化超过 $+1.0\%$，立即触发早停。
2. **第二维：单帧边缘锐度恢复核验（检验 `H0` 中 $\mathcal{L}_{\text{edge\_floor}}$ 的作用）**
   * 检查 `edge_strength_by_video_qp.csv` 中的单帧 `Sobel` 边缘强度。
   * **合格标准**：相比 `B300`（比 `A2000` 低 $8\%$），`H0 / H1 / H2` 的单帧边缘强度应明显回升（向 `GT` 靠拢），彻底洗脱“靠变软降时序误差”的嫌疑。
3. **第三维：机理级防闪烁指标核验（检验 `H1` 与 `H2` 是否精准命中两大源头）**
   * **源头 A 核验（看 `H1` vs. `H0`）**：复测 `REPORT_3.md` 表 1 中的“近静止树叶 `Bridge latent` 运动补偿归一化 $L_1$”（`B300` 原为 $0.05810$），预期在 `H1` 中下降 $25\% \sim 50\%$，且开发集整体 `GT-flow Sobel/RGB` 时序残差以及 `FloLPIPS` 同步下降。
   * **源头 B 核验（看 `H2` vs. `H1`）**：复测 `REPORT_3.md` 表 3 中的“无插值特征 $1$ 格（RGB $8$ 像素）整数平移等变误差”（`B300` 原为 $0.0193 \sim 0.0418$），预期在 `H2` 中大幅下降 $50\%$ 以上！
4. **第四维：原尺寸 1:1 无损 ROI 视频盲测核验**
   * 导出与 `REPORT_2.md / REPORT_3.md` 完全同规格的 `FFV1` 30fps 原像素无损对比视频（包含 `GT`、`Original`、`B300`、`H0`、`H1`、`H2`），重点观察近静止树叶与微动叶片边缘是否不再出现每帧形状改写。

---

## 九、 第二阶段预案：当 `H2` 稳住闪烁后，如何进一步拉高单帧画质？

一旦通过 `H1 / H2` 在冻结 `de-tokenizer` 的前提下把“潜变量漂移”和“8 像素相位闪烁”彻底解决（拿到扎实的论文核心创新点一：**潜空间自锚定与亚网格平移等变解码**），如果还需要进一步拉高单帧感知画质（让静态截图更接近 `GT`，形成论文核心创新点二），可在 `H2` 最佳权重之上按以下两条低风险路径二选一推进：

### 路径 A（零 GAN 风险，强烈推荐）：在 `Bridge` 中注入零初始化码流退化先验（`QP` + 帧间解码残差调制）
* **为什么能提画质**：目前 `Bridge` 对所有码率（`QP 0` 到 `QP 9`）和所有空间区域使用同一套静态映射。通过将当前帧的量化档位 `QP`（经零初始化 `nn.Embedding(10, 2*C)` 做通道 FiLM 调制）以及前端解码特征的运动补偿残差图 $\Vert \hat{f}_t - \operatorname{Warp}(\hat{f}_{t-1}) \Vert_1$（经零初始化轻量卷积做空间 SFT 调制）注入 `Bridge`，`Bridge` 就能明确知道哪些区域落入了量化死区需要强力特征恢复、哪些区域保留了高比特真实结构，从而在不解冻 `de-tokenizer` 的情况下进一步降低 `LPIPS` 与 `DISTS`。

### 路径 B（若需微调 `de-tokenizer` 末端）：仅解冻 `stage4 + head` 并配套预热的 PatchGAN
* 如果后续一定要放开 `de-tokenizer` 权重来改写高频树叶画法，绝不能再像 `E/F/G0/G1` 那样无判别器全量裸训；必须只放开最末端的高分辨率层 `stage4 + head`，且严格按照 `REPORT_2.md` 的建议，先单独预热并验证 **2D 单帧 PatchGAN + 3 帧条件时序 PatchGAN** 能够准确区分真假纹理与正常/打乱时间序后，再以小权重接入训练。
