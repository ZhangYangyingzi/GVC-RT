# V1.2 固定协议与计算路径

## 基线与数据

- 本轮固定基线只取 V1 **stage_D/checkpoint.pt 的 enhancement.decoder**，输入全零 V1 u 和原版 ell：`ell_c = ell_base + D_V1_D(0, ell_base)`。它对应 Val6 完整 I/P 的 PSNR 26.450357、MS-SSIM 0.810394、LPIPS 0.305983。
- C 对照单独取 V1.1 六帧诊断所用的 `stage_c/checkpoint.pt` 中 `base_only`。它不参与定义 ell_c。V1 最终阶段 D 的另训 base-only 可作为额外历史参照，不能与 C 或固定 ZERO 混称。
- 原版 I 帧原样保留，P 的新增修正仅影响显示。原版参考 feature/RGB、QP offset 和 reset=96 完全由原基础流确定。
- 训练、归一化拟合、直接量化概率校准只使用 V1.1 固定六帧：两段各 1/8/15。
- 初期 Train6 诊断仍封装完整 1I+15P：每段仅三帧有教师修正，其他 P 精确留在固定 ZERO；明确标为 sparse。报告 TARGET_P（六帧）、P（30帧）和 IP（32帧），不能将六帧增量当作全序列增量。
- 六帧量化实际信息门槛通过后，才在编码端对 Train2 其余24个P帧生成临时目标以评估完整增强。这些帧不加入训练或校准。完整 Train2 实际信息门槛通过后才允许 Val6。
- Val6 保留六段、各1I+15P和真实帧率。正常接收端只读取基础/增强流、共享模型；禁止源 RGB、教师/oracle/基础训练缓存访问。

## 教师恢复与选择

只考虑 V1.1 的 MSE+0.001×LPIPS 候选（O/Z 两起点的0/25/50/100/150检查点）。先测本轮阶段D ZERO，再筛选 PSNR 更高且 LPIPS 不恶化的点，取 PSNR 最高者。

已保存的 `.pt` 是相对于其原起点的 delta，而不是最终 latent。必须先按其 `initialization` 恢复 `ell_start`，验证 `ell_start_sha256`，再做：

```
ell_oracle = ell_start + saved_delta
r_star = ell_oracle - ell_c_stage_D
```

六帧实际都选中旧 C-ZERO 起点的 MSE_LPIPS 150步；因此新编码端优化也固定采用同一 C-ZERO 起点、Adam lr=1e-3、150步、MSE+0.001×LPIPS。这是优化起点，不是输出基线；输出始终以阶段D ell_c 为基线。新样本按相同预定质量规则从保存的优化步选择，若无合格点则标记并发送零新增修正。

历史六帧优化没有重跑。保存旧完整优化运行时间与当前恢复/核验时间；不能把缓存查找时间当作从视频产生 r_star 的成本。新评估帧的计时包含源 PNG 读取、初始化、150步优化及检查点度量，分别列出分析、熵编码与接收端开销。

## 固定残差 codec

实际张量：`r_star, ell_c: [1,18,68,120]`，编码器输出 `[1,8,17,30]`，每帧4080符号。

- 下采样每轴4倍，空间点数缩小16倍；与146880个完整残差标量相比，码值数缩小36倍。
- E 为36通道（归一化残差18 + ell_c18）输入的轻量CNN，hidden64、两次stride2、8通道输出；D 复用V1可微 Synthesis 结构，hidden64、两个深度可分离残差块。
- 每通道残差 RMS 只由六帧 r_star 拟合；不减均值；作为共享 buffer 冻结。
- `pred_norm = D(u_hat,ell_c) - D(zeros_like(u_hat),ell_c)`。
- `r_hat = pred_norm * residual_scale`，反归一化后再加到 ell_c。两次D是确定性同模型调用；u_hat=0 时 pred_norm/r_hat 严格为0。
- G冻结但不在增强损失前向外包 no_grad。优化器仅包含新E/D及阶段B的独立熵模型。
- 阶段A：1000步，MSE+0.001LPIPS+0.01 normalized-residual-MSE。
- 阶段B：1500步，四档按独立轮换覆盖；前300步lambda_R=0，然后400步0.001、400步0.01、400步0.05。第300步真实码流适配检查不通过就停止后续rate扩展。
- checkpoint每250步及300步适配点保存，只取固定最终步，不用Val6选择。

## 实际 ORC2 码流

ORC2 与基础熵 coder 独立，复用V1已验证的32bit算术编码与16bit CDF转换。符号先检验有限、整数、int8范围；超范围抛错，绝不裁剪。

序列头 `struct '<4sBBHHHHHHf32s32s'` 共86字节：

| 字段 | 字节 |
|---|---:|
| magic ORC2 | 4 |
| 方法（direct/learned）、量化档号 | 2 |
| 有效H/W、帧数、C、网格H/W | 12 |
| delta float32 | 4 |
| 原始基础流SHA256 | 32 |
| 固定共享模型文件SHA256 | 32 |

每帧一个标志：

- 2：I帧，不附加载荷；保持原版I。
- 0：P全零网格，无熵载荷、无额外length/CRC；接收端由序列shape恢复零整数，输出精确为固定ZERO。
- 1：非零P，另有4字节payload长度、4字节CRC32和真正的算术payload。

因此整段无新增修正时仅86+16=102字节（816bit）；不是V1中逐个昂贵地编码零符号。量化参数/shape每序列写一次，无CDF表、均值、逐帧源scale或oracle位置参数。共享模型文件包含固定校准/归一化数据和baseline checkpoint指纹。所有文件头、标志、CRC、算术终止与对齐都计入实际文件字节。

direct 在物理 r_star 单位量化，固定六点几何网格由Train6 RMS决定。learned 的四档由阶段A的Train6 u RMS确定，在阶段B开始前保存；没有在Val6调节。

正常接收进程重解真实基础文件，得到原版 ell/q，再独立计算固定 ell_c，解增强整数、反量化、D差分、反归一化、G。不读取发送端 r_star、ell_oracle、源图或缓存。零输入还实际核对D差分为零及G输出相等。

WRONG_SOURCE 对各自合法文件先独立解码，再按固定循环后继视频、相同帧号交换整数/表示；当前 ell_c/q/参考不交换。它与连续教师/codec、ZERO_INPUT都仅作诊断，不赋予正式RD点。

## 评价与预算

有效RGB区域1080×1920、范围[0,1]，沿用V1的PSNR、五尺度MS-SSIM、LPIPS Alex normalize=True。基础编码精度FP16，增强G精度FP32；另保存原版各QP的匹配FP32 G参考，同码率推断只用这些匹配精度点和真实重叠区间。

阶段门槛在config中预定：连续TARGET_P平均+0.2dB、LPIPS最多增加0.001，且REAL比WRONG至少+0.01dB、LPIPS最多增加0.001。量化门槛平均+0.1dB并满足同样匹配条件。所有逐帧/逐视频值仍保留，不以非零率代替信息效益。

预算按**每段**实际增强文件bit / 基础文件bit，列10/25/50%的可达到点及偏差。全零点虽便宜但没有新增信息收益。直接量化高码率不证明残差不可压缩。源信息利用、实际码率效率、泛化分别判断；不把固定ZERO相对原版的既有收益记到新增分支上。
