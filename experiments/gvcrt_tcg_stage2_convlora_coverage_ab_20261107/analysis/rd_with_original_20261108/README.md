# 含原始 GVC-RT 的独立数据集 R-D 对比

日期：2026-11-08。仅复用已核验实测数据，未重新编码、解码或训练。

原版数据来自 2026-11-05 独立评测 metrics_original.csv，加载项目 checkpoints/GVC-RT_I.pt 与 GVC-RT_P.pt；原版源码固定于 b2a0895c0cf31cf2be3ab1c2571ca07c05866c03，不包含 TCG，不加载实验适配检查点。原版权重当前 SHA 与历史协议一致。

45 个视频、每视频同一段 64 帧、QP 0/5/9，共 135 点；与本轮每个模型的源文件/RGB 哈希、源帧索引、实际帧 QP 调度、码流字节/SHA/bpp、指标网络权重逐点对齐。原版历史评测为独立 fresh RANS 编解码；本次仅将其原始实测结果加入图表，不以 parent0 代替。

## 四条线

- 黑色虚线：原始 GVC-RT 独立评测。
- 蓝色：TCG B+1500 matched-FP32，无 LoRA。
- 橙色：TCG + LoRA A，stage3/head，开发集最佳 2000 步。
- 绿色：TCG + LoRA B，A + upsample.conv1，开发集最佳 2000 步。

本轮基础模型含 TCG，不能称为原版；A/B 表示本轮 LoRA 覆盖实验，不是早期数据扩充的 A/B。

## 相对原版的变化

按每个数据集内视频等权计算各 QP 均值，再平均三个 QP 的相对变化；各数据集不混合。负的 LPIPS/DISTS/FloLPIPS 变化表示改善，不表示同质量码率节省。

|数据集|模型|FloLPIPS|LPIPS|DISTS|PSNR dB|GT-flow warp error|
|---|---|---:|---:|---:|---:|---:|
|ULong_new_holdout_32|baseline|-12.1880%|-11.9419%|-16.3866%|-0.2983|+4.5810%|
|ULong_new_holdout_32|A_control|-12.6327%|-12.3436%|-16.1844%|-0.2691|+4.2916%|
|ULong_new_holdout_32|B_expanded|-12.6852%|-12.3753%|-16.2499%|-0.2855|+4.4455%|
|ULong_historical_6|baseline|-11.8997%|-10.5651%|-14.2721%|-0.1859|+3.3198%|
|ULong_historical_6|A_control|-12.2735%|-10.7676%|-14.3157%|-0.1715|+3.1328%|
|ULong_historical_6|B_expanded|-12.3156%|-10.7149%|-14.3157%|-0.1895|+3.2977%|
|HEVC_B_5|baseline|-3.2018%|-1.8325%|-3.1959%|-0.2226|+3.9153%|
|HEVC_B_5|A_control|-4.7791%|-2.9124%|-4.2117%|-0.2100|+3.8374%|
|HEVC_B_5|B_expanded|-4.8187%|-2.8768%|-4.1480%|-0.2142|+3.8545%|
|UVG_historical_HoneyBee_Jockey_2|baseline|+3.6274%|+3.3456%|-2.1930%|-0.4899|+4.2249%|
|UVG_historical_HoneyBee_Jockey_2|A_control|+2.1091%|+1.9032%|-4.0592%|-0.4353|+3.7014%|
|UVG_historical_HoneyBee_Jockey_2|B_expanded|+2.1152%|+1.9235%|-4.0611%|-0.4430|+3.7294%|

ULong 上 LoRA 相对 TCG 的 DISTS 小幅回退，与最终模型相对原版的 DISTS 改善是两个不同参照。不能称“ULong DISTS 比原版差”。同时，UVG 两视频的 FloLPIPS/LPIPS 仍差于原版，各组 PSNR 与 GT-flow warp 指标也有退化，不能声称全面胜过原版。

## 图和数据

- ULong held-out subset (32 videos)：[R-D 曲线](ULong_new_holdout_32_with_original.png) · [相对原版差值](ULong_new_holdout_32_relative_to_original.png)
- ULong historical subset (6 videos)：[R-D 曲线](ULong_historical_6_with_original.png) · [相对原版差值](ULong_historical_6_relative_to_original.png)
- HEVC-B (5 videos)：[R-D 曲线](HEVC_B_5_with_original.png) · [相对原版差值](HEVC_B_5_relative_to_original.png)
- UVG subset (HoneyBee and Jockey only)：[R-D 曲线](UVG_historical_HoneyBee_Jockey_2_with_original.png) · [相对原版差值](UVG_historical_HoneyBee_Jockey_2_relative_to_original.png)

- [540 个实测点](metrics_with_original.csv)
- [各数据集 48 个 R-D 点](rd_points_by_dataset.csv)
- [相对原版的分数据集统计](changes_vs_original.csv)
- [405 项配对检查](point_pairing_checks.csv)
- [来源与复用审计](reuse_audit.json)

这些结果使用既有反复评估过的测试集；ULong/UVG 标签均指明具体子集，不能声称全数据集或全新独立确认集的结果。指标处理与当前实验一致，不声称复现论文全部默认评测设置。

建议下一步：固定 A2000 为轻量候选、B2000 为覆盖消融；先分析开发集按 QP 的 LPIPS/DISTS 梯度方向、裁剪与原生分辨率差异及时间一致性，再只改变一个因素开展受控实验。最终确认时补充事先保留且来源组独立的数据，不依据本轮测试集继续调参。
