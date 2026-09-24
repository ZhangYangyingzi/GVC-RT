# GVC-RT V1.3 — Fixed Decoder Code Optimization

只更新每个样本自己的4080维增强码字，所有历史网络、归一化、熵模型和基础参考规则冻结。Q1连续码字与Q2量化/实际码率问题采用独立门槛。

## 复现命令

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_latent_enhancement_v1_3_code_optimization_diagnosis/reproduce.py --output reproduction_01
```

必须使用新目录，固定GPU6/UUID空闲时才启动。使用已有gvc-rt环境，不下载资源，不覆盖V1/V1.1/V1.2。新运行直接使用已修复的显式量化路径；历史无效尝试仅保留在本轮原结果中。

## 文件

- `config.json`、`interface.md`：表示域、三个不同模型、优化器、lambda单位、零输入、真实预算和选择规则。
- `bridge.py`、`prepare.py`：只读复用原模型/数据，固定6/24/90划分、实际码字尺度与起点梯度检查。
- `optimization.py`：连续u或量化域z的150步优化，硬前向、固定模型梯度、真实snapshot码流和最小目标选择。
- `stages.py`：六帧校准与独立Q1/Q2扩展门槛。
- `sequences.py`：完整1I+15P实际码流、合法解码后的wrong-source干预、整段候选预算选择，不跨lambda混帧。
- `receiver.py`：源数据访问阻断的独立接收进程；ORC2格式和V1.2解码模型不变。
- `postprocess.py`、`verify.py`、`visualize.py`：全部快照、非支配点、分组指标、真实预算、原版匹配码率、视频/裁剪/曲线和核验。
- `results_v1/optimizations/`：每个有界运行的日志、0/25/50/100/150码字、图片、B诊断码流、所选码字和检查记录。
- `results_v1/sequences/`：实际完整候选、独立接收输出、P/IP与6/24/90分组CSV、播放视频。
- `results_v1/*budget_selections.csv`：由真实整段候选选出的可行点，包含明确的无增强文件ZERO退回。
- `results_v1/invalid_attempts.json`：初期B阶段域分派错误的完整记录；该批结果不用于Q2结论。
- `report.md`：分项结论及茶园/桥梁链条。

Q1主结果最小化记录点L_image，另保留150步。Q2主结果最小化L_image+lambda×实际frame-share-bpp，包含step0。正式预算只看完整文件；102字节全零容器和0字节无流退回分开记录。

B固定ZERO_START，不需要完整latent教师或当前E前向才能编码；为对照读取教师缓存不计作编码算法依赖。A的ENCODER_START依赖V1.2教师残差和E，历史完整教师优化耗时单列，不能因缓存复用写成0或宣称实时。
