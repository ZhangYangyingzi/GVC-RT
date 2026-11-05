# GVC-RT 原版独立评测

已完成 45 个视频 × 3 个 QP = 135 个原版实测点。原版代码固定于 `b2a0895c0cf31cf2be3ab1c2571ca07c05866c03`，直接加载项目 checkpoints 中的原版 I/P 权重，不包含 TCG，不加载 v6 实验检查点。

每点均重新编码为实际 RANS 码流，并用新建解码器独立解码。视频、64 帧窗口、QP 调度、精度、预处理和指标计算与已有 TCG 实验对齐。这里是同协议下的原版编解码实现评测，不声称复现论文全部默认测试设置。

与 parent0 比较：135/135 点码流完全相同，135/135 点重建哈希完全相同。若两条曲线重合，仍保留独立结果和图例，不复制 parent0 数值代替原版测试。

每张图有六条线：GVC-RT original、parent0（零残差实验基线）、TCG parent5000、A+1500、A 最优+500、B 最优+1500。

| 数据集/划分 | 视频数 | 图 |
|---|---:|---|
| ULong 新测试集 | 32 | [R-D](plots/ulong_new_test_rate_quality.png) |
| ULong 历史子集 | 6 | [R-D](plots/ulong_historical_rate_quality.png) |
| HEVC-B 历史子集 | 5 | [R-D](plots/hevc_b_rate_quality.png) |
| UVG 历史子集（HoneyBee、Jockey） | 2 | [R-D](plots/uvg_holdout_rate_quality.png) |

数据集图只在各自子集内部平均；逐视频图完全不跨视频平均。各图只有 QP 0、5、9 三个实测点，未估算 BD-rate。

[原版逐点指标](metrics_original.csv) · [与 parent0 逐点比较](original_vs_parent0.csv) · [完整性检查](final_integrity.json)

## 逐视频曲线

- hevc_b: [BQTerrace](plots/per_video/hevc_b/BQTerrace.png)
- hevc_b: [BasketballDrive](plots/per_video/hevc_b/BasketballDrive.png)
- hevc_b: [Cactus](plots/per_video/hevc_b/Cactus.png)
- hevc_b: [Kimono1](plots/per_video/hevc_b/Kimono1.png)
- hevc_b: [ParkScene](plots/per_video/hevc_b/ParkScene.png)
- ulong: [1fe6bebe-c876-4b4d-ad2c-e0caf03ad404](plots/per_video/ulong_historical/1fe6bebe-c876-4b4d-ad2c-e0caf03ad404.png)
- ulong: [268543d9-eb03-48c4-9e91-fa8455277f71](plots/per_video/ulong_historical/268543d9-eb03-48c4-9e91-fa8455277f71.png)
- ulong: [641202b1-6955-422a-b45f-af8c1d2b1777](plots/per_video/ulong_historical/641202b1-6955-422a-b45f-af8c1d2b1777.png)
- ulong: [a6f9b7f7-d805-474b-a664-924f1e39d7d1](plots/per_video/ulong_historical/a6f9b7f7-d805-474b-a664-924f1e39d7d1.png)
- ulong: [cfa79e7c-cce2-4db3-a692-55e5d4b27493](plots/per_video/ulong_historical/cfa79e7c-cce2-4db3-a692-55e5d4b27493.png)
- ulong: [df773e20-1706-4d5b-93d9-3b534d989214](plots/per_video/ulong_historical/df773e20-1706-4d5b-93d9-3b534d989214.png)
- uvg_holdout: [HoneyBee](plots/per_video/uvg_holdout/HoneyBee.png)
- uvg_holdout: [Jockey](plots/per_video/uvg_holdout/Jockey.png)
- ulong_new: [081343d7-3c56-46bf-93a2-28718f33e6a7](plots/per_video/ulong_new_test/081343d7-3c56-46bf-93a2-28718f33e6a7.png)
- ulong_new: [0effaece-e02d-41d0-8f23-b301a8dc84cb](plots/per_video/ulong_new_test/0effaece-e02d-41d0-8f23-b301a8dc84cb.png)
- ulong_new: [0f709ea5-ad91-4391-9d36-ade3948025e2](plots/per_video/ulong_new_test/0f709ea5-ad91-4391-9d36-ade3948025e2.png)
- ulong_new: [12c581b7-3c2c-4701-8889-808a30c68a0f](plots/per_video/ulong_new_test/12c581b7-3c2c-4701-8889-808a30c68a0f.png)
- ulong_new: [1a271afc-ba22-4e96-9085-ed993ce85272](plots/per_video/ulong_new_test/1a271afc-ba22-4e96-9085-ed993ce85272.png)
- ulong_new: [1f88524d-47c4-4ee7-8fd5-e3831f1dace1](plots/per_video/ulong_new_test/1f88524d-47c4-4ee7-8fd5-e3831f1dace1.png)
- ulong_new: [25057ac0-fd2e-4adf-800b-8483f2fb49e3](plots/per_video/ulong_new_test/25057ac0-fd2e-4adf-800b-8483f2fb49e3.png)
- ulong_new: [3a70e370-36df-4de9-ba84-52dd56dd43b8](plots/per_video/ulong_new_test/3a70e370-36df-4de9-ba84-52dd56dd43b8.png)
- ulong_new: [3ba2cd19-6c74-4935-8725-5359c0453845](plots/per_video/ulong_new_test/3ba2cd19-6c74-4935-8725-5359c0453845.png)
- ulong_new: [4155fb19-b81e-4b1a-b845-65d716a6c21f](plots/per_video/ulong_new_test/4155fb19-b81e-4b1a-b845-65d716a6c21f.png)
- ulong_new: [5415f5ae-5cae-449e-a37f-fa24c26d2b7d](plots/per_video/ulong_new_test/5415f5ae-5cae-449e-a37f-fa24c26d2b7d.png)
- ulong_new: [62af4da7-8b12-4015-b184-c035884571a9](plots/per_video/ulong_new_test/62af4da7-8b12-4015-b184-c035884571a9.png)
- ulong_new: [6e327f54-9f3a-416d-aadb-109adf734c52](plots/per_video/ulong_new_test/6e327f54-9f3a-416d-aadb-109adf734c52.png)
- ulong_new: [756c549f-6ac1-4cc4-96b0-5eab759cb799](plots/per_video/ulong_new_test/756c549f-6ac1-4cc4-96b0-5eab759cb799.png)
- ulong_new: [76cc3375-2b9f-444c-9376-b83851d6b7d2](plots/per_video/ulong_new_test/76cc3375-2b9f-444c-9376-b83851d6b7d2.png)
- ulong_new: [7789166a-ef52-4f9f-8249-3d0623c0f69f](plots/per_video/ulong_new_test/7789166a-ef52-4f9f-8249-3d0623c0f69f.png)
- ulong_new: [778c7f22-4da2-4570-a032-a9a4a87b6b61](plots/per_video/ulong_new_test/778c7f22-4da2-4570-a032-a9a4a87b6b61.png)
- ulong_new: [7e566226-fcda-494f-b843-bf4cc5183f6e](plots/per_video/ulong_new_test/7e566226-fcda-494f-b843-bf4cc5183f6e.png)
- ulong_new: [7fce67d2-cda8-42d8-a1c8-7eb5831b603e](plots/per_video/ulong_new_test/7fce67d2-cda8-42d8-a1c8-7eb5831b603e.png)
- ulong_new: [8423fe52-7e1a-4195-b13b-9c3deea06043](plots/per_video/ulong_new_test/8423fe52-7e1a-4195-b13b-9c3deea06043.png)
- ulong_new: [8730e5b1-3d66-4baa-9f80-af4395576a99](plots/per_video/ulong_new_test/8730e5b1-3d66-4baa-9f80-af4395576a99.png)
- ulong_new: [8dbd8b63-2824-467b-b624-1ba80e5d54bb](plots/per_video/ulong_new_test/8dbd8b63-2824-467b-b624-1ba80e5d54bb.png)
- ulong_new: [9e38df87-db86-42c5-8613-4e61d1cc2133](plots/per_video/ulong_new_test/9e38df87-db86-42c5-8613-4e61d1cc2133.png)
- ulong_new: [a0792649-6ecf-48ee-9ce1-8807987542f4](plots/per_video/ulong_new_test/a0792649-6ecf-48ee-9ce1-8807987542f4.png)
- ulong_new: [a4bc7de3-0cb6-45ab-b32d-93ff3560c662](plots/per_video/ulong_new_test/a4bc7de3-0cb6-45ab-b32d-93ff3560c662.png)
- ulong_new: [b0bfea85-7212-4284-94b1-6f83d8df949c](plots/per_video/ulong_new_test/b0bfea85-7212-4284-94b1-6f83d8df949c.png)
- ulong_new: [cd1d962e-5b8a-4a14-bbdc-388c84804aaf](plots/per_video/ulong_new_test/cd1d962e-5b8a-4a14-bbdc-388c84804aaf.png)
- ulong_new: [dce599d3-d196-4d87-b2b1-698acdbbb3d4](plots/per_video/ulong_new_test/dce599d3-d196-4d87-b2b1-698acdbbb3d4.png)
- ulong_new: [e0a079c6-e4a7-484e-bfcd-cce68304a27d](plots/per_video/ulong_new_test/e0a079c6-e4a7-484e-bfcd-cce68304a27d.png)
- ulong_new: [eef96391-f11c-4c66-bcf6-5652277f2496](plots/per_video/ulong_new_test/eef96391-f11c-4c66-bcf6-5652277f2496.png)
- ulong_new: [f41c938b-7aac-42c0-8f51-ab58888e8bb6](plots/per_video/ulong_new_test/f41c938b-7aac-42c0-8f51-ab58888e8bb6.png)
- ulong_new: [f652f610-78f0-45f4-969c-47a3482b3f9d](plots/per_video/ulong_new_test/f652f610-78f0-45f4-969c-47a3482b3f9d.png)
