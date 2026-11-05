# 各数据集及逐视频 R-D 曲线

这里使用已完成实验的 675 条实测记录；没有新增训练或全数据集推理。不同数据集和不同测试划分分别作图。

数据集图在各自子集内按视频等权平均，这是数据集曲线的汇总方式。下面另附 45 张逐视频曲线，完全不跨视频平均。

每张图包含 LPIPS、DISTS、FloLPIPS、PSNR、MS-SSIM、GT-flow warping error；横轴为实测 bpp。只有 QP 0、5、9 三点，不估算 BD-rate。

基线标为 Original-weight baseline (parent0)：与用户确认的原版权重在加载精度下逐张量一致，TCG 残差为零。它不是原版代码全测试集独立重跑的记录。见 [基线核查](BASELINE_FINDINGS.md)。

| 数据集/划分 | 视频数 | 曲线 |
|---|---:|---|
| ULong: new test subset | 32 | [查看](ulong_new_test_rate_quality.png) |
| ULong: historical subset | 6 | [查看](ulong_historical_rate_quality.png) |
| HEVC-B: historical subset | 5 | [查看](hevc_b_rate_quality.png) |
| UVG: held-out historical subset | 2 | [查看](uvg_holdout_rate_quality.png) |

这些是已评测子集，UVG 只有 HoneyBee、Jockey 两段，不能标为完整 UVG 评测。历史 ULong 6 段与新 ULong 32 段保持分开。

表格：[按数据集数据](rd_points_by_dataset.csv) · [逐视频原始点](rd_points_per_video.csv) · [同码率变化](same_rate_changes_by_dataset.csv)

## 逐视频曲线

| 数据集/划分 | 视频 |
|---|---|
| ulong_new_test | [081343d7-3c56-46bf-93a2-28718f33e6a7](per_video/ulong_new_test/081343d7-3c56-46bf-93a2-28718f33e6a7.png) |
| ulong_new_test | [0effaece-e02d-41d0-8f23-b301a8dc84cb](per_video/ulong_new_test/0effaece-e02d-41d0-8f23-b301a8dc84cb.png) |
| ulong_new_test | [0f709ea5-ad91-4391-9d36-ade3948025e2](per_video/ulong_new_test/0f709ea5-ad91-4391-9d36-ade3948025e2.png) |
| ulong_new_test | [12c581b7-3c2c-4701-8889-808a30c68a0f](per_video/ulong_new_test/12c581b7-3c2c-4701-8889-808a30c68a0f.png) |
| ulong_new_test | [1a271afc-ba22-4e96-9085-ed993ce85272](per_video/ulong_new_test/1a271afc-ba22-4e96-9085-ed993ce85272.png) |
| ulong_new_test | [1f88524d-47c4-4ee7-8fd5-e3831f1dace1](per_video/ulong_new_test/1f88524d-47c4-4ee7-8fd5-e3831f1dace1.png) |
| ulong_new_test | [25057ac0-fd2e-4adf-800b-8483f2fb49e3](per_video/ulong_new_test/25057ac0-fd2e-4adf-800b-8483f2fb49e3.png) |
| ulong_new_test | [3a70e370-36df-4de9-ba84-52dd56dd43b8](per_video/ulong_new_test/3a70e370-36df-4de9-ba84-52dd56dd43b8.png) |
| ulong_new_test | [3ba2cd19-6c74-4935-8725-5359c0453845](per_video/ulong_new_test/3ba2cd19-6c74-4935-8725-5359c0453845.png) |
| ulong_new_test | [4155fb19-b81e-4b1a-b845-65d716a6c21f](per_video/ulong_new_test/4155fb19-b81e-4b1a-b845-65d716a6c21f.png) |
| ulong_new_test | [5415f5ae-5cae-449e-a37f-fa24c26d2b7d](per_video/ulong_new_test/5415f5ae-5cae-449e-a37f-fa24c26d2b7d.png) |
| ulong_new_test | [62af4da7-8b12-4015-b184-c035884571a9](per_video/ulong_new_test/62af4da7-8b12-4015-b184-c035884571a9.png) |
| ulong_new_test | [6e327f54-9f3a-416d-aadb-109adf734c52](per_video/ulong_new_test/6e327f54-9f3a-416d-aadb-109adf734c52.png) |
| ulong_new_test | [756c549f-6ac1-4cc4-96b0-5eab759cb799](per_video/ulong_new_test/756c549f-6ac1-4cc4-96b0-5eab759cb799.png) |
| ulong_new_test | [76cc3375-2b9f-444c-9376-b83851d6b7d2](per_video/ulong_new_test/76cc3375-2b9f-444c-9376-b83851d6b7d2.png) |
| ulong_new_test | [7789166a-ef52-4f9f-8249-3d0623c0f69f](per_video/ulong_new_test/7789166a-ef52-4f9f-8249-3d0623c0f69f.png) |
| ulong_new_test | [778c7f22-4da2-4570-a032-a9a4a87b6b61](per_video/ulong_new_test/778c7f22-4da2-4570-a032-a9a4a87b6b61.png) |
| ulong_new_test | [7e566226-fcda-494f-b843-bf4cc5183f6e](per_video/ulong_new_test/7e566226-fcda-494f-b843-bf4cc5183f6e.png) |
| ulong_new_test | [7fce67d2-cda8-42d8-a1c8-7eb5831b603e](per_video/ulong_new_test/7fce67d2-cda8-42d8-a1c8-7eb5831b603e.png) |
| ulong_new_test | [8423fe52-7e1a-4195-b13b-9c3deea06043](per_video/ulong_new_test/8423fe52-7e1a-4195-b13b-9c3deea06043.png) |
| ulong_new_test | [8730e5b1-3d66-4baa-9f80-af4395576a99](per_video/ulong_new_test/8730e5b1-3d66-4baa-9f80-af4395576a99.png) |
| ulong_new_test | [8dbd8b63-2824-467b-b624-1ba80e5d54bb](per_video/ulong_new_test/8dbd8b63-2824-467b-b624-1ba80e5d54bb.png) |
| ulong_new_test | [9e38df87-db86-42c5-8613-4e61d1cc2133](per_video/ulong_new_test/9e38df87-db86-42c5-8613-4e61d1cc2133.png) |
| ulong_new_test | [a0792649-6ecf-48ee-9ce1-8807987542f4](per_video/ulong_new_test/a0792649-6ecf-48ee-9ce1-8807987542f4.png) |
| ulong_new_test | [a4bc7de3-0cb6-45ab-b32d-93ff3560c662](per_video/ulong_new_test/a4bc7de3-0cb6-45ab-b32d-93ff3560c662.png) |
| ulong_new_test | [b0bfea85-7212-4284-94b1-6f83d8df949c](per_video/ulong_new_test/b0bfea85-7212-4284-94b1-6f83d8df949c.png) |
| ulong_new_test | [cd1d962e-5b8a-4a14-bbdc-388c84804aaf](per_video/ulong_new_test/cd1d962e-5b8a-4a14-bbdc-388c84804aaf.png) |
| ulong_new_test | [dce599d3-d196-4d87-b2b1-698acdbbb3d4](per_video/ulong_new_test/dce599d3-d196-4d87-b2b1-698acdbbb3d4.png) |
| ulong_new_test | [e0a079c6-e4a7-484e-bfcd-cce68304a27d](per_video/ulong_new_test/e0a079c6-e4a7-484e-bfcd-cce68304a27d.png) |
| ulong_new_test | [eef96391-f11c-4c66-bcf6-5652277f2496](per_video/ulong_new_test/eef96391-f11c-4c66-bcf6-5652277f2496.png) |
| ulong_new_test | [f41c938b-7aac-42c0-8f51-ab58888e8bb6](per_video/ulong_new_test/f41c938b-7aac-42c0-8f51-ab58888e8bb6.png) |
| ulong_new_test | [f652f610-78f0-45f4-969c-47a3482b3f9d](per_video/ulong_new_test/f652f610-78f0-45f4-969c-47a3482b3f9d.png) |
| ulong_historical | [1fe6bebe-c876-4b4d-ad2c-e0caf03ad404](per_video/ulong_historical/1fe6bebe-c876-4b4d-ad2c-e0caf03ad404.png) |
| ulong_historical | [268543d9-eb03-48c4-9e91-fa8455277f71](per_video/ulong_historical/268543d9-eb03-48c4-9e91-fa8455277f71.png) |
| ulong_historical | [641202b1-6955-422a-b45f-af8c1d2b1777](per_video/ulong_historical/641202b1-6955-422a-b45f-af8c1d2b1777.png) |
| ulong_historical | [a6f9b7f7-d805-474b-a664-924f1e39d7d1](per_video/ulong_historical/a6f9b7f7-d805-474b-a664-924f1e39d7d1.png) |
| ulong_historical | [cfa79e7c-cce2-4db3-a692-55e5d4b27493](per_video/ulong_historical/cfa79e7c-cce2-4db3-a692-55e5d4b27493.png) |
| ulong_historical | [df773e20-1706-4d5b-93d9-3b534d989214](per_video/ulong_historical/df773e20-1706-4d5b-93d9-3b534d989214.png) |
| hevc_b | [BQTerrace](per_video/hevc_b/BQTerrace.png) |
| hevc_b | [BasketballDrive](per_video/hevc_b/BasketballDrive.png) |
| hevc_b | [Cactus](per_video/hevc_b/Cactus.png) |
| hevc_b | [Kimono1](per_video/hevc_b/Kimono1.png) |
| hevc_b | [ParkScene](per_video/hevc_b/ParkScene.png) |
| uvg_holdout | [HoneyBee](per_video/uvg_holdout/HoneyBee.png) |
| uvg_holdout | [Jockey](per_video/uvg_holdout/Jockey.png) |
