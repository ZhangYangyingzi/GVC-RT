# VIDEO_INDEX

实验目录：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113`

所有视频均为预先固定的两个ULong开发案例（24747b35、de12efc2），连续64帧，原生1920×1080、原帧率30fps。它们是已使用开发源，不是独立最终测试。FFV1是full-range RGB8无损文件，已逐字节回读核验；MP4为BT.709限幅YUV420有损播放副本。

图例：GT；独立Original GVC-RT（只读Original I/P与独立代码）；冻结完整B300；A_zero_base（零基础位移，但offset/mask与融合可学习，仍有公共运动证据）；B_flow_guided（独立Original解码参考上的冻结双向RAFT基础引导）。LoRA s=0.5。

固定展示点100和300；beta0经64帧精确等价核验后与B300去重。beta0.25/0.5/1均独立reset/rollout，不插值最终RGB。原尺寸ROI为[800, 448, 256, 256]；每栏256×256，单独48像素标题栏，未遮挡或缩放内容。

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B_flow_guided**：step=100, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mp4`
  - SHA256：`68d11f7f7b2a4e39a76489ce2080e168a01891d1a26b96c1512d18ac12dba2c5`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / A_zero_base**：step=100, beta=0.5, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mp4`
  - SHA256：`18614090a3de69e2ff58285145c56ec035dbccfd52df2bc5a9c6bfffb8548d58`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B_flow_guided**：step=300, beta=0.5, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mp4`
  - SHA256：`38beaefcf8f2b14d2bbccdde487ca2bda45076e926702b80544ed6a2abff1a76`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / A_zero_base**：step=100, beta=0.5, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mp4`
  - SHA256：`a71e6992ed61e15b01473aaa354224937607f42830ff877973ad67c9ebb4557e`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B_flow_guided**：step=100, beta=0.25, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mp4`
  - SHA256：`5222473beba1b825a8c610fbede45f54a6605acc79ef414dfffd962e5fba5c32`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / A_zero_base**：step=100, beta=0.25, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mp4`
  - SHA256：`bb5a2ee2f26f5b6c8cce4a607d44971d155089b49225c4e192c12d8bf45c56f4`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B_flow_guided**：step=100, beta=0.5, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mp4`
  - SHA256：`5df0aea75b492c750c4c931557dcc8fe20eb7f308a504e3d2d6d78eda434711b`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B300**：step=0, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B300_s0000_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B300_s0000_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mp4`
  - SHA256：`0ef5d41670849b598c32b64e0a1a0f254bfd3baa126aa76355f4dd527a8502f3`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B_flow_guided**：step=300, beta=0.25, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mp4`
  - SHA256：`4219af6d7766b3d5141586195bfa7dbb047dbbaa09a975a783914dcdc5482d73`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / A_zero_base**：step=100, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mp4`
  - SHA256：`22a212aa7cc5f3f87b8c982b2a0b9371e264b0393fe05e0a55034542953dd4f5`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B_flow_guided**：step=300, beta=0.5, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mp4`
  - SHA256：`ea114396904d0626d80fe37abf41644842478eb991b49e3b765069f20808f344`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / Original**：step=0, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/Original_s0000_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/Original_s0000_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mp4`
  - SHA256：`9f972ff3cba7a25fd9a34ff7af9ba5af770f29917f9f2d589d89271920ba6eaa`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B_flow_guided**：step=300, beta=0.25, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mp4`
  - SHA256：`4da440956a59c5bb3e8a717b49c51de78bc07d0eabd0aad1b6ec36aaeae0e8cc`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / A_zero_base**：step=300, beta=0.25, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mp4`
  - SHA256：`c0ee526cf15936d2bfca921b9274b215f87e3cde298855af5f78d55541cb85ec`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / A_zero_base**：step=300, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mp4`
  - SHA256：`973e0b5844d0c742e1399835b941217be2649991b495464bfc92db0ef84520e3`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / A_zero_base**：step=100, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mp4`
  - SHA256：`7b128025cf7ecc605089875582f7fa9a8a7ebbd1c149bc20bc630942e815a84a`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B_flow_guided**：step=300, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B_flow_guided_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mp4`
  - SHA256：`d63a4b72dade6dc0abb458c049ff65d7b7fb91855a995295a1f7b6c41589c954`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / A_zero_base**：step=300, beta=0.25, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mp4`
  - SHA256：`75c6c4471b8f277a75bca71d261fc6895f02ed9ea166dc243b180b10bd89a682`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / B300**：step=0, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B300_s0000_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/B300_s0000_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b1.mp4`
  - SHA256：`8f3b21970abb19cd13bc0d9bd287b285c767d76a45584a8420b5f0d7c022a92a`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B_flow_guided**：step=100, beta=0.25, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.25.mp4`
  - SHA256：`9093fc0e7152103443eb93071ce22fce1d54e59e79acc75f40ccd813675d887e`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B_flow_guided**：step=300, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mp4`
  - SHA256：`7fcc4b073252b0a774016c44a2989a05b910aabdf26b025b012705fb0e724d2f`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / Original**：step=0, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/Original_s0000_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/Original_s0000_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mp4`
  - SHA256：`747382f037a5c74c0647c8031a2f269066e1ac41e02370100355637f999095b4`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / A_zero_base**：step=300, beta=0.5, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0300_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.5.mp4`
  - SHA256：`e752ad5c7e6d8feef8ebd7aa49f6e282a96b9bc9c1c440105db830c81cdc749a`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B_flow_guided**：step=100, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mp4`
  - SHA256：`948b285317e278a6aa16c8ce318f46236b4bb1627f54ed52227950e1ddcb2f57`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / A_zero_base**：step=300, beta=0.5, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mp4`
  - SHA256：`09ae2377dd58329a825533be5a95e3f757578c818b1e210f3d9db0101ec5937e`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / A_zero_base**：step=300, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/A_zero_base_s0300_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b1.mp4`
  - SHA256：`0f1de8020328d577fb8d16cde22437641158d958b9f80f23f8d4ad46401e5861`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / B_flow_guided**：step=100, beta=0.5, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/B_flow_guided_s0100_dv_24747b35-04ca-416c-b1d0-0bc670fbac99_q5_b0.5.mp4`
  - SHA256：`cfbb529ecbfe8a227f082ca36c2092ef8fc8c2f57f0dba62027b14903c23c751`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / A_zero_base**：step=100, beta=0.25, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/A_zero_base_s0100_dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254_q5_b0.25.mp4`
  - SHA256：`18827aaa0cf1044092262838fb4ade55c779e604b02ef65ae06bfeea130838da`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT**：step=0, beta=None, externalQP=None, bpp=None, source frames=0..63, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_native.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_native.mp4`
  - SHA256：`4833dc86db0dbec8b3da526b06ab960cd724ae9dccb7a4d7ac53111b2991b970`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT**：step=0, beta=None, externalQP=None, bpp=None, source frames=128..191, FPS=30/1, 1920×1080
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_native.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_native.mp4`
  - SHA256：`c8172e91353cad9e442d0c3937752074d8f6fd3688f8ba7b492a6eb578c87168`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=100, beta=0.25, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0100_beta0.25_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0100_beta0.25_native_ROI.mp4`
  - SHA256：`7d55b3029f9cd277c1537c91810fd38f55d5c99d73f4c8ce952bc003df7b8b03`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=100, beta=0.5, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0100_beta0.5_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0100_beta0.5_native_ROI.mp4`
  - SHA256：`6ef20c1cdee1927df467caf24b6ed3d6a0f77c0ccb591b63745b0a102a36bbcc`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=100, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0100_beta1_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0100_beta1_native_ROI.mp4`
  - SHA256：`c020eeb1998487aabc53bffd7d7c108674c77dda3ac39f731ad343da6f1f35bb`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=300, beta=0.25, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0300_beta0.25_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0300_beta0.25_native_ROI.mp4`
  - SHA256：`a02e8b4a76f4707f4485d98a608fd11c27a08acffb0ed343f9f52bd846c93379`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=300, beta=0.5, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0300_beta0.5_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0300_beta0.5_native_ROI.mp4`
  - SHA256：`9f1dcdf693b4f3bb987e48571191c3e5b82b345fe571bbf0abca445612fbe407`

- **dv_24747b35-04ca-416c-b1d0-0bc670fbac99 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=300, beta=1.0, externalQP=5, bpp=0.004021026234567901, source frames=0..63, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0300_beta1_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_24747b35-04ca-416c-b1d0-0bc670fbac99/GT_Original_B300_A_B_s0300_beta1_native_ROI.mp4`
  - SHA256：`411f01962380bcdd9f7a14a87d1c6c9ae677a0d4d8a79094412789d686f70568`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=100, beta=0.25, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0100_beta0.25_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0100_beta0.25_native_ROI.mp4`
  - SHA256：`359fc49eeb01bb95defd36479afa161a266c2d5b0999880fb64ab8bcbc815b97`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=100, beta=0.5, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0100_beta0.5_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0100_beta0.5_native_ROI.mp4`
  - SHA256：`8b47c5da7d771f528389fba8202edb3ecf0e41b2237288437fff64e3bf57d3fd`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=100, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0100_beta1_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0100_beta1_native_ROI.mp4`
  - SHA256：`ba9898791e4db9e91d8368ef7f97c6967180c6dbf1df92c7f71955db407b3c7e`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=300, beta=0.25, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0300_beta0.25_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0300_beta0.25_native_ROI.mp4`
  - SHA256：`92c1537e3f2a35e06234d2a008a4479841863a1c79d7bd30eab4d046490fd678`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=300, beta=0.5, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0300_beta0.5_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0300_beta0.5_native_ROI.mp4`
  - SHA256：`233a1ee7dc1d4597ae15e5c4720713fd1da27afe8c445461fc9601668e440f22`

- **dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254 / GT | Original | B300 | A_zero_base | B_flow_guided**：step=300, beta=1.0, externalQP=5, bpp=0.011998396508487654, source frames=128..191, FPS=30/1, 1280×304
  - FFV1：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0300_beta1_native_ROI.mkv`
  - MP4：`/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/bridge_flowguided_deformable_ab_20261113/experiments/gvcrt_bridge_flowguided_deformable_ab_20261113/videos/dv_de12efc2-a6f3-4e96-a74a-921fc5fa2254/GT_Original_B300_A_B_s0300_beta1_native_ROI.mp4`
  - SHA256：`a04fc590050d9cce1e208d04513377f250e999f3cd0ac04fef3a7402ff86e58c`

完整路径、大小、FFV1/MP4哈希、RGB8回读哈希及对应checkpoint身份见 audits/video_delivery.json 和 large_artifacts.json。大型媒体仅保存在本地项目内，未上传普通Git。
