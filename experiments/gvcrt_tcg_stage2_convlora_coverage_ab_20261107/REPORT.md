# Generator Conv-LoRA coverage A/B execution snapshot

UTC: 2026-11-07T20:50:36Z

Status:RUNNING; stage:final_integrity_report_plotting.

A: stage3+head59392 parameters. B: A plus upsample.conv1(512→2048), total135168. No Router. All original/TCG weights frozen.

Both start fresh from strict pilot+B1500 overlay,TCG cumulative6500,source cursor1500. Effective batch4, same four sample keys per update. Seed20261106,lr5e-5.

Separate min_delta0.0005,patience4,validation250,cap3000. No quality-dependent configuration changes.

Existing test videos are reused; no new unseen confirmation-set claim. Final quality evaluation starts after both selections are frozen.

Full test/report/plot/publication acceptance is pending. Historical baseline reuse is separately audited.

A_control committed optimizer updates:3000.

```json

{
  "status": "EARLY_STOPPED_PATIENCE",
  "anchor": 0.9959518954881964,
  "bad_check_count": 4,
  "last_validation_step": 3000,
  "checks": [
    {
      "step": 250,
      "score": 1.000613666863154,
      "F_q": {
        "0": 0.18620783980337105,
        "5": 0.11257635448904087,
        "9": 0.09168242139276117
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "f1d1b690d25b40842bc255d32ad442e9374b5ff13d477c34f8aa65df6afd6999",
      "checkpoint_sha256": "2d2008aad2b99ce611dd3594533bfd115abce92e28e8fa7eec2baac44a1f9801",
      "time": 1794032138.4670696
    },
    {
      "step": 500,
      "score": 1.0008113012462934,
      "F_q": {
        "0": 0.1862094386865104,
        "5": 0.11262114561026147,
        "9": 0.09169950556021095
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "bad_check_count": 2,
      "threshold_improvement": false,
      "metrics_sha256": "495639b7a758e87a2a978a5ef5c8f6ef78c39c161f0e136854add6daee2b99d5",
      "checkpoint_sha256": "e26a0ded6afaf3aea6b3a3c3224b68dd001e6b3e6d68b30d257df9940caf5e7a",
      "time": 1794036329.6179652
    },
    {
      "step": 750,
      "score": 0.9992004265498498,
      "F_q": {
        "0": 0.1859978060940251,
        "5": 0.11242124951067603,
        "9": 0.09152362154083445
      },
      "anchor_before": 1.0,
      "anchor_after": 0.9992004265498498,
      "bad_check_count": 0,
      "threshold_improvement": true,
      "metrics_sha256": "b52ad5edf5668dcc337a07582e821078937ad3ef308c2594b39da5b00cfe2abf",
      "checkpoint_sha256": "77f7679a77d7904496f7db5c5668ce4bc5fc68fbd57f2b0d9237f54a99e5583f",
      "time": 1794041657.9633598
    },
    {
      "step": 1000,
      "score": 0.9987921349307806,
      "F_q": {
        "0": 0.1858851601931429,
        "5": 0.11242849854308934,
        "9": 0.09146095515554031
      },
      "anchor_before": 0.9992004265498498,
      "anchor_after": 0.9992004265498498,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "a43c41f899c146f029d9407062af25f2c08fcadd4311508b143d444f2da07930",
      "checkpoint_sha256": "f5324c98738954888d5689c74194e478491cd00a47672537a3d412860b18a83a",
      "time": 1794046597.0337346
    },
    {
      "step": 1250,
      "score": 0.9969510111635783,
      "F_q": {
        "0": 0.18558918395715338,
        "5": 0.11223686001326386,
        "9": 0.0912566066715896
      },
      "anchor_before": 0.9992004265498498,
      "anchor_after": 0.9969510111635783,
      "bad_check_count": 0,
      "threshold_improvement": true,
      "metrics_sha256": "3772193710d48a3dfe0d0c6178f114c02ee0abfd3613f6e74cfc8cc1a5b619e3",
      "checkpoint_sha256": "fce61ea8c4d1adde00baabcf59ab1dafe0c21453756846dd90a89ab91270f22d",
      "time": 1794051211.2790895
    },
    {
      "step": 1500,
      "score": 0.9988637897392859,
      "F_q": {
        "0": 0.18584758161737155,
        "5": 0.11243157480780538,
        "9": 0.09149670873727975
      },
      "anchor_before": 0.9969510111635783,
      "anchor_after": 0.9969510111635783,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "de0459288019fb060e19744dba73fdb7576f001c09f5ac1743fea420a05c2396",
      "checkpoint_sha256": "1ac16109a6210b7992cbcfc37540c45661f37922f683f2eacc8f123a94d1c8e5",
      "time": 1794055447.8255572
    },
    {
      "step": 1750,
      "score": 0.9974551386888878,
      "F_q": {
        "0": 0.1856843177820482,
        "5": 0.11226516456753459,
        "9": 0.09132530634468865
      },
      "anchor_before": 0.9969510111635783,
      "anchor_after": 0.9969510111635783,
      "bad_check_count": 2,
      "threshold_improvement": false,
      "metrics_sha256": "151dbe56910ff57f45c8d56a9175c3b0b69e640a2eb5f9532421da8968627604",
      "checkpoint_sha256": "2869cf9454b118e1191fcd801741a021ded43af803d53fa34319aab53f26fd16",
      "time": 1794059643.7963145
    },
    {
      "step": 2000,
      "score": 0.9959518954881964,
      "F_q": {
        "0": 0.18553798517862718,
        "5": 0.1120600097199222,
        "9": 0.09115110046880703
      },
      "anchor_before": 0.9969510111635783,
      "anchor_after": 0.9959518954881964,
      "bad_check_count": 0,
      "threshold_improvement": true,
      "metrics_sha256": "16989b78047bef6e06ca0f17c4fdab0cff4bde03d035efb1fa629ce4510fd8a7",
      "checkpoint_sha256": "8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e",
      "time": 1794063849.9385831
    },
    {
      "step": 2250,
      "score": 0.996674883254049,
      "F_q": {
        "0": 0.18532122440990947,
        "5": 0.11221548326734808,
        "9": 0.09133028325584111
      },
      "anchor_before": 0.9959518954881964,
      "anchor_after": 0.9959518954881964,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "fe949cb992f017017658fffdee4c69be7a06463887a4b82a454a7d4541a488d2",
      "checkpoint_sha256": "74de3d085e71e58b9e6e9878fb8d27e7e4744f220d13dad930963b6625aa6994",
      "time": 1794068033.6951177
    },
    {
      "step": 2500,
      "score": 0.9967767762204821,
      "F_q": {
        "0": 0.18538343091864906,
        "5": 0.11221136236452453,
        "9": 0.0913309815112457
      },
      "anchor_before": 0.9959518954881964,
      "anchor_after": 0.9959518954881964,
      "bad_check_count": 2,
      "threshold_improvement": false,
      "metrics_sha256": "c97cc724ba93a89a238b03afa9e0a8613d66c448bdb886fc6a683283278bf2d4",
      "checkpoint_sha256": "d7536fc45d8f4f3529585667fb65ddd60320164f407059d5792f56ac61ca743e",
      "time": 1794072232.1963432
    },
    {
      "step": 2750,
      "score": 0.9964655039639861,
      "F_q": {
        "0": 0.1854138229345341,
        "5": 0.1121356471604298,
        "9": 0.09129204720051752
      },
      "anchor_before": 0.9959518954881964,
      "anchor_after": 0.9959518954881964,
      "bad_check_count": 3,
      "threshold_improvement": false,
      "metrics_sha256": "da59a72efc8384742b87a08ce933b489ca748339f5e4e6990c790bbafd60b45b",
      "checkpoint_sha256": "7f034a95fe7f7f48908d2852964b172fd0f596d0380d9f155033504655816e32",
      "time": 1794076428.6493688
    },
    {
      "step": 3000,
      "score": 0.9961009764666139,
      "F_q": {
        "0": 0.1854088624407138,
        "5": 0.11209584468151516,
        "9": 0.09122663311704639
      },
      "anchor_before": 0.9959518954881964,
      "anchor_after": 0.9959518954881964,
      "bad_check_count": 4,
      "threshold_improvement": false,
      "metrics_sha256": "01fe759aaf4888b8f8a86e497c21b03e4fc3dc41d26958052a011da267f13bcc",
      "checkpoint_sha256": "4a38a8b244725e278da3c5c3471d6b47d91f085588122268e294e71edc384a6c",
      "time": 1794080601.4156137
    }
  ],
  "raw_best": {
    "step": 2000,
    "score": 0.9959518954881964,
    "checkpoint": "/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_coverage_ab_20261107/A_control/checkpoints/step_2000.pt",
    "checkpoint_sha256": "8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e"
  },
  "baseline_F_q": {
    "0": 0.18586495117883065,
    "5": 0.11252980649800202,
    "9": 0.0917207126289473
  },
  "baseline_metrics_sha256": "e57ebfd6c4eec8415252fe0a6066340ae37788586fdf5149a28f4fc5a1f82615",
  "initialization_engineering_checks": "PASS; no full96point candidate initialization validation claimed"
}

```

B_expanded committed optimizer updates:3000.

```json

{
  "status": "EARLY_STOPPED_PATIENCE",
  "anchor": 0.9954554167531234,
  "bad_check_count": 4,
  "last_validation_step": 3000,
  "checks": [
    {
      "step": 250,
      "score": 1.000679887872671,
      "F_q": {
        "0": 0.18625945780825404,
        "5": 0.11257459036821114,
        "9": 0.09167660833133148
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "d2f3ccaf100b1fe76a743dfcfd46976d4cbf2d2b10d2113f9a6aa9027bc776c1",
      "checkpoint_sha256": "fb7deb91d9ea08d64105df831f01c1e04d51248e3eff829bbf693d94d05c8839",
      "time": 1794033081.8049061
    },
    {
      "step": 500,
      "score": 1.0009830427888398,
      "F_q": {
        "0": 0.18623093401067078,
        "5": 0.11263444412699235,
        "9": 0.09172531546348528
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "bad_check_count": 2,
      "threshold_improvement": false,
      "metrics_sha256": "c3df718638b29543a418936417d32845a4df5c614478530f00cc2ffb73c59627",
      "checkpoint_sha256": "2093304ae7ec71a1ead54b0b9f6a089e02ef6e05cde2820df8d7b16b44e969ec",
      "time": 1794037307.486868
    },
    {
      "step": 750,
      "score": 0.999251793411255,
      "F_q": {
        "0": 0.1860243851607222,
        "5": 0.1124004054748054,
        "9": 0.09154162904901784
      },
      "anchor_before": 1.0,
      "anchor_after": 0.999251793411255,
      "bad_check_count": 0,
      "threshold_improvement": true,
      "metrics_sha256": "2a34caaa4f7b67dddd6d5aea1719d914c4315c303787984b9d4475e8a0bf2b7f",
      "checkpoint_sha256": "3395de5632b78677c1b4708549159f11224ecd3c4ff709c0b3480ded871eb2b6",
      "time": 1794042843.6087947
    },
    {
      "step": 1000,
      "score": 0.9988154526165065,
      "F_q": {
        "0": 0.18587390027223302,
        "5": 0.11240329004875901,
        "9": 0.09149347477137953
      },
      "anchor_before": 0.999251793411255,
      "anchor_after": 0.999251793411255,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "e202143b59c276433d19e66b788ede4e91d55e7c6d524e486c49f5d3f005118a",
      "checkpoint_sha256": "46b48c975f207eefe86431a9dce2be71dbdbf3b0c92bf0cf7a6adae7a6737cbf",
      "time": 1794047619.1959805
    },
    {
      "step": 1250,
      "score": 0.9972333397923557,
      "F_q": {
        "0": 0.1856127650022418,
        "5": 0.11225305167433347,
        "9": 0.09130945854953357
      },
      "anchor_before": 0.999251793411255,
      "anchor_after": 0.9972333397923557,
      "bad_check_count": 0,
      "threshold_improvement": true,
      "metrics_sha256": "97601e33d163abbe7e49987bb35bc663b4e976bd159c5d9ab60ec35ce7b144a2",
      "checkpoint_sha256": "01afafc45a120882704c89d1f5b5f4025a5a4e1d0227441e38a03f1ccdb2c894",
      "time": 1794052203.3886905
    },
    {
      "step": 1500,
      "score": 0.9988334543646588,
      "F_q": {
        "0": 0.18576897426308797,
        "5": 0.11242671037048456,
        "9": 0.09153111768232304
      },
      "anchor_before": 0.9972333397923557,
      "anchor_after": 0.9972333397923557,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "e8bf956e5b8ef6cf8e2a21ba01b767430ba964ef5569d2e13b623778cb7491f6",
      "checkpoint_sha256": "0a6092b010d5a1607e3c7c2fb83701bd89d14955f0105e3be83ffa7530635b95",
      "time": 1794056383.4414396
    },
    {
      "step": 1750,
      "score": 0.9973798801459592,
      "F_q": {
        "0": 0.1856144770207475,
        "5": 0.11225384983256281,
        "9": 0.09134828549669316
      },
      "anchor_before": 0.9972333397923557,
      "anchor_after": 0.9972333397923557,
      "bad_check_count": 2,
      "threshold_improvement": false,
      "metrics_sha256": "bea96e4ddcbbed75a6204f1fb08ddf6e53d39fa469f6898c1d6e25a79bff529b",
      "checkpoint_sha256": "ee89568bb37c14f2ae77d93fef5fce66692241d5327bd4552180e6294b92bc0d",
      "time": 1794060581.539767
    },
    {
      "step": 2000,
      "score": 0.9954554167531234,
      "F_q": {
        "0": 0.18538643301848973,
        "5": 0.11200908154973553,
        "9": 0.0911307868368495
      },
      "anchor_before": 0.9972333397923557,
      "anchor_after": 0.9954554167531234,
      "bad_check_count": 0,
      "threshold_improvement": true,
      "metrics_sha256": "d9b1cec61c59dc36489c445a633e3bb2211ea0c942fe3978cd3e74be822ef4ae",
      "checkpoint_sha256": "aa000d8e3c5701636cf4a1ecc8b26b2d7db235bbe8dac39a2acc277c1a7903eb",
      "time": 1794064787.6073008
    },
    {
      "step": 2250,
      "score": 0.9963953877935022,
      "F_q": {
        "0": 0.18517468498081027,
        "5": 0.11218503627478929,
        "9": 0.09135050773768434
      },
      "anchor_before": 0.9954554167531234,
      "anchor_after": 0.9954554167531234,
      "bad_check_count": 1,
      "threshold_improvement": false,
      "metrics_sha256": "6d1329f12ba7ecf58d96a9d6632657fe42e1799a8e96bc17ccadef07b24af1fa",
      "checkpoint_sha256": "5dfae039f7e15a2a69288b35f7122b3dbb762ebf51b64519cedbee792a6ce684",
      "time": 1794068979.3872333
    },
    {
      "step": 2500,
      "score": 0.9966437259886135,
      "F_q": {
        "0": 0.18517003549883762,
        "5": 0.11221689541753943,
        "9": 0.09139516770780559
      },
      "anchor_before": 0.9954554167531234,
      "anchor_after": 0.9954554167531234,
      "bad_check_count": 2,
      "threshold_improvement": false,
      "metrics_sha256": "b1c92359a9fed158487091046929d4f696674a053d8380572eaf88c3d8aecad7",
      "checkpoint_sha256": "a07af4508201685a9b7b33667726399d069ca3c36e27f53cb64978239766a3ea",
      "time": 1794073177.9828818
    },
    {
      "step": 2750,
      "score": 0.9961672319383673,
      "F_q": {
        "0": 0.1851757073919067,
        "5": 0.11214875458418169,
        "9": 0.09131679579889077
      },
      "anchor_before": 0.9954554167531234,
      "anchor_after": 0.9954554167531234,
      "bad_check_count": 3,
      "threshold_improvement": false,
      "metrics_sha256": "d53fa279dddf53e94b3db75dc289a46c5a6058dd899ccc0eb806d9d6ce3e25b7",
      "checkpoint_sha256": "14099c22e77646395c274ac05148088be3c06bf68276d3482722606c1c39bc88",
      "time": 1794077370.4998984
    },
    {
      "step": 3000,
      "score": 0.9955885006069903,
      "F_q": {
        "0": 0.18513249820061325,
        "5": 0.11206830423676394,
        "9": 0.09124444714850849
      },
      "anchor_before": 0.9954554167531234,
      "anchor_after": 0.9954554167531234,
      "bad_check_count": 4,
      "threshold_improvement": false,
      "metrics_sha256": "e43a3e9c01d92ca02ecec9199b3b3018049a4fa032be6dd6049f0edc1d58a86a",
      "checkpoint_sha256": "9acfbbe7965c00acdad6d1ca54c741e214691a5e3003f28f9e1836f91ae6c1f4",
      "time": 1794081541.3248258
    }
  ],
  "raw_best": {
    "step": 2000,
    "score": 0.9954554167531234,
    "checkpoint": "/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_coverage_ab_20261107/B_expanded/checkpoints/step_2000.pt",
    "checkpoint_sha256": "aa000d8e3c5701636cf4a1ecc8b26b2d7db235bbe8dac39a2acc277c1a7903eb"
  },
  "baseline_F_q": {
    "0": 0.18586495117883065,
    "5": 0.11252980649800202,
    "9": 0.0917207126289473
  },
  "baseline_metrics_sha256": "e57ebfd6c4eec8415252fe0a6066340ae37788586fdf5149a28f4fc5a1f82615",
  "initialization_engineering_checks": "PASS; no full96point candidate initialization validation claimed"
}

```
