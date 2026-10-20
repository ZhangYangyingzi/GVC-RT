# Fixed v6.38 protocol

{
  "methods": {
    "O": [],
    "B": [
      "bridge"
    ],
    "G": [
      "generator"
    ],
    "BG": [
      "bridge",
      "generator"
    ]
  },
  "updates": 100,
  "checkpoints": [
    0,
    50,
    100
  ],
  "seed": 20261034,
  "objective": "LPIPS + 0.5 DISTS + 0.1383589248954108 reconstruction_L1 + beta_q(schedule_s1p0)*rate_bpp + 0.01 proxy_L1",
  "training": {
    "batch": 1,
    "crop": 256,
    "frames": 64,
    "warmup_P": [
      1,
      48
    ],
    "supervised_P": [
      49,
      63
    ],
    "reference_detach": true,
    "precision": "FP32 trainable parameters/gradients, historical mixed codec precision",
    "reconstruction_backward_scale": 4096,
    "global_clip": 1,
    "AdamW": {
      "bridge_lr": 6e-06,
      "generator_lr": 1e-06,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "weight_decay": 0.0001
    }
  },
  "selection": {
    "identities": [
      [
        "ulong",
        "clips_long_1920/6bea102c-a661-4195-b089-d12022eac35d.mp4"
      ],
      [
        "uvg",
        "Beauty"
      ]
    ],
    "counts": {
      "ulong": 1,
      "uvg": 99
    },
    "original_first100_distinct_identities": 49,
    "rule": "historical plan filtered to first two encountered identities, then first100 matching records; original source rows/crop/QP/time remain unchanged"
  },
  "evaluation": {
    "dataset": "hevc_b",
    "videos": [
      "BasketballDrive",
      "BQTerrace"
    ],
    "QP": [
      0,
      5,
      9
    ],
    "frames": 64,
    "IP": -1,
    "native_I": true,
    "fps_accounting": 20,
    "metrics": [
      "LPIPS",
      "DISTS",
      "FloLPIPS",
      "PSNR",
      "L1"
    ],
    "paired_same_real_RANS": true
  },
  "limits": "Single seed, two training identities, two heldout test videos, 100 updates; trend screening only, no significance or generalization claim"
}

The original first100 rows contain49 identities, incompatible with a two-video screen. We fix the first encountered U-Long source and Beauty, filter the original5000-row plan to these identities and retain its first100 matching rows. No source row is regenerated or cycled. All three conditions share this derived100-row plan.
