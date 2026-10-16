# v6.33 fixed protocol

Fixed before training and evaluation.

{
  "schema": "v633_p_causal",
  "status": "FIXED_BEFORE_TRAINING",
  "seed": 20261024,
  "steps": 1000,
  "sanity_steps": [
    0,
    100,
    250
  ],
  "primary_checkpoint": 1000,
  "lambda_rec": 0.1383589248954108,
  "loss": "LPIPS + 0.5*DISTS(require_grad=True) + beta(q)*rate_bpp + 0.01*proxy_L1 + branch_lambda_rec*mean(abs(rec-GT))",
  "objective_choice": "v6.22 B1 fixed objective; no v6.32 C/D distribution additions or C1 stat addition",
  "optimizer": {
    "wrapper_lr": 5e-05,
    "bridge_lr": 3e-06,
    "generator_lr": 5e-07,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "lr_schedule": "constant"
  },
  "training_plan": "/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/plans/shared.json",
  "receiver_initialization": "original GVC-RT_P.pt receiver for all four branches",
  "P_initialization": "existing NeuralWrapper architecture, fresh zero-output identity initialization; C/D identical seed",
  "source_B1_role": "verified historical reference, not loaded as initialization, to avoid inherited joint adaptation confounding original receiver factorial",
  "bypass": "torch.nn.Identity; no normalization/clipping/geometry operation",
  "clip_length": 64,
  "crop": 256,
  "batch": 1,
  "supervised_positions": [
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63
  ],
  "I_frame_protocol": "native RGB I0 bypass; IP=-1",
  "external_qps": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "allowed_gpus": [
    4,
    5,
    6,
    7
  ],
  "branches": [
    "A_noP_frozenReceiver",
    "B_noP_trainReceiver",
    "C_P_train_frozenReceiver",
    "D_P_train_trainReceiver"
  ],
  "matching": {
    "primary_relative_bpp_tolerance": 0.01,
    "secondary_relative_bpp_tolerance": 0.02,
    "selection": "rate only; no quality-based pairing",
    "denominator": "no-P bpp"
  },
  "rate_terms": {
    "actual": "total persisted RANS bytes including SPS/header/I frame",
    "channel": "estimated entropy-model channel rate only; never actual per-channel RANS bits"
  },
  "utility": {
    "interventions": [
      "zero",
      "per-channel spatial mean",
      "attenuation 0.95"
    ],
    "position": "post entropy-decode y_hat entering pm.dec",
    "classification": "channel generative sensitivity / finite-difference proxy"
  },
  "latency": {
    "warmup": 20,
    "iterations": 100,
    "CUDA_synchronize": true
  },
  "no_test_model_selection": true
}

Training reuses the v6.22 B1 loss and v6.32b shared sample/crop/QP plan, 64-frame native-I causal rollout with 48 warmup P frames and 15 supervised P frames. AdamW is fresh; receiver initialization is the original deployed receiver in every branch. This is a fresh factorial training experiment, not continuation of the historically jointly adapted B1. The specified B1 checkpoint is hash-verified as a historical reference. Compression stays frozen. No tests select objectives, hyperparameters or checkpoints.

Sanity must pass at step0/100/250 before progressing to1000. Actual evaluation uses the existing independent-decode RANS/metric pipeline and fixed five datasets. Receiver trainability and P presence are the only branch factors. No-P means identity bypass. A has no training. Channel deletion/attenuation occurs after decoding, never by modifying stream bits. Channel rate attribution, if verifiable, is an entropy-model estimate; only complete persisted bitstreams provide actual rate. Utility is not a mathematical marginal bit utility or KKT optimality proof.
