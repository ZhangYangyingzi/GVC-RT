# Stage2 Router-only execution record


Snapshot UTC: 2026-11-05T10:09:39Z

Status: FAILED; stage: development_validation_complete; committed Router updates: 250.

Baseline: B+1500, cumulative TCG step6500; only the new 33889-parameter Router is trainable.

Frozen data: ULong pool8470, quarantined source groups236; source-plan cursor starts at1500.

Cap3000; validation every250 updates; FloLPIPS normalized to B+1500; patience4; min_delta0.001.

Codec FP16; training Bridge/generator/Router FP32; evaluation Bridge/generator FP16 and Router FP32.

No final PASS is claimed before full integrity validation.

```json

{
  "status": "RUNNING",
  "anchor": 1.0,
  "bad_check_count": 1,
  "checks": [
    {
      "step": 250,
      "score": 0.9999870072228473,
      "F_q": {
        "0": 0.18585476332292375,
        "5": 0.11253462297498204,
        "9": 0.09171842563424318
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 1,
      "metrics_sha256": "3bb9d7189b3264fe9223dc95f779f5658c6b18a40cf04face4f0445b837e2824",
      "checkpoint_sha256": "d81802f590a3a63586849b5d69bd8f7fbad8db1f66c86ac5b36a472334072683",
      "time": 1793873379.667765
    }
  ],
  "enabled0": {
    "score": 1.00016385591597,
    "F_q": {
      "0": 0.18586198494969203,
      "5": 0.11256397703832519,
      "9": 0.09173959893422083
    },
    "path": "/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_router_only_20261105/experiments/gvcrt_tcg_stage2_router_only_20261105/evaluation/development_router_0000/metrics.csv"
  },
  "baseline_F_q": {
    "0": 0.18585684113130566,
    "5": 0.11253160509162598,
    "9": 0.09172343527565577
  },
  "raw_best": {
    "kind": "trained_router",
    "step": 250,
    "score": 0.9999870072228473,
    "checkpoint": "/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_router_only_20261105/experiments/gvcrt_tcg_stage2_router_only_20261105/checkpoints/step_250.pt",
    "checkpoint_sha256": "d81802f590a3a63586849b5d69bd8f7fbad8db1f66c86ac5b36a472334072683"
  },
  "best_trained": {
    "kind": "trained_router",
    "step": 250,
    "score": 0.9999870072228473,
    "checkpoint": "/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_router_only_20261105/experiments/gvcrt_tcg_stage2_router_only_20261105/checkpoints/step_250.pt",
    "checkpoint_sha256": "d81802f590a3a63586849b5d69bd8f7fbad8db1f66c86ac5b36a472334072683"
  },
  "last_validation_step": 250
}

```
