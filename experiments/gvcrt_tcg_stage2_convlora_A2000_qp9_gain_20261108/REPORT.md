# A2000 fixed-QP inference gain diagnostic

Status: COMPLETE (local evaluation, audits, tables and plots). Publication is recorded separately in publication_status.json.

Checkpoint: step_2000.pt; SHA256 8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e. A_control LoRA step2000; frozen TCG initialization cumulative6500. Composite B1500 matched-FP32 is not original GVC-RT.

No training,optimizer,loss changes,weight updates or new checkpoint. Five existing LoRA branches only: head, stage3.blocks.0.dc.2, stage3.blocks.1.dc.2, stage3.blocks.2.dc.2, stage3.blocks.3.dc.2. Runtime multiplier s multiplies the FP32 LoRA delta, then FP32 addition and cast to base output dtype. rank4,alpha4,alpha/r1 unchanged. TCG,Bridge,base convolutions and final RGB are not gain-scaled.

Scope: existing32 development videos,externalQP9 only,full64frames. The same gain applies to all63P frames including inherited internalQP changes. Separate new model and empty temporal state per measured video/gain. Smoke sequence chosen lexicographically before reading new metrics.

Source/crop/frame RGB/actual QP/stream bytes/bpp/SHA/codec traces and GT-flow hashes agree. Full-sequence endpoint reconstructions are exact; inherited metric tolerances unchanged.64 historical endpoint points reused after audits;32 s0.5 points independently decoded,including the matching smoke point counted once.

Metrics: matched_rgb_fp32_v2,float before clamp/range conversion and native crop. Spatial metrics aggregate64frames; temporal metrics63transitions; PSNR uses total pixel SSE. PWC sampler remains pixel-center,align_cornersFalse,W-1/H-1 normalization,invalidzero,full-image denominator. GT-flow and reconstruction-flow errors remain separate.

Means weight videos equally. Relative percent=(mean_candidate/mean_reference-1)*100. No otherQP,checkpoint,finaltest,training or gain scan was run. These are fixed-QP gain curves,not R-D curves or BD-rate.

## Measured means

|s|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|GT-flow warp|Recon-flow warp|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.0|0.0159928517283|26.9708848316|0.925655269937|0.0964490470287|0.0617947800492|0.0917207126289|0.0343932256607|0.0343774236944|
|0.5|0.0159928517283|26.9796072356|0.925621311995|0.0961138550028|0.0618055227678|0.091397592906|0.0343790932398|0.0343615369389|
|1.0|0.0159928517283|26.9955409057|0.925544761267|0.0958670216878|0.0619303396088|0.0911511004688|0.0343345537364|0.0343151811523|

## Absolute and relative changes / video counts

```json

{
  "changes": {
    "s0.5_vs_s0": {
      "bpp": {
        "candidate_mean": 0.0159928517282745,
        "reference_mean": 0.0159928517282745,
        "absolute_delta": 0.0,
        "relative_percent": 0.0,
        "lower_is_better": null
      },
      "kbps": {
        "candidate_mean": 967.6457345896291,
        "reference_mean": 967.6457345896291,
        "absolute_delta": 0.0,
        "relative_percent": 0.0,
        "lower_is_better": null
      },
      "PSNR": {
        "candidate_mean": 26.979607235583835,
        "reference_mean": 26.970884831632187,
        "absolute_delta": 0.008722403951647806,
        "relative_percent": 0.03234007340173761,
        "lower_is_better": false
      },
      "MS_SSIM": {
        "candidate_mean": 0.9256213119952008,
        "reference_mean": 0.9256552699371241,
        "absolute_delta": -3.395794192329049e-05,
        "relative_percent": -0.0036685300701222445,
        "lower_is_better": false
      },
      "LPIPS": {
        "candidate_mean": 0.09611385500284086,
        "reference_mean": 0.09644904702872736,
        "absolute_delta": -0.0003351920258864993,
        "relative_percent": -0.34753275041344844,
        "lower_is_better": true
      },
      "DISTS": {
        "candidate_mean": 0.061805522767826915,
        "reference_mean": 0.06179478004924022,
        "absolute_delta": 1.0742718586698174e-05,
        "relative_percent": 0.017384508170659174,
        "lower_is_better": true
      },
      "FloLPIPS": {
        "candidate_mean": 0.09139759290597535,
        "reference_mean": 0.0917207126289473,
        "absolute_delta": -0.0003231197229719568,
        "relative_percent": -0.3522865378064899,
        "lower_is_better": true
      },
      "WarpingError_GTFlow_PWC": {
        "candidate_mean": 0.03437909323977129,
        "reference_mean": 0.03439322566070283,
        "absolute_delta": -1.4132420931539658e-05,
        "relative_percent": -0.04109071091777938,
        "lower_is_better": true
      },
      "WarpingError_ReconFlow_PWC": {
        "candidate_mean": 0.03436153693889667,
        "reference_mean": 0.03437742369437945,
        "absolute_delta": -1.588675548278018e-05,
        "relative_percent": -0.04621275760515742,
        "lower_is_better": true
      }
    },
    "s0.5_vs_s1": {
      "bpp": {
        "candidate_mean": 0.0159928517282745,
        "reference_mean": 0.0159928517282745,
        "absolute_delta": 0.0,
        "relative_percent": 0.0,
        "lower_is_better": null
      },
      "kbps": {
        "candidate_mean": 967.6457345896291,
        "reference_mean": 967.6457345896291,
        "absolute_delta": 0.0,
        "relative_percent": 0.0,
        "lower_is_better": null
      },
      "PSNR": {
        "candidate_mean": 26.979607235583835,
        "reference_mean": 26.995540905708474,
        "absolute_delta": -0.015933670124638866,
        "relative_percent": -0.05902334085577943,
        "lower_is_better": false
      },
      "MS_SSIM": {
        "candidate_mean": 0.9256213119952008,
        "reference_mean": 0.9255447612667922,
        "absolute_delta": 7.655072840861976e-05,
        "relative_percent": 0.008270883442085264,
        "lower_is_better": false
      },
      "LPIPS": {
        "candidate_mean": 0.09611385500284086,
        "reference_mean": 0.09586702168780903,
        "absolute_delta": 0.0002468333150318358,
        "relative_percent": 0.2574746880482426,
        "lower_is_better": true
      },
      "DISTS": {
        "candidate_mean": 0.061805522767826915,
        "reference_mean": 0.06193033960880712,
        "absolute_delta": -0.00012481684098020196,
        "relative_percent": -0.20154393108229307,
        "lower_is_better": true
      },
      "FloLPIPS": {
        "candidate_mean": 0.09139759290597535,
        "reference_mean": 0.09115110046880703,
        "absolute_delta": 0.00024649243716831637,
        "relative_percent": 0.2704217896444039,
        "lower_is_better": true
      },
      "WarpingError_GTFlow_PWC": {
        "candidate_mean": 0.03437909323977129,
        "reference_mean": 0.034334553736409854,
        "absolute_delta": 4.4539503361439026e-05,
        "relative_percent": 0.12972209775428123,
        "lower_is_better": true
      },
      "WarpingError_ReconFlow_PWC": {
        "candidate_mean": 0.03436153693889667,
        "reference_mean": 0.03431518115232595,
        "absolute_delta": 4.6355786570716206e-05,
        "relative_percent": 0.1350882758419436,
        "lower_is_better": true
      }
    }
  },
  "counts": {
    "s0.5_vs_s0": {
      "bpp": {
        "videos": 32,
        "improved": 0,
        "worsened": 0,
        "equal": 32,
        "direction": "rate_identity_only"
      },
      "kbps": {
        "videos": 32,
        "improved": 0,
        "worsened": 0,
        "equal": 32,
        "direction": "rate_identity_only"
      },
      "PSNR": {
        "videos": 32,
        "improved": 20,
        "worsened": 12,
        "equal": 0,
        "direction": "higher_is_better"
      },
      "MS_SSIM": {
        "videos": 32,
        "improved": 14,
        "worsened": 18,
        "equal": 0,
        "direction": "higher_is_better"
      },
      "LPIPS": {
        "videos": 32,
        "improved": 29,
        "worsened": 3,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "DISTS": {
        "videos": 32,
        "improved": 18,
        "worsened": 14,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "FloLPIPS": {
        "videos": 32,
        "improved": 30,
        "worsened": 2,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "WarpingError_GTFlow_PWC": {
        "videos": 32,
        "improved": 20,
        "worsened": 12,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "WarpingError_ReconFlow_PWC": {
        "videos": 32,
        "improved": 20,
        "worsened": 12,
        "equal": 0,
        "direction": "lower_is_better"
      }
    },
    "s0.5_vs_s1": {
      "bpp": {
        "videos": 32,
        "improved": 0,
        "worsened": 0,
        "equal": 32,
        "direction": "rate_identity_only"
      },
      "kbps": {
        "videos": 32,
        "improved": 0,
        "worsened": 0,
        "equal": 32,
        "direction": "rate_identity_only"
      },
      "PSNR": {
        "videos": 32,
        "improved": 6,
        "worsened": 26,
        "equal": 0,
        "direction": "higher_is_better"
      },
      "MS_SSIM": {
        "videos": 32,
        "improved": 21,
        "worsened": 11,
        "equal": 0,
        "direction": "higher_is_better"
      },
      "LPIPS": {
        "videos": 32,
        "improved": 7,
        "worsened": 25,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "DISTS": {
        "videos": 32,
        "improved": 19,
        "worsened": 13,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "FloLPIPS": {
        "videos": 32,
        "improved": 7,
        "worsened": 25,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "WarpingError_GTFlow_PWC": {
        "videos": 32,
        "improved": 8,
        "worsened": 24,
        "equal": 0,
        "direction": "lower_is_better"
      },
      "WarpingError_ReconFlow_PWC": {
        "videos": 32,
        "improved": 8,
        "worsened": 24,
        "equal": 0,
        "direction": "lower_is_better"
      }
    }
  }
}

```

## Objective boolean records

```json

{
  "s05_mean_FloLPIPS_and_DISTS_lower_than_s1": false,
  "s05_mean_FloLPIPS_lower_than_s0": true,
  "s05_mean_DISTS_higher_than_s0": true
}

```

## Files

metrics_s05.csv;endpoint_metrics.csv;metrics_combined.csv;paired_deltas.csv;summary.json;stream_identity_checks.csv;plots/mean_metrics_vs_gain.png;plots/per_video_paired_deltas.png;plots/video_index.csv; evaluation/s05/{points,frames,transitions,codec_trace,masks};endpoints/s0/ and endpoints/s1/;audits/smoke.json;audits/endpoint_reuse.json;sources.json;final_integrity.json. Historical source files remain unchanged.
