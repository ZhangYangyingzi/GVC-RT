# Generator Conv-LoRA execution report

UTC: 2026-11-06T20:02:57Z

Integrity:PASS. Committed LoRA updates:1000. Stop reason:EARLY_STOPPED_PATIENCE.

Best trained dev step:1000; score:0.999150799871; final retention selection:trained_LoRA. Both baseline and best trained LoRA were tested135points each.

Baseline: original I/P codec plus exact pilot Bridge/generator and B+1500 FP32 TCG overlay (TCG cumulative6500). This composite is not original GVC-RT weights. No Router or prior gating checkpoint was loaded.

The pilot SHA transcription correction was explicitly authorized by the user; original blocked records and correction reason are preserved in audits/sha_correction_20261106/. All actual weights were rehashed; checks were not waived.

Only59392 LoRA parameters train. Four rank4 parallel1x1+3x3 updates sit beside stage3 depthwise3x3 layers (320→4→320), not a dense weight factorization; the head branch is320→4→192. alpha/r=1, no added bias, up kernels zero-initialized. TCG,codec,Bridge,original generator and quality networks remain frozen.

Fresh AdamW lr5e-5,betas(.9,.999),eps1e-8,weight_decay1e-4,global clip1. No LR schedule or joint unfreezing. Effective batch4=2UVG+2ULong; source cursor1500; native64frames/256crop; original warmup and15 supervised-frame mean.

Loss.py and VJP4096/unscale/order retain historical SHA. LPIPS,DISTS,RGBL1 andRAFT-motion gradient contributions were checked. Frozen codec rate is a constant/monitoring term for LoRA; Identity proxy is zero.

LoRA and TCG are FP32 internally; deployment nonTCG retains historical FP16 and scoring is RGB float-before-clamp. Stage3/head explicitly use safe PyTorch operators from both generator entries; actual CUDA-kernel availability is recorded. No native fused LoRA kernel is claimed.

Baseline and every candidate independently decoded the same verified pre-encoded streams. Fresh encoding was used only for the fixed engineering RANS identity probes. Every dev/final point has exact paired bpp,byte and stream-SHA checks; decoded codec feature traces agree.

Four dataset/split groups are separate; video weights are equal within each group/QP. UVG historical is only HoneyBee/Jockey. Each curve uses the3 measured QPs0/5/9; no extrapolation orBD-rate.

GT-flow and reconstruction-flow PWC metrics remain separately named with the inherited sampler. No scientific-effectiveness, convergence or next-stage conclusion is drawn.

|Dataset|Model|QP|Videos|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|GT-flow warp|Recon-flow warp|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ULong_new_holdout_32|B+1500 matched-FP32 baseline (pilot receiver + TCG)|0|32|0.001980046402|23.88333067|0.8168819782|0.1981636482|0.1205279703|0.2021595945|0.05094161127|0.05063019304|
|ULong_new_holdout_32|B+1500 matched-FP32 baseline (pilot receiver + TCG)|5|32|0.007415758298|26.64194659|0.8952858929|0.118728414|0.07355130435|0.116105985|0.03855347756|0.03886104514|
|ULong_new_holdout_32|B+1500 matched-FP32 baseline (pilot receiver + TCG)|9|32|0.01681243331|27.7081501|0.9194805727|0.09691834704|0.06081058766|0.09323129702|0.03486007888|0.03496419081|
|ULong_new_holdout_32|Generator Conv-LoRA rank4 (dev-selected checkpoint)|0|32|0.001980046402|23.90915301|0.817095633|0.1983342111|0.1211142878|0.2024123794|0.05078755042|0.05048140412|
|ULong_new_holdout_32|Generator Conv-LoRA rank4 (dev-selected checkpoint)|5|32|0.007415758298|26.65333886|0.8952710565|0.1186101554|0.07396250681|0.115912716|0.03849175144|0.03878314969|
|ULong_new_holdout_32|Generator Conv-LoRA rank4 (dev-selected checkpoint)|9|32|0.01681243331|27.71326839|0.9193660705|0.09675838091|0.06119213437|0.09298287941|0.03482841074|0.03492018505|
|ULong_historical_6|B+1500 matched-FP32 baseline (pilot receiver + TCG)|0|6|0.002052810732|22.83041904|0.8161183194|0.1967703515|0.1238921306|0.2043696201|0.0522550004|0.05253352468|
|ULong_historical_6|B+1500 matched-FP32 baseline (pilot receiver + TCG)|5|6|0.007712653517|25.37685206|0.8949712214|0.1215371792|0.08082431462|0.1222019345|0.03977694534|0.03969857029|
|ULong_historical_6|B+1500 matched-FP32 baseline (pilot receiver + TCG)|9|6|0.01722558795|26.30716294|0.9178283014|0.09985707741|0.06999322074|0.0969102909|0.03629650461|0.03608337693|
|ULong_historical_6|Generator Conv-LoRA rank4 (dev-selected checkpoint)|0|6|0.002052810732|22.84444623|0.8162466581|0.1971051369|0.1244377435|0.2040496633|0.05212354429|0.05240285943|
|ULong_historical_6|Generator Conv-LoRA rank4 (dev-selected checkpoint)|5|6|0.007712653517|25.36772633|0.8948892124|0.1215785871|0.08097431622|0.1222375511|0.039757871|0.03967625089|
|ULong_historical_6|Generator Conv-LoRA rank4 (dev-selected checkpoint)|9|6|0.01722558795|26.28927016|0.9176821037|0.09981899393|0.07003629534|0.09675446995|0.03631469519|0.03609995008|
|HEVC_B_5|B+1500 matched-FP32 baseline (pilot receiver + TCG)|0|5|0.002624144001|22.38442199|0.777361542|0.2638753167|0.1158041239|0.2536200784|0.05617009669|0.05566481004|
|HEVC_B_5|B+1500 matched-FP32 baseline (pilot receiver + TCG)|5|5|0.00941855951|25.00576522|0.8707301171|0.180678259|0.08748065215|0.1824935212|0.04361858948|0.04322722694|
|HEVC_B_5|B+1500 matched-FP32 baseline (pilot receiver + TCG)|9|5|0.0207631414|26.03316208|0.8977229198|0.1530851423|0.07680772338|0.149664619|0.04001376273|0.03979278253|
|HEVC_B_5|Generator Conv-LoRA rank4 (dev-selected checkpoint)|0|5|0.002624144001|22.39680533|0.7771460077|0.2614928675|0.115300414|0.2506146034|0.05611250895|0.05559359848|
|HEVC_B_5|Generator Conv-LoRA rank4 (dev-selected checkpoint)|5|5|0.00941855951|25.00640899|0.8704239456|0.1784890901|0.08687464371|0.1797758492|0.04360201326|0.0431948368|
|HEVC_B_5|Generator Conv-LoRA rank4 (dev-selected checkpoint)|9|5|0.0207631414|26.02660061|0.8973482475|0.1509392899|0.0760755714|0.1472541639|0.04002299685|0.03976637193|
|UVG_historical_HoneyBee_Jockey_2|B+1500 matched-FP32 baseline (pilot receiver + TCG)|0|2|0.004222366898|22.78535606|0.8288980685|0.241365451|0.1362398118|0.2464144781|0.07342122581|0.07402927683|
|UVG_historical_HoneyBee_Jockey_2|B+1500 matched-FP32 baseline (pilot receiver + TCG)|5|2|0.01374568986|26.10724026|0.9064923339|0.1438776583|0.09072726499|0.1485482034|0.05798408785|0.05878266961|
|UVG_historical_HoneyBee_Jockey_2|B+1500 matched-FP32 baseline (pilot receiver + TCG)|9|2|0.02780204114|27.64811776|0.9319553627|0.1091048394|0.07240281254|0.1088330173|0.05241462916|0.05288529437|
|UVG_historical_HoneyBee_Jockey_2|Generator Conv-LoRA rank4 (dev-selected checkpoint)|0|2|0.004222366898|22.83182943|0.8292263038|0.2389924519|0.1346709481|0.2439169077|0.07311331572|0.0736230702|
|UVG_historical_HoneyBee_Jockey_2|Generator Conv-LoRA rank4 (dev-selected checkpoint)|5|2|0.01374568986|26.1390617|0.9064161512|0.1415068085|0.08955914248|0.1458553043|0.05782316699|0.05857868522|
|UVG_historical_HoneyBee_Jockey_2|Generator Conv-LoRA rank4 (dev-selected checkpoint)|9|2|0.02780204114|27.66359304|0.9316760465|0.1070828935|0.07129840413|0.1067025335|0.05234614538|0.05278827117|


Development validation and patience records:


```json

{
  "status": "EARLY_STOPPED_PATIENCE",
  "anchor": 1.0,
  "bad_check_count": 4,
  "checks": [
    {
      "step": 250,
      "score": 1.0008561732255008,
      "F_q": {
        "0": 0.18627164785062805,
        "5": 0.112605423610177,
        "9": 0.09169396830238549
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 1,
      "checkpoint_sha256": "e721cbb0b56ecc5c508402f9e8387b19c2faae1a7656c188143402ec08d6a23f",
      "metrics_sha256": "04dffd8b32f3190796b3e8e5d266be5c2fd6af373f4d9248e59c2c24f0c29696",
      "time": 1793985882.30494
    },
    {
      "step": 500,
      "score": 1.0008270900049514,
      "F_q": {
        "0": 0.18627750197063303,
        "5": 0.11261262383223289,
        "9": 0.09167720805594905
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 2,
      "checkpoint_sha256": "8808b37d8b1b07d67aa623f239344082f95b292510eea2cc83cd33713c94eb7b",
      "metrics_sha256": "3558ac04137c698eedf879b5ceb8f825aa88505a5e4fe591277974ed9afee0bf",
      "time": 1793988138.8120608
    },
    {
      "step": 750,
      "score": 0.9994189148866107,
      "F_q": {
        "0": 0.186113330866358,
        "5": 0.1124104707712485,
        "9": 0.09153551757354142
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 3,
      "checkpoint_sha256": "f1dfaafd1295e6cf934644cba0133b36d96685be3ecdb44fbda35a6e2abf875c",
      "metrics_sha256": "1bcb85757557e10e09776289ea5ce1e17ee648bb47abf118eb6b34eda9aa2cbd",
      "time": 1793990311.2154555
    },
    {
      "step": 1000,
      "score": 0.999150799870855,
      "F_q": {
        "0": 0.18601368628959689,
        "5": 0.11243573281570293,
        "9": 0.09149032455059655
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 4,
      "checkpoint_sha256": "32da3fcc71070128a3f065faa78b56f61a69f715b6f53fbf39eeb26ab3dfe492",
      "metrics_sha256": "46843009095968afbbac69f94338554ba57951bb21ee5604ce250b278d9ade1f",
      "time": 1793992461.6945536
    }
  ],
  "raw_best": {
    "step": 1000,
    "score": 0.999150799870855,
    "checkpoint": "/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_generator_convlora_20261106/experiments/gvcrt_tcg_stage2_generator_convlora_20261106/checkpoints/step_1000.pt",
    "checkpoint_sha256": "32da3fcc71070128a3f065faa78b56f61a69f715b6f53fbf39eeb26ab3dfe492"
  },
  "last_validation_step": 1000,
  "initialization": {
    "score": 1.0,
    "metrics_sha256": "3dd5de284ec9139b53005fca8539ad8a6fc2a8fe915161220f64a6302c8808e2"
  },
  "baseline_F_q": {
    "0": 0.18586495117883065,
    "5": 0.11252980649800202,
    "9": 0.0917207126289473
  },
  "baseline_metrics_sha256": "b7b6161ed535b37252543a1dfe3cc48589c813818dd0b318a2dc2b79dc731a43"
}

```


Actual training data coverage (identity_visits are in training_coverage.json):


```json

{
  "uvg": {
    "samples": 2000,
    "unique_identities": 4,
    "unique_windows": 4,
    "unique_crops": 1472,
    "exact_combinations": 2000,
    "exact_repeats": 0,
    "QP_counts": {
      "6": 189,
      "7": 184,
      "4": 198,
      "3": 204,
      "1": 203,
      "8": 205,
      "0": 204,
      "2": 199,
      "9": 210,
      "5": 204
    },
    "ULong_pool": null,
    "ULong_identity_coverage": null,
    "new_independent_source_claim": false
  },
  "ulong": {
    "samples": 2000,
    "unique_identities": 2000,
    "unique_windows": 2000,
    "unique_crops": 2000,
    "exact_combinations": 2000,
    "exact_repeats": 0,
    "QP_counts": {
      "1": 192,
      "9": 218,
      "5": 223,
      "0": 192,
      "3": 218,
      "7": 190,
      "2": 193,
      "8": 189,
      "4": 186,
      "6": 199
    },
    "ULong_pool": 8470,
    "ULong_identity_coverage": 0.2361275088547816,
    "new_independent_source_claim": false
  }
}

```


Files: metrics.csv,development_metrics.csv,training_history.csv,rd_points_by_dataset.csv,stream_identity_checks.csv,checkpoints_manifest.json,selection.json,plots/datasets/,plots/per_video/,audits/. Publication status is separate in publication_status.json. Checkpoints and streams stay local. Shared-device operational timings are not a dedicated inference benchmark.
