# Stage2 Router-only execution report

Snapshot UTC: 2026-11-05T15:55:09Z

Integrity: PASS. Committed Router updates: 1000. Stop reason: EARLY_STOPPED_PATIENCE.

Best trained Router: step250; normalized development FloLPIPS=0.999987007223. Final selection: trained_router.

Only Router was trained. Baseline is B+1500 at cumulative TCG step6500. Receiver and all frozen components are recorded in baseline_provenance.json.

Protocol: protocol.json. Training loss is the actual inherited Stage1 implementation; not claimed identical to the original paper training loss.

Final testing:135 fresh Router RANS encodes with new independent decoders. Original135 and B+1500 135 reused only after hash and aggregation audits; provenance in audits/reuse_*.json.

Historical subsets remain historical references. UVG contains only HoneyBee and Jockey. Equal video weighting applies separately within each dataset/split and QP.

Mask denotes historical residual suppression weight. PWC sampler retains the documented existing pixel-center/W−1 normalization behavior. No strict S2VC reproduction claim.

Seven quality metrics and actual bitrate are tabulated below; no rate interpolation or BD-rate is computed.

|Dataset|Model|QP|bpp|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|WarpGT|WarpRecon|

|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

|ULong_new_holdout_32|GVC-RT original (independent)|0|0.0019800464|24.5079713|0.82296729|0.217382966|0.133372425|0.223901937|0.0469298216|0.0466315898|

|ULong_new_holdout_32|GVC-RT original (independent)|5|0.0074157583|26.8599783|0.895593223|0.13714097|0.09077596|0.134260242|0.0372747714|0.0374727122|

|ULong_new_holdout_32|GVC-RT original (independent)|9|0.0168124333|27.7604843|0.919168714|0.112119968|0.0765436236|0.107572588|0.0342558139|0.0342915606|

|ULong_new_holdout_32|TCG B+1500|0|0.0019800464|23.8832169|0.816880741|0.198165125|0.120528524|0.202161749|0.0509421973|0.0506310954|

|ULong_new_holdout_32|TCG B+1500|5|0.0074157583|26.64189|0.895285488|0.118728627|0.073552023|0.116109083|0.0385536546|0.0388578969|

|ULong_new_holdout_32|TCG B+1500|9|0.0168124333|27.7081067|0.919480225|0.0969179812|0.0608107299|0.0932270127|0.0348603051|0.0349638227|

|ULong_new_holdout_32|TCG B+1500 + Router (best trained)|0|0.0019800464|23.8832001|0.816880579|0.198165191|0.120529013|0.202164525|0.0509423712|0.0506311462|

|ULong_new_holdout_32|TCG B+1500 + Router (best trained)|5|0.0074157583|26.6417974|0.895285201|0.118729144|0.0735519354|0.116107616|0.0385540684|0.0388593808|

|ULong_new_holdout_32|TCG B+1500 + Router (best trained)|9|0.0168124333|27.7081677|0.919480333|0.0969180854|0.06081078|0.0932305712|0.0348601306|0.0349638747|

|ULong_historical_6|GVC-RT original (independent)|0|0.00205281073|23.2970582|0.820566136|0.214890967|0.136963132|0.232919291|0.0490211404|0.0492341112|

|ULong_historical_6|GVC-RT original (independent)|5|0.00771265352|25.4874236|0.894928052|0.137271282|0.0968528084|0.138107445|0.0388442829|0.0387381665|

|ULong_historical_6|GVC-RT original (independent)|9|0.0172255879|26.2876|0.917199035|0.113217605|0.0840491907|0.110031518|0.0359508739|0.035713847|

|ULong_historical_6|TCG B+1500|0|0.00205281073|22.8303273|0.816117319|0.196768914|0.123892893|0.20434943|0.0522564103|0.0525350829|

|ULong_historical_6|TCG B+1500|5|0.00771265352|25.3765248|0.894970449|0.121536879|0.0808246094|0.122199966|0.0397780645|0.0396993|

|ULong_historical_6|TCG B+1500|9|0.0172255879|26.3069755|0.917827843|0.0998566331|0.0699937112|0.0969018659|0.0362973739|0.0360844147|

|ULong_historical_6|TCG B+1500 + Router (best trained)|0|0.00205281073|22.8303514|0.816117323|0.196768671|0.123893061|0.204354836|0.0522558118|0.0525346377|

|ULong_historical_6|TCG B+1500 + Router (best trained)|5|0.00771265352|25.3765491|0.894969714|0.121537358|0.0808253662|0.122192818|0.0397781794|0.0397004357|

|ULong_historical_6|TCG B+1500 + Router (best trained)|9|0.0172255879|26.3068778|0.917827405|0.0998568153|0.0699941817|0.0969121716|0.0362976996|0.0360844526|

|HEVC_B_5|GVC-RT original (independent)|0|0.002624144|22.8016137|0.783828979|0.270021323|0.11840028|0.266129301|0.05238066|0.0519598014|

|HEVC_B_5|GVC-RT original (independent)|5|0.00941855951|25.1810506|0.873671633|0.183004617|0.0904026654|0.186738729|0.0423351994|0.041958055|

|HEVC_B_5|GVC-RT original (independent)|9|0.0207631414|26.1085333|0.900527209|0.156130134|0.0801440394|0.153709746|0.0394301922|0.039237103|

|HEVC_B_5|TCG B+1500|0|0.002624144|22.3843222|0.7773604|0.263877155|0.115807141|0.253623359|0.0561706882|0.0556661123|

|HEVC_B_5|TCG B+1500|5|0.00941855951|25.005784|0.870729939|0.1806798|0.0874820234|0.182497627|0.0436183329|0.0432280948|

|HEVC_B_5|TCG B+1500|9|0.0207631414|26.0331181|0.897722922|0.153086493|0.0768091001|0.149662901|0.0400136972|0.0397931182|

|HEVC_B_5|TCG B+1500 + Router (best trained)|0|0.002624144|22.3844509|0.777361212|0.263876363|0.115806683|0.253618393|0.0561692583|0.0556649426|

|HEVC_B_5|TCG B+1500 + Router (best trained)|5|0.00941855951|25.0056514|0.870729649|0.180680247|0.0874828473|0.182502842|0.0436189112|0.0432302548|

|HEVC_B_5|TCG B+1500 + Router (best trained)|9|0.0207631414|26.0331823|0.897723006|0.153086432|0.0768091377|0.149667452|0.0400139945|0.0397940867|

|UVG_historical_HoneyBee_Jockey_2|GVC-RT original (independent)|0|0.0042223669|23.5677594|0.83648393|0.231672521|0.13525085|0.239978068|0.0680238054|0.0688819751|

|UVG_historical_HoneyBee_Jockey_2|GVC-RT original (independent)|5|0.0137456899|26.5536573|0.909266544|0.136248907|0.0922976001|0.138041141|0.0561146274|0.0570280169|

|UVG_historical_HoneyBee_Jockey_2|GVC-RT original (independent)|9|0.0278020411|27.8889641|0.932907783|0.108828799|0.0767051601|0.108196185|0.0516865635|0.0520204113|

|UVG_historical_HoneyBee_Jockey_2|TCG B+1500|0|0.0042223669|22.7850909|0.828896755|0.24136959|0.13624334|0.246421851|0.0734233603|0.0740363193|

|UVG_historical_HoneyBee_Jockey_2|TCG B+1500|5|0.0137456899|26.1071011|0.906490733|0.143883092|0.0907311873|0.148569107|0.0579848845|0.0587794785|

|UVG_historical_HoneyBee_Jockey_2|TCG B+1500|9|0.0278020411|27.6479947|0.931954596|0.109108027|0.0724055259|0.108803865|0.0524154127|0.0528878061|

|UVG_historical_HoneyBee_Jockey_2|TCG B+1500 + Router (best trained)|0|0.0042223669|22.7849983|0.828896289|0.241368159|0.136243675|0.246417964|0.0734237404|0.0740366895|

|UVG_historical_HoneyBee_Jockey_2|TCG B+1500 + Router (best trained)|5|0.0137456899|26.1069044|0.906489951|0.143885366|0.0907331891|0.148549253|0.0579852099|0.0587784948|

|UVG_historical_HoneyBee_Jockey_2|TCG B+1500 + Router (best trained)|9|0.0278020411|27.6481038|0.931954788|0.109107835|0.072404881|0.108825996|0.0524147763|0.0528769413|


Development checks:


```json

{
  "status": "EARLY_STOPPED_PATIENCE",
  "anchor": 1.0,
  "bad_check_count": 4,
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
    },
    {
      "step": 500,
      "score": 1.0000109556202175,
      "F_q": {
        "0": 0.18585584748218517,
        "5": 0.11253824594360406,
        "9": 0.09172152742440443
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 2,
      "metrics_sha256": "d1b209a47f1875d83919531a40e976b78af71739601956f448f7875cad64f8b2",
      "checkpoint_sha256": "2e0f9f2eb1ea3c1130841e8b6f991ac0046a7bb868e68097f6ad4d3afed0e3bb",
      "time": 1793883747.8190708
    },
    {
      "step": 750,
      "score": 1.000003689495904,
      "F_q": {
        "0": 0.18585411690375103,
        "5": 0.11253771411036954,
        "9": 0.091720815564114
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 3,
      "metrics_sha256": "a534f0553c35a328b9eba81a0d6a2187fd93ed7db6e0a5866fc977dae85816b4",
      "checkpoint_sha256": "3e7b53e35cb6a7051cefbe1bae94a4cd7cbef291ee6d623741a383b2d21c0a6c",
      "time": 1793887344.6419168
    },
    {
      "step": 1000,
      "score": 0.9999898960476327,
      "F_q": {
        "0": 0.1858553721547304,
        "5": 0.11253436301772055,
        "9": 0.0917191319731212
      },
      "anchor_before": 1.0,
      "anchor_after": 1.0,
      "substantive_improvement": false,
      "bad_check_count": 4,
      "metrics_sha256": "871c2158c3c4583c76eeec90fcbcdb91e55b1ee92b68721f63ccc159cc249842",
      "checkpoint_sha256": "55e22d1202d8d0affd7ad815f4ca1884b204f35acabc38917dd00f5a1bad314a",
      "time": 1793890909.4878335
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
  "last_validation_step": 1000
}

```


Coverage:


```json

{
  "uvg": {
    "samples": 2000,
    "distinct_video_identities": 4
  },
  "ulong": {
    "samples": 2000,
    "distinct_video_identities": 2000
  }
}

```


Limitations and execution records: initialized enabled0 is measured, not identical to bypass; repeated historical samples are not new independent videos. Initial engineering harness call/dtype fixes and all retries remain in logs and repair audits. Latency timing, if shared GPUs are active, is not presented as a dedicated speed measurement.
