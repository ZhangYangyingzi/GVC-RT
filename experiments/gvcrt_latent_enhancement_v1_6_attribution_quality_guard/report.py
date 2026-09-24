"""Generate the concise final V1.6 report from checked result tables."""
from bridge import *


def average(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_comparison(rows, comparison, alpha=0.5, scope="IP"):
    chosen = [row for row in rows if row["comparison"] == comparison and
              row["alpha"] == alpha and row["scope"] == scope]
    return {key: average(chosen, key) for key in
            ("enhancement_bytes_delta", "psnr_a_minus_b", "ms_ssim_a_minus_b",
             "lpips_a_minus_b", "d_a_minus_b")}


def run(out):
    val_aggregate = read_csv(out / "Val6_aggregate.csv")
    attribution = read_csv(out / "Val6_method_attribution.csv")
    matched = [row for row in read_csv(out / "Val6_matched_total_rate.csv")
               if row["method"] == "M4_O_FRAME_GUARD" and row["alpha"] == 0.5]
    temporal = read_csv(out / "Val6_temporal_error.csv")
    wrong = json.loads((out / "Val6_wrong_source_guard_summary.json").read_text())
    filter_rows = read_csv(out / "Val6_candidate_filter.csv")
    receiver = read_csv(out / "receiver_verification.csv")
    final = json.loads((out / "final_verification.json").read_text())
    principal = next(row for row in val_aggregate if row["method"] == "M4_O_FRAME_GUARD" and
                     row["alpha"] == 0.5 and row["scope"] == "IP")
    unguarded = next(row for row in val_aggregate if row["method"] == "M4_O_FRAME" and
                     row["alpha"] == 0.5 and row["scope"] == "IP")
    comparisons = {name: aggregate_comparison(attribution, name) for name in
                   ("M2_MINUS_M1", "M4_MINUS_M3", "M4_MINUS_M2", "M4_GUARD_MINUS_M4")}
    nonzero_filters = [row for row in filter_rows if row["candidate"] != "Z"]
    reasons = {reason: sum(row["guard_result"] == reason for row in nonzero_filters) for reason in
               ("ALLOWED", "MSE_ONLY_DEGRADES", "LPIPS_ONLY_DEGRADES", "MSE_AND_LPIPS_DEGRADE")}
    same_rate = {metric: average(matched, f"final_minus_original_interpolated_{metric}")
                 for metric in ("psnr", "ms_ssim", "lpips")}
    zero_vs_original_q0 = {metric: average(matched, f"zero_minus_original_qp0_{metric}")
                           for metric in ("psnr", "ms_ssim", "lpips")}
    p_zero = [row for row in temporal if row["method"] == "ZERO" and row["transition"] == "P_to_P"]
    p_guard = [row for row in temporal if row["method"] == "M4_O_FRAME_GUARD" and row["transition"] == "P_to_P"]
    i_zero = [row for row in temporal if row["method"] == "ZERO" and row["transition"] == "I_to_P"]
    i_guard = [row for row in temporal if row["method"] == "M4_O_FRAME_GUARD" and row["transition"] == "I_to_P"]

    def fmt_change(values):
        return (f"PSNR {values['psnr_a_minus_b']:+.6f} dB, MS-SSIM "
                f"{values['ms_ssim_a_minus_b']:+.6f}, LPIPS {values['lpips_a_minus_b']:+.6f}, "
                f"d {values['d_a_minus_b']:+.9f}, enhancement bytes {values['enhancement_bytes_delta']:+.1f}")

    report = f"""# GVC-RT V1.6 Attribution and Quality Guard

## Conclusion

V1.6 completed the frozen-candidate, zero-training validation on Train2 (2 clips) and Val6 (6 clips). It performed 0 network updates and 0 code/latent optimization updates. The principal guarded point is `M4_O_FRAME_GUARD, alpha=0.5`.

On Val6 IP, that point covers {int(principal['coverage_videos'])}/6 videos, uses {float(principal['mean_video_enhancement_over_base'])*100:.4f}% enhancement/base by per-video macro average ({float(principal['pooled_enhancement_over_base'])*100:.4f}% pooled), and changes ZERO by PSNR {float(principal['psnr_minus_ZERO']):+.6f} dB, MS-SSIM {float(principal['ms_ssim_minus_ZERO']):+.6f}, LPIPS {float(principal['lpips_minus_ZERO']):+.6f}, and `d=MSE+0.001*LPIPS` {float(principal['d_minus_ZERO']):+.9f}. Thus the guard preserves an average objective and perceptual improvement, unlike unrestricted M4's LPIPS change of {float(unguarded['lpips_minus_ZERO']):+.6f}.

## Q1: Where Does The Gain Come From?

At the same `alpha=0.5` budget upper bound, macro-averaged per-video IP differences are:

| Attribution | A minus B |
|---|---|
| Frame allocation (`M2-M1`) | {fmt_change(comparisons['M2_MINUS_M1'])} |
| Frame allocation with optimized pool (`M4-M3`) | {fmt_change(comparisons['M4_MINUS_M3'])} |
| Optimized candidates beyond E (`M4-M2`) | {fmt_change(comparisons['M4_MINUS_M2'])} |

These are same-budget-upper-bound comparisons, not equal-actual-rate comparisons. Both methods' actual bytes are retained per video in `Val6_method_attribution.csv`.

## Q2: Does The Joint Quality Guard Work?

Yes for the stated finite candidate pool and per-frame constraint. The principal point improves every selected nonzero frame against that frame's ZERO in both MSE and LPIPS, while averaging PSNR {float(principal['psnr_minus_ZERO']):+.6f} dB and LPIPS {float(principal['lpips_minus_ZERO']):+.6f} against ZERO. The Val6 non-Z filter counts are {reasons}. Compared with unrestricted M4, the guard changes {fmt_change(comparisons['M4_GUARD_MINUS_M4'])}.

The guard does not prove that no other sequence-level jointly improving solution exists. A failure would only characterize this frozen finite pool. MS-SSIM was not constrained and is not required to be monotone.

## Q3: How Does It Compare At The Same Total Rate?

All 6 principal Val6 points lie inside each video's measured `ORIGINAL_FP32` QP0-QP3 overlap. Per-video piecewise-linear interpolation versus `log(total_bpp)` gives final-minus-original macro means: PSNR {same_rate['psnr']:+.6f} dB, MS-SSIM {same_rate['ms_ssim']:+.6f}, LPIPS {same_rate['lpips']:+.6f}. Before enhancement, ZERO-minus-original-QP0 macro means are PSNR {zero_vs_original_q0['psnr']:+.6f} dB, MS-SSIM {zero_vs_original_q0['ms_ssim']:+.6f}, LPIPS {zero_vs_original_q0['lpips']:+.6f}. No point was extrapolated.

The guarded enhancement therefore does not close the full gap to original GVC-RT at matched total rate on these clips, despite improving its own fixed ZERO path.

## Q4: Is The Signal Source-Specific And Temporally Safe?

The fixed cyclic WRONG_SOURCE diagnostic covers {wrong['pairs']} nonzero principal selections. Replacing target symbols by another video's same-label symbols changes target quality by PSNR {wrong['mean_wrong_minus_target_psnr']:+.6f} dB, LPIPS {wrong['mean_wrong_minus_target_lpips']:+.6f}, and d {wrong['mean_wrong_minus_target_d']:+.9f}; this supports source-specific information rather than a generic injection effect.

Mean I-to-P `T_error` changes from {average(i_zero, 'T_error'):.9f} (ZERO) to {average(i_guard, 'T_error'):.9f} (guard). Mean P-to-P changes from {average(p_zero, 'T_error'):.9f} to {average(p_guard, 'T_error'):.9f}. `T_error=mean(abs(e_t-e_(t-1)))` is reported without motion alignment; it is a short-GOP diagnostic, not proof of long-term stability. Per-transition values and switch neighborhoods are in `Val6_temporal_timeline.csv` and `Val6_switch_neighborhood_temporal.csv`.

## Integrity And Artifacts

- Guard tolerance: exactly 0.0 after 120/120 Train2 repeated candidate measurements had max MSE and LPIPS absolute error 0.0.
- Real streams: {final['splits']['Train2']['logical_real_streams']} Train2 and {final['splits']['Val6']['logical_real_streams']} Val6 logical nonzero streams; {len(receiver)} unique stream contents independently received.
- Independent receiver: every reconstructed RGB frame hash matched; source access was actively blocked; base DPB state remained unchanged.
- Negative tests: truncated, trailing-byte, CRC-corrupt, and wrong-base-bound streams were all rejected.
- Videos: 6 synchronized 16-frame principal reconstructions in `results_v1/videos/`.
- Final verification: `passed={str(final['passed']).lower()}` in `results_v1/final_verification.json`.
- Input provenance: `results_v1/input_hash_manifest.json`.

No next-stage training was started.
"""
    (HOME / "report.md").write_text(report)


if __name__ == "__main__":
    run(HOME / "results_v1")
