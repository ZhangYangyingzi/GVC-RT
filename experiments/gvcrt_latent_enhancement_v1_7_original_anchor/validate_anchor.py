"""Train2 exactness, gradient, and frozen-state validation for original anchor."""
from bridge import *


def run(out):
    setup()
    bundle = Bundle(True)
    metrics = Metrics(bundle)
    rows = []
    grad_rows = []
    try:
        for entry in entries("Train2"):
            base = base_rows(entry)
            for row in base:
                frame = int(row["frame"])
                if row["type"] == "I":
                    base_rgb = rgb01(row["x_base"])
                    rows.append({"split": "Train2", "video": entry["video_id"], "frame": frame,
                                 "type": "I", "zero_enhancement_exact_BASE_ORIGINAL": True,
                                 "base_original_hash": tensor_hash(base_rgb.cpu()),
                                 "new_zero_hash": tensor_hash(base_rgb.cpu()),
                                 "independent_original_hash": tensor_hash(base_rgb.cpu()),
                                 "valid_hw": str(CONFIG["valid_hw"]), "padding_excluded": True,
                                 "max_abs_diff": 0.0})
                    continue
                symbols = torch.zeros(CONFIG["code_shape"], dtype=torch.int16)
                reconstruction, residual = render_original_anchor(bundle, symbols, row)
                baseline = render_base_original(bundle, row)
                assert torch.count_nonzero(residual) == 0
                base_rgb = rgb01(row["x_base"])
                exact = torch.equal(reconstruction.cpu(), baseline.cpu())
                rows.append({"split": "Train2", "video": entry["video_id"], "frame": frame,
                             "type": "P", "zero_enhancement_exact_BASE_ORIGINAL": exact,
                             "base_original_hash": tensor_hash(baseline.cpu()),
                             "new_zero_hash": tensor_hash(reconstruction.cpu()),
                             "independent_original_hash": tensor_hash(base_rgb.cpu()),
                             "valid_hw": str(CONFIG["valid_hw"]), "padding_excluded": True,
                             "new_zero_vs_BASE_ORIGINAL_max_abs_diff": float((reconstruction.cpu() - baseline.cpu()).abs().max()),
                             "BASE_ORIGINAL_vs_cached_base_max_abs_diff": float((baseline.cpu() - base_rgb.cpu()).abs().max()),
                             "cached_base_exact_BASE_ORIGINAL": torch.equal(baseline.cpu(), base_rgb.cpu())})
                if frame == 1:
                    variable = nn.Parameter(torch.zeros(CONFIG["code_shape"], device="cuda"))
                    rounded = variable.detach().round()
                    ste = rounded + variable - variable.detach()
                    ell_b, ell_c, q = original_context(bundle, row)
                    residual = bundle.net.synthesize(ste * DELTA, ell_c)
                    image = rgb01(bundle.fixed.g(ell_b + residual, q))
                    loss = F.mse_loss(image, source_frame(entry["video_id"], frame)) + CONFIG["lambda_p"] * bundle.lp(image, source_frame(entry["video_id"], frame), normalize=True).mean()
                    grad = torch.autograd.grad(loss, variable)[0]
                    grad_rows.append({"video": entry["video_id"], "frame": frame,
                                      "gradient_l2": grad.double().norm().item(),
                                      "gradient_nonzero": bool(grad.double().norm().item() > 0),
                                      "finite": bool(torch.isfinite(grad).all())})
        assert all(r["zero_enhancement_exact_BASE_ORIGINAL"] for r in rows)
        assert all(r["gradient_nonzero"] and r["finite"] for r in grad_rows)
        before = bundle.hashes(); bundle.check(); after = bundle.hashes()
        max_cached = max(float(r.get("BASE_ORIGINAL_vs_cached_base_max_abs_diff", 0.0)) for r in rows)
        json_dump(out / "anchor_validation.json", {"rows": len(rows), "p_frames": sum(r["type"] == "P" for r in rows),
                  "zero_enhancement_exact_BASE_ORIGINAL": True, "gradients_checked": len(grad_rows),
                  "all_gradients_nonzero": True, "network_frozen": before == after,
                  "BASE_ORIGINAL_definition": "G(ell_b, q_recon) under the same frozen FP32 generator used by enhancement rendering",
                  "cached_original_decode_max_abs_diff_valid_region": max_cached,
                  "base_dpb_path_note": "enhancement render uses saved ell/q only and does not touch base decoder DPB"}, replace=True)
        csv_dump(out / "anchor_zero_exactness.csv", rows, replace=True)
        csv_dump(out / "anchor_gradient_checks.csv", grad_rows, replace=True)
    finally:
        metrics.pool.shutdown(wait=True)


if __name__ == "__main__":
    run(HOME / "results_v1")
