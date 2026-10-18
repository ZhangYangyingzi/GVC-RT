# v6.35 fixed protocol

- Work type: audit + fixed real-codec evaluation. No N/P training in this run.
- Models: `original`, validated v6.34b `N_noP_BG_FD` cumulative step 500, and validated v6.34b `P_PBG_FD` cumulative step 500.
- Evaluation set: six exact U-Long clips from v6.34b; UVG holdout manifest entries 2/3 (HoneyBee/Jockey) and all five HEVC-B entries from v6.33. Sixty-four fixed frames, QP 0/5/9, IP=-1, native I-frame bypass, true RANS, independent decode.
- Metrics: LPIPS, DISTS, FloLPIPS; pooled dataset FID from Inception2048 features, float64 sample covariance (ddof=1). Actual bpp includes every bitstream byte divided by 64×valid width×height. `kbps_20fps` is a display conversion only.
- Point accounting: 117 logical points = 75 verified historical points + 42 new N/P UVG/HEVC encodes. U-Long 54 points reused from v6.34b; UVG/HEVC original 21 points reused from v6.24. The remaining 42 points are new real encodes.
- Original objective branch `A_refLoss_noP_500`: not run. `reference_objective_feasibility.json` records `BLOCKED_MISSING_REFERENCE_OBJECTIVE`, zero updates, and missing source details. No proxy or substitute objective was used.
- This is not a full 2×2 design. The comparisons are original vs complete trained pipeline, N vs P under the same complete v6.34b objective, and—only if the official target had been reproducible—N vs original objective. The last comparison is unavailable.

See `effective_config.json`, `source_hashes.json`, `audits/`, and task receipts for exact identities and hashes.
