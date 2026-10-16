# Execution record

B/C: 1000 effective updates each. Real bytes are from the pre-existing local non-official GLC rANS extension, including GOP and frame headers. Original author pretraining data unknown. Mixed U-Long/UVG training; independent data pools and UVG diagnostic results are separate. No checkpoint selection. All-frame and P-only FID use separate fixed frame pools. FloLPIPS covers all 63 adjacent transitions (each transition targets a P frame).

Evaluation: 1380/1380; independent points: 1350; diagnostic points: 30.

## Fixed q=28 numerical extract

| Dataset | Method | Actual mean bpp | LPIPS | DISTS | FloLPIPS | Pooled FID |
|---|---|---:|---:|---:|---:|---:|
| ulong | A_original | 0.00844052162 | 0.0930653044 | 0.077540247 | 0.0901580265 | 15.9868525 |
| ulong | B_rgb_P | 0.00843742465 | 0.0934070789 | 0.07685075 | 0.0906451895 | 16.0882582 |
| ulong | C_latent_P | 0.0110776774 | 0.0922140285 | 0.0730036263 | 0.0909700268 | 16.3840775 |
| uvg_holdout | A_original | 0.0160957694 | 0.0987334956 | 0.0748096732 | 0.100225036 | 21.5645581 |
| uvg_holdout | B_rgb_P | 0.0161210274 | 0.0991732417 | 0.0746599641 | 0.100785983 | 20.598494 |
| uvg_holdout | C_latent_P | 0.0176369297 | 0.100245459 | 0.0715224096 | 0.101745053 | 21.9632065 |
| uvg_validation | A_original | 0.0127701823 | 0.0953619126 | 0.072616661 | 0.0740752937 | 15.9803047 |
| uvg_validation | B_rgb_P | 0.0128652464 | 0.0944913054 | 0.0717428215 | 0.0733987432 | 16.5084258 |
| uvg_validation | C_latent_P | 0.0152002556 | 0.0940803284 | 0.0698322458 | 0.0740671225 | 18.221411 |
| hevc_b | A_original | 0.0120344449 | 0.15007135 | 0.0898740223 | 0.137815247 | 27.8469104 |
| hevc_b | B_rgb_P | 0.0120845751 | 0.150298771 | 0.0880705658 | 0.138271306 | 27.2721921 |
| hevc_b | C_latent_P | 0.0155343123 | 0.14740474 | 0.079841679 | 0.134109702 | 28.4561507 |
| mcljcv | A_original | 0.0117512197 | 0.112121684 | 0.0889164101 | 0.104522131 | 18.7283222 |
| mcljcv | B_rgb_P | 0.0117685205 | 0.112796118 | 0.0892997602 | 0.105187807 | 18.9579126 |
| mcljcv | C_latent_P | 0.0148278236 | 0.112130735 | 0.0834686636 | 0.106252405 | 19.3839277 |

All q values and raw records: results/per_video_all_q.csv and results/all_q_summary.csv. Common-rate coverage and deltas: results/equal_rate_summary.csv; fixed 100-point grids: results/common_rate_grid.csv. Per-dataset RD PNGs are in results/<dataset>/. MCL29 rows are a sensitivity subset with zero additional encodes, not an independent test pool. Source and stream checks: final_integrity.json. Fixed frame16/32/63 comparisons are in previews/<dataset>/.

Attempt/recovery audit references and hashes are listed in final_integrity.json. Failed attempts do not count as completed points; all final receipts passed unchanged strict independent-decode checks. The retained codec mismatch/retry record does not establish a root cause.
