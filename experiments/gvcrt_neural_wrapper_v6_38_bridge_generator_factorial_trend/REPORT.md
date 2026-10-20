# Results summary

Status: PASS. Updates O=0, B=100, G=100, BG=100. RD24/24, six shared real-RANS streams, two heldout HEVC-B videos, QP0/5/9,64 frames,63 flow transitions. Failed attempts: 1.

All methods have identical full stream bytes/SHA and decoded y/f/q hashes for each video/QP. Training module freezes and exact100 committed optimizer steps pass.

|Comparison|Metric|Mean absolute delta|Mean relative delta|Wins/6|
|---|---|---:|---:|---:|
|B/O|LPIPS|0.0146418025|0.0919296|0|
|B/O|DISTS|0.00482672888|0.058068|1|
|B/O|FloLPIPS|0.0101173753|0.064261|0|
|B/O|PSNR|-0.659638078|-0.0273372|0|
|B/O|L1|0.00392609403|0.0940887|0|
|G/O|LPIPS|-0.00527145669|-0.0324599|6|
|G/O|DISTS|-0.00526477629|-0.0543156|6|
|G/O|FloLPIPS|-0.0058146056|-0.0364063|6|
|G/O|PSNR|-0.0919959596|-0.0037233|0|
|G/O|L1|0.000546668142|0.0138185|0|
|BG/O|LPIPS|0.00218132113|0.0151259|1|
|BG/O|DISTS|-0.00455146019|-0.0449994|4|
|BG/O|FloLPIPS|0.000437189997|0.00614455|3|
|BG/O|PSNR|-0.496462986|-0.0202188|0|
|BG/O|L1|0.00306308216|0.0765094|0|
|BG/B|LPIPS|-0.0124604814|-0.0689255|6|
|BG/B|DISTS|-0.00937818907|-0.0969012|6|
|BG/B|FloLPIPS|-0.00968018533|-0.0542154|5|
|BG/B|PSNR|0.163175092|0.00729739|5|
|BG/B|L1|-0.000863011868|-0.0162689|5|
|BG/G|LPIPS|0.00745277782|0.0496307|1|
|BG/G|DISTS|0.000713316103|0.0094604|2|
|BG/G|FloLPIPS|0.0062517956|0.044163|1|
|BG/G|PSNR|-0.404467027|-0.0165781|0|
|BG/G|L1|0.00251641402|0.0615965|0|

Delta means first condition minus reference. Lower LPIPS/DISTS/FloLPIPS/L1 and higher PSNR are better. Averages give each of six video/QP anchors equal weight. No statistical inference is made.

This is a single-seed,100-update trend screen on two training identities (1 historical U-Long row,99 historical Beauty rows), not evidence of convergence, significance or generalization. No extra branch, alpha, seed, checkpoint selection or training extension was run.

See `results/evaluation.csv` for24 individual points, `results/paired_differences.csv` for150 metric differences, `branches/*/receipts/` for all committed update logs, and `logs/task_events.jsonl` for actual GPU and exit records.
