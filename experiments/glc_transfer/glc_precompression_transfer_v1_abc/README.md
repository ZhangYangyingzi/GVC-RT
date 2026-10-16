# GLC precompression transfer v1 A/B/C

Training and evaluation complete; publication status is recorded separately on the server. B/C: 1000 effective updates each. A: unchanged original GLC. Mixed U-Long/UVG training. Fixed step1000, no checkpoint selection.

See REPORT.md, final_integrity.json and results/ for raw records, all-frame and P-only metrics, pooled FID, common-rate comparisons and RD PNGs. Rate axis: actual bpp from the pre-existing local non-official GLC rANS extension, including full stream headers and I frames. UVG validation is diagnostic; MCL29 is a sensitivity subset, not another independent dataset.

Final added sender weights are in final_P/. Original GLC weights, datasets, streams, feature arrays and full optimizer/RNG recovery checkpoints remain on the server. Restore commands are in RUNBOOK.md.
