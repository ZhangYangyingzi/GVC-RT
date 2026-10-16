# v6.33 completion

B/C/D:1000 fixed updates each. A:0 updates. Actual-RANS evaluation:1840 points,46 videos,10 QPs,4 branches.

Channel audit:20 system points,128 decoded main-latent channels per point,3 interventions per channel. Rates are estimated installed-CDF main-y rates; actual per-channel RANS rate is unavailable.

Final D step1000 latency means (ms/frame):

|Mode|Encode|Decode|End-to-end|
|---|---:|---:|---:|
|No-P|52.624784260988235|108.29810753464699|160.92289179563522|
|With-P|180.46852439641953|115.11985696852207|295.5883813649416|

Full numerical records:REPORT.md, evaluation/equal_rate_summary.csv, utility/distribution_summary.csv, utility/distribution_changes.csv and latency/latency_summary.csv. Actual total RANS bitrate and estimated per-channel entropy-model rate remain separate.
