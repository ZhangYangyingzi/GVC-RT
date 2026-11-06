# Internal history gate joint execution snapshot

UTC: 2026-11-06T03:37:23Z

Status:RUNNING; stage:development_matched_FP32; committed joint updates:0.

Baseline B+1500 cumulative TCG6500. New seed20261106 Router M0=.2. Only Router+TCG171649 parameters train.

Gate previous_adapter output before concat; current_adapter retained; no outer residual gate. Router/TCG parameters and internals FP32; output returns to incoming dtype.

Fresh AdamW Router1e-4/TCG1e-5; effective batch4=2UVG+2ULong; source cursor1500; inherited loss and VJP4096.

Cap3000, validation250, patience4, min_delta.001; normalization uses independent matched-FP32 B1500.

Final integrity PASS is not claimed until every required point, plot and assertion has completed.
