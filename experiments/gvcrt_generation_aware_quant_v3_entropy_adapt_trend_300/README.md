# Generation-aware quantization v3: entropy adaptation trend

Fixed logical controls N0/N1/Q0/Q1, only N1 and Q1 newly trained300 updates each. See PROTOCOL.md, PROBABILITY_PARAMETER_SCOPE.md and effective_config.json. N1 is a no-new-H original-quantization-structure adaptation baseline, not a fixed-local-step baseline.

## Server run and recovery

Use `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B` with the indexed server artifacts. `prepare.py` verifies frozen v2 data/plan, dependencies, weights and indexed legacy checkpoints. `smoke.py` runs required initial/gradient/real-CDF checks on GPU6 and accepts the first2 budget-counted updates only after PASS. `run.py` waits for that gate, verifies/imports controls, resumes N1/Q1 to300 on GPUs4/5, evaluates fixed snapshots on GPU7 (released GPUs4/5 share the queue after training) and adds only rate-bracketed limited supplemental QPs. `reuse_decode_audit.py` supplies readonly GPU6 parity checks where old N0 tasks did not retain explicit P-latent/qmap flags. Old recorded values and metadata are never overwritten.

Checkpoint latest is committed atomically every update with complete Phi/G/H, optimizer,RNG,cursor and last receipt. Resume with run.py. Snapshot0/100/300 only; no budget extension. The first2 updates are not repeated. Explicit PYTHONPATH in the runner selects the current experiment and the verified original server source checkout. Do not run historical scripts as new tasks.

`audit_behavior_symbols.py` reads the existing three multiplier-check streams after a training GPU is released and retains integer-symbol hashes/counts; this is a read-only audit with no new encoding or optimizer updates.

After computation, `finish.py` validates, summarizes all metrics, plots and publishes only this experiment to `experiment/gvcrt-quant-v3-entropy-adapt-trend-300`, without force push. Large weights, streams, caches and console logs remain on the server. Git contains tables, code, configuration, source/weight/checkpoint/delivery indexes, figures and fixed previews.

Use `finish.py --prepare-only` to complete local audits, tables, plots and the report before publication. GPU evaluation must run with host NVIDIA device access; a sandbox without `/dev/nvidia*` cannot execute it. Resuming completed training resets the status to `EVALUATION` and retains the 300-update checkpoints.

Required outputs: initialization_manifest.json,reuse_provenance.json,preflight.json,cdf_consistency_audit.json,checkpoint_index.json,training_log.csv,evaluation.csv,estimated_vs_actual_P_rate.csv,observed_dominance.csv,matched_rate_comparisons.csv,REPORT.md,figures/,completion.json,publication_status.json.

Limits:single seed,300-step trend,8 fixed U-Long sources,original window/QP coupling,two validation clips and two descriptive test clips; historical UVG exposure is not excluded. No FID,formal BD-rate,convergence or significance claim. Probability-model adaptation gains are reported separately from gains attributable to adding H under matched Phi/G scope.
