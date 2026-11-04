# Evaluation-only process scheduling addendum

Research protocol remains online-development-earlystop-v3. The frozen training sources, model, optimizer, four logical RNG streams, data plans, loss, precision, learning rate, effective batch and early-stop/selection rules are unchanged.

The existing boundary pause mechanism stopped A at additional1028 / absolute6028 and B at additional528 / absolute5528. Each atomic latest and its committed receipt were verified and copied read-only to migrations/eval_parallel_v1/checkpoints/. No optimizer update is repeated or reset. Native_geometry_v1 and all previous protocol/source migrations remain intact.

The candidate evaluator uses two independent processes per physical GPU4–7, eight in total. Video assignment is disjoint; all checkpoints and QPs for each assigned video remain in the same worker. The fixed real-RANS numerical point function is reused without numerical edits. Each setup audit has its own worker directory, and native input decoding is bounded by four shared leases.

Before adoption, the independent engineering workload uses parent5000 and eight fixed alphabetically sorted development sequences already evaluated under v3, QP0/5/9,64 native frames:24 points. Run order is four processes, eight processes, then four processes again. No new model-selection or patience decision is created. Persisted streams, reconstructions, source/receiver hashes and per-frame/transition/reference CSVs must match exactly; metric absolute error must remain below the existing1e-6 reuse threshold. The eight-process run must be at least1.05x faster than the faster four-process reference; otherwise the original count is retained.

All immutable v3 files and SHA checks remain unchanged. New scheduling adapters and their generated function source are pinned in migrations/eval_parallel_v1/source_manifest.json. The adapter changes worker assignment, input leases and output-audit paths; it keeps the original encoding, independent decoding, quality computation and aggregation. Final acceptance composes the unchanged v3 integrity assertions with new worker/GPU/provenance checks. No final PASS is written before both layers complete.

Active entry after migration: launch_eval_parallel.py --resume, using the fixed gvc-rt Python, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=4,5,6,7. It resumes existing checkpoints and wraps the immutable v3 controller. It waits for inherited evaluation lifecycle locks and does not create duplicate old runners. Actual commands/PIDs and benchmark results are recorded under migrations/eval_parallel_v1/.

GitHub publication uses the existing isolated worktree, explicit file allowlist and normal fast-forward branch. Weights, videos, optical-flow/reconstruction caches and raw logs are excluded. The experiment remains RUNNING until training, final evaluations, final integrity and verified result publication finish.
