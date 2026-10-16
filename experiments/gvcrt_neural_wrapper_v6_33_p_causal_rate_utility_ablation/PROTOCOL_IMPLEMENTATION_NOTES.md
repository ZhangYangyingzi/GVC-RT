# Effective backend seed

The immutable protocol records configured seed20261024. Canonical src/utils/common.py:set_torch_env resets Torch and NumPy to0 during codec construction. All three saved step0 CPU RNG states verify initial_seed=0. C/D P initialization hashes, all receiver initialization hashes and all quality network hashes agree. Python and the fixed shared sample/crop/QP plan retain20261024. No stochastic augmentation or nonzero receiver dropout is used. This behavior was recorded, not changed after observing quality. See seed_runtime_audit.json. The original protocol and its checkpoint signatures remain unchanged.

# Exact crop cache and recovery

The source PlanFrames implementation verified every full-frame RGB and cropped RGB hash. The cache producer additionally verified exact equality of reconstructed float32 model inputs for all1000 windows. Cache use changes only source I/O; model inputs, sample order, QPs, objective, optimizer and committed step budget are unchanged. Process migrations restore the last atomic model/optimizer/Torch CPU+CUDA RNG cursor; any uncommitted in-flight step may be replayed. Migration and pre-cache logs are retained.

# Evaluation resource recovery

The initial evaluation scheduler allowed up to three workers per GPU. Other processes increased GPU4 memory use during evaluation and three C evaluation workers encountered CUDA OOM. Their tracebacks and the original scheduler failure are preserved under logs/ and audits/. Completed atomic point receipts and persisted artifacts were retained. Recovery initially limits this experiment to one evaluation worker per GPU and checks at least8000MiB free at launch after over30 minutes of allocation observation (observed worker allocation approximately5.6GiB), with bounded OOM retries and a300-second GPU cooldown. A scheduler-only restart adopted live workers by exact process command and unique branch/shard, without restarting those workers. No training checkpoint, loss, sample, QP or metric protocol changed. Resource selection never uses quality. See evaluation/resource_recovery_protocol.json and evaluation/scheduler_adoption.json.

After more than two hours of failure-free real evaluation, a second worker per GPU is permitted only when at least18000MiB remains free before launch; maximum concurrency per GPU remains2, with approximately12GiB remaining after the observed additional-worker peak. The first-worker launch threshold remains8000MiB. This scheduler-only change uses memory, not quality. Fully completed shards are recognized from all expected atomic PASS receipts.

# Post-RD resource scheduling

All1840 RD points finished successfully. The initial final-latency launch threshold9000MiB was unavailable because other jobs occupied the allowed GPUs. No latency measurement had started. The observed same-architecture B1 latency peak was3309MiB, so the final latency threshold is6000MiB; channel audit permits one worker per GPU with at least7000MiB initially free. These are resource checks only. The exact inherited report32.pooled FID helper now runs independently on CPU while GPU measurements continue; all1840 source feature hashes and point receipt hashes are checked, and the final report verifies the CSV receipt instead of recomputing the identical aggregation. No metric formula or sample pool changed.

After five full128-channel ×3-intervention points, observed peak allocated memory was3688–3690MiB. The utility launch threshold was reduced to5500MiB with at most one utility worker per GPU. A scheduler-only restart adopts any live utility worker by exact command and pair ID; complete result receipts are skipped. Final D latency is reused only when its checkpoint SHA matches the fixed step1000 checkpoint. Neither completed numerical measurements nor the matched-rate pair plan are changed.

The5500MiB threshold briefly launched a GPU5 utility worker during scheduler recovery. Observed process-level GPU usage including allocator reservations was approximately5072MiB on GPU5 and6113MiB on GPU6, above the allocated-memory figure. The scheduler conservatively restored the7000MiB launch threshold and adopted both existing workers; completed outcomes remain subject to the same integrity checks. Existing final-D latency is checked before GPU selection, so scheduler recovery does not wait for or repeat an already completed benchmark. No utility computation or output was modified.

# Reporting environment

All20 full channel audit system points completed successfully. The gvc-rt inference environment has no matplotlib, so CPU reporting uses the existing/data1/anaconda3_new/anaconda_program/bin/python environment with NumPy/SciPy/Matplotlib. Model inference, metrics, persisted RANS and the inherited pooled FID aggregation remain the completed measurements from gvc-rt. No model experiment is rerun or substituted. CSV reporting formulas and plotting code are unchanged. See report_environment.json and the preserved dependency traceback.
