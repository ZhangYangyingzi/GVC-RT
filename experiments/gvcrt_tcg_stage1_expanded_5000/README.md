# TCG Stage 1 expanded training

Fixed 5000 synchronized global updates on physical GPUs 4–7, effective batch 4. Stage 1 only, no router or gates.

20000 unique planned samples from 940 audited training identities. Histories detach; 48 warmup P frames and 15 supervised P frames per sample. Source initialization is pilot step 0, not pilot step 100.

Run runner.py using the gvc-rt Python environment with CUDA_VISIBLE_DEVICES=4,5,6,7 in a GPU-enabled execution context. It resumes from the latest atomic checkpoint. Each receipt follows its committed checkpoint; missing receipts are recovered from that checkpoint.

Main checkpoint is preregistered step 5000. Formal heldout evaluation starts only after training integrity passes. Model and large artifacts are local only and excluded from publication. See protocol.json, data_plan.json, split_audit.json and source_manifest.json.

Runtime acceleration: from committed global update 2456, resume using runner_accelerated.py. Original entrypoints remain immutable. The new runtime prefetches the next fixed-plan sample and verifies frozen state with exact GPU byte equality every update, plus full SHA-256 at fixed checkpoints and completion. Losses, precision, effective batch, optimizer state, RNG, cursor, source checkpoint and total budget are unchanged. Engineering equivalence and transition evidence are in audits/acceleration_equivalence_*.json and audits/runtime_transition.json.

Parallel GPU runtime: from committed global update 2624, use runner_parallel.py to resume latest. Each physical GPU 4–7 has one training rank and one separate frozen RAFT target-flow process. Only the four training ranks perform gradient all-reduce and optimizer updates; effective batch remains 4. Target flows are prepared one fixed-plan sample ahead, with exact row identities, model hashes and byte-checked packets. Actor outputs and losses were bit-exact in both-domain engineering audits; gradient replay uses the inherited v6.40 relative-L2 tolerance. See audits/runtime_transition_parallel.json and audits/parallel_flow_equivalence_*.json.
