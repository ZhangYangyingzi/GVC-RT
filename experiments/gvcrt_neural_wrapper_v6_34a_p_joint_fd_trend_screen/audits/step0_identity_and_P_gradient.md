# Reused v6.34 training audit

- Source `audits/sanity.json`: PASS for both branches; common initialization and step-0 path were checked.
- Step-0 receiver deployment hashes in both branch indexes equal the original receiver hashes in the evaluation config.
- Step-1 training receipts are available for both branches. The P branch receipt records the actual wrapper gradient/update, establishing that the historical P run was not silently frozen. Checkpoint/module/optimizer/EMA metadata and hashes are captured in `budget_override.json`.
- Both step-100 and step-250 full training checkpoints load successfully with wrapper/bridge/generator, optimizer, crop/global RNG, buffers, and EMA state. Index optimizer steps equal 100 and 250; step-250 EMA counters sum to 250 per branch.
- Step-0 N/P are represented by a shared original evaluation alias. The checkpoint hashes remain source references; no checkpoint is copied or rewritten.
