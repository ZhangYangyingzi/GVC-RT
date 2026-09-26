# V4 execution

Training uses GPU4 (beta_low), GPU5 (beta_mid), GPU6 (beta_high).
GPU7 evaluates Original and V3 baselines and the independent V4 step-zero models.
Each branch has an exclusive file lock and its own fresh AdamW optimizer.
All P/Bridge/G states originate from V3 stage_b/step_20000.pt.

The detached run_pipeline.py process waits for each branch to finish 10,000
updates, validates checkpoints on that branch's GPU, and selects only after
the complete 408-point validation grid is verified. That grid comprises
360 V4 points and 48 Original/V3 baseline points. Validation reads completed
shards across GPUs under a per-stage lock, so earlier step-zero work is reused.

After selection and initialization audits pass, the pipeline evaluates the
32 V4 test points on GPU4–7. The 96 existing Original/V2/V3 final-test points
are reused with bitstream size/hash, independent-decode, exact test key, and
V3 checkpoint checks. No final-test metric enters selection. Final audit
failure stops the pipeline and is recorded in pipeline_status.json.

Check current progress using pipeline_status.json, logs/pipeline.log,
training_logs/training_beta_*.csv, and parts/checkpoint_validation_*.csv.
Completion requires final_integrity.json with status PASS. stdout.log is
created only after finalization and contains the requested terminal fields.

To resume an interrupted pipeline, from the repository root in a GPU-enabled
environment:

```bash
/data1/anaconda3_new/anaconda_program/bin/python -u \
  experiments/gvcrt_neural_wrapper_v4_progressive_joint/run_pipeline.py
```

The pipeline lock prevents a second controller. Training locks prevent
duplicate branch processes; unfinished branches resume their latest complete
atomic checkpoint. Training logs are trimmed to the restored step.

The training/validation interpreter is the existing gvc-rt environment.
Plotting uses /data1/anaconda3_new/anaconda_program/bin/python, which already
provides matplotlib. No learning-rate scheduler or temporal-loss component
is introduced. Parameter, initialization, split, bitstream, decode, and final
audits are stored in this experiment directory.
