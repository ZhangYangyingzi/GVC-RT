# V6.26 execution

Python: `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`.

Run in `/Huang_group/zyyz/Projects/GVC-RT`:

```bash
nohup /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_26_ms_illm_decoder_control/run_pipeline.py > experiments/gvcrt_neural_wrapper_v6_26_ms_illm_decoder_control/logs/pipeline.log 2>&1 < /dev/null &
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_26_ms_illm_decoder_control/run_pipeline.py --status
```

The actual launch record is `launch.json`. The persistent scheduler has an exclusive flock. Do not start another scheduler when its recorded PID/start identity remains active. An exited scheduler can be restarted with the same command. Training restores model, optimizers, independent discriminator RNG, Python/NumPy/Torch RNG and plan cursor. Verified points are skipped.

The source is v6.22 B1 step1000, not a trained v6.24/v6.25 branch. The v6.25 shared plan is copied byte for byte. Each branch supervises P49–63 after native I0 and detached causal P1–48. Full B1 gradients and unit adversarial gradients are separately averaged over all 15 target frames; reconstruction and adversarial gradients use unit scaling4096 and FP32 restoration. Lambda is computed as 0.05*norm(full B1 gradient)/(norm(unweighted adversarial gradient)+1e-12), without a fixed ceiling. One global clip and G update follow merging.

B uses conditional Projected hinge supervision with frozen pretrained EfficientNet-Lite1 and CCM/CSM. C uses pretrained frozen VQ labels and conditional U-Net local classification. Both receive detached pixel-unshuffled actual decoded P features at bridge input; the same condition is used for real and fake. Their discriminator architectures differ. Image gradients remain enabled.

Physical GPUs4–7 map individually to process CUDA0. Every started task reserves its estimated peak immediately, including model-loading tasks. Measurements and margins are recorded in the schedule. Other programs are preserved.

Main evaluation is800 points. Existing v6.25 FID decomposition reuses800 verified points; no historical training. Dataset pooled FID uses fixed Inception2048 features. Main equal-rate comparisons use PCHIP in the measured common ln(bpp) interval. FID component diagnostics use shared piecewise-linear interpolation weights and fixed equal-width low/mid/high thirds.

Stage publications are partial. Only the final integrity and verified final publication establish completion. Text shards are at most512KiB and restore with `restore_text_shards.py`; model weights, bitstreams, features and datasets stay on the server.

Wrapper is frozen. Only bridge and generator receive AdamW updates. B uses actual conditional Projected hinge; C uses official pretrained VQ targets and conditional U-Net CE. No lambda ceiling is used.
