# Run and resume

From the repository root, run `experiments/gvcrt_generation_aware_quant_v2_skip_domain_trend_300/run.sh` in the existing gvc-rt environment. It performs preflight first, launches H10/H01/H11 on physical GPUs4/5/6, and runs preflight/evaluation serially on GPU7. Occupied GPUs queue. Do not launch another driver concurrently.

Resume with the same command. Each branch restores latest.pt optimizer, Python/NumPy/Torch/CUDA RNG and exact plan cursor; the saved last receipt repairs an interrupted CSV append. Already passing evaluation tasks are skipped. Resume checks training/runtime/config SHA to prevent silently mixing implementations. Keep the recorded dependency versions and source/checkpoint hashes. The source files listed in dependency_manifest.json and native RANS extension must be present; pretrained codec, generator, perceptual and flow weights and original dataset files remain server artifacts.

Per-group commands, after PASS preflight: `python train.py --branch H10 --gpu 4 --stop 300` (H01=5, H11=6). Individual evaluation: `python evaluate.py --branch H10 --step 300 --source UVal0 --qp 2 --gpu 7`. The driver checks GPU occupancy; individual commands require an idle assigned GPU.

Weights, complete streams, large caches and data stay under ignored server_artifacts or original source locations. Public delivery contains scripts, configuration, audits, small tables, plots and fixed previews. A/H00 training and eligible evaluation measurements are references to v1 A/B, explicitly labeled in provenance, never represented as fresh training/measurement.

Plotting/final tables use `/data1/anaconda3_new/anaconda_program/bin/python` (Matplotlib3.9.2), because the codec environment has no Matplotlib. Override `GVCRT_PLOT_PYTHON` with another plotting-capable Python if needed. GPU training/metrics/preview decoding always use the original gvc-rt Python; changing the plotting interpreter does not rerun training or change metrics. To regenerate tables and plots from complete artifacts, run the plotting Python on `finalize.py`.
