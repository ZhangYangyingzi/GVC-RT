# Independent dataset distribution audit

Only statistics and Original GVC-RT profiling. No training, tuning, test-guided sample selection, or changes to V4/V5/V6.

The user explicitly approved seven real Vimeo frames instead of the requested32 because the local septuplets have only7 frames. No looping, frame duplication, interpolation, or cross-sequence concatenation is used. U-Long fresh codec profiles use32 frames; frozen UVG64 and full VIRAT Original results are validated and reused. All frame lengths and comparison FPS are retained in the tables; bpp is the primary rate descriptor.

Source cohort:1024 exact existing U-Long training videos,1024 seeded disjoint unused U-Long videos,2048 seeded official Vimeo training sequences,7 frozen UVG sequences,8 unique VIRAT segments. Source-level VIRAT is counted once using the720p source;720p and480p codec records remain separate. Primary coverage uses one VIRAT720 codec descriptor per content to avoid double counting.

Two source views are kept separate. Standardized content uses long side512 with preserved aspect ratio and up to32 ordered frames at approximately <=30fps. Actual codec-input statistics preserve native geometry and frozen ordering. The unchanged wrapper's actual four-frame random256 crop pipeline is documented separately rather than presented as full-frame codec input.

Camera estimation failures are explicit missing values, not identity/zero replacements. Coverage only uses complete-feature codec-profiled subsets, so “U-Long train1024” and “Vimeo train2048” support labels do not mean every one of the1024/2048 sources has codec features. Exact complete support counts are saved. Scaler/PCA are fit only on complete candidate U-Long and Vimeo rows, never on UVG/VIRAT. Neighbors are diagnostics only and never added to training.

PWC-Net and ResNet50 use already-cached local weights; no download. One heavy worker runs on each of physical GPUs4,5,6,7. CPU source statistics use12 processes. Both GPU phases use deterministic workload balancing and separate per-sample/per-QP output files. Existing successful raw outputs are reused on restart.

From repository root:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/dataset_distribution_audit_v1/run_audit.py --launch
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/dataset_distribution_audit_v1/run_audit.py --status
tail -f experiments/dataset_distribution_audit_v1/logs/pipeline.log
```

Only `final_integrity.json: PASS` means all stages completed. A failure record identifies the exact log; no results are silently fabricated to complete a table.
