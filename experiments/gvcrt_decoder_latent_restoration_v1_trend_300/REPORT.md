# Decoder latent restoration v1: execution record

Status: BLOCKED before GPU preflight and training. Updated 2026-10-27, Asia/Shanghai. No research conclusion or quality result is claimed.

| Branch | Effective optimizer updates | Status |
| --- | ---: | --- |
| A | 0 | Fixed baseline; no training prescribed |
| G | 0 / 300 | Not started |
| R | 0 / 300 | Not started |
| RQ | 0 / 300 | Not started |

## Verified existing work

The experiment already had an isolated worktree at `/Huang_group/zyyz/home_dir/tmp/gvcrt-decoder-latent-restoration-v1`, on branch `experiment/gvcrt-decoder-latent-restoration-v1-trend-300`, based on `509ad3a8b618fd5f09fa097f6cad58d05f893047`. Its experiment directory was entirely untracked before this continuation. The original repository was not modified. No applicable AGENTS.md was found in the worktree or the checked ancestor paths.

The specified A generator checkpoint SHA256 is `17a7635ae6e4f74922b5bfe368a790bf2f48b3a7bff2b2f516c4dea24148404f`. All files in `weight_manifest.json` passed size and SHA verification. The loading code takes the bridge from `G_basic_100.pt`, then replaces its generator with the specified A checkpoint, matching the referenced v5 A loading sequence. Actual loaded GPU component hashes remain pending; file verification is not a substitute for that check.

The v5 train plan SHA256 is `b9f7c050449be8fcf4751b6559a280875cd4378fe59ad7386a36e3a0639ee2b9`. Both the 300-row plan and evaluation sources are byte-identical to the current referenced v5 files. Training source SHA checks passed. The 8 existing A native streams are present locally, with the 16-byte v5 experiment header removed. All native stream SHA, sizes, 64 frame headers, I/P pattern, shifted QP sequence, and exact byte consumption passed a CPU parser check. This does not establish entropy-decoder or reference-trajectory equivalence.

`offline_audit.json` records CPU checks. The Adapter has 126,994 nominal parameters. Synthetic CPU inputs verified zero initial residual and gradient connectivity through a frozen differentiable surrogate, including earlier layers after changing the output weights. The temporary diagnostic model was discarded without an optimizer step. Actual decoded-frame identity and gradients through the actual A generator remain mandatory in `preflight.py`.

## Continuation changes

- Corrected the stale `common.py` snapshot hash and documented its existing pinned checkpoint root.
- Explicitly pinned the `core` module loader to the worktree's tracked script and registered repository Python dependencies with SHA values. `src/utils/video_writer.py` differs from the original repository; it is not imported by this experiment's custom quality writer.
- Saved training CSV rows inside each atomic checkpoint, allowing recovery after interruption between checkpoint and CSV writes.
- Isolated LPIPS initialization from the saved torch training RNG on resume.
- Tightened training cache reuse to require matching code, plan and weight provenance.
- Added reference RGB, reference POC and unchanged I-frame output checks to independent decoding/finalization. Expanded the GPU preflight entrypoint to independently decode all 16 native frames, compare zero-residual branch outputs and disabled restoration, verify unchanged reference feature/RGB/POC and y_hat, and match P12–P15 inputs and quantization maps back to the shared cache. This entrypoint passed syntax checks but has not run on GPU.
- Added a resumable pipeline entrypoint that schedules subprocesses only on idle physical GPUs 4–7, runs the required preflight, uses smoke updates within the 300-update budget, and keeps decoding separate from GT-based quality evaluation. It does not publish or write COMPLETE.json.

## Execution blockers

The existing `gvc-rt` environment reports torch `2.6.0+cu124`, but `torch.cuda.is_available()` is false inside the current sandbox. No `/dev/nvidia*` devices are exposed, and `nvidia-smi` cannot communicate with the driver. An existing resource-wait receipt at `2026-10-27T04:17:30Z` (12:17:30 Asia/Shanghai) reports 100% utilization on physical GPUs 4–7 and no driver-query failures. The external waiting process itself and live occupancy could not be inspected from this sandbox. No second GPU launcher was started here.

A requested sandbox-external GPU/process query was not executed because automatic approval review failed with HTTP `502 Bad Gateway`. This was an approval-service failure, not a finding that the action was unsafe. No permission bypass was attempted and no other job was terminated. User guidance on restoring execution access was requested.

The initial GitHub remote query failed inside the sandbox with `Could not resolve hostname github.com`. The subsequent sandbox-external `git push -u origin experiment/gvcrt-decoder-latent-restoration-v1-trend-300` was also not executed because automatic approval review returned HTTP 502. The experiment has local commits, but push and remote SHA verification remain uncompleted. Publication state is recorded separately in `publication_status.json`; remote publication must be verified before completion.

## Remaining work and local outputs

No training cache, branch checkpoint, actual GPU preflight result, decoded evaluation frames or quality metrics exist. All three branches still need exactly 300 effective updates, including smoke updates 1 and 2; all 32 step300 quality points, 63 transitions per point, paired differences, fixed comparisons, maps, training curves, same-device timing and final consistency checks remain pending. The implementation has passed Python syntax/provenance checks; its GPU execution has not been validated.

Run `run_pipeline.py` with the existing environment after GPU execution access is restored. It resumes saved updates without adding steps. Review the resulting checks, update this factual report, publish the experiment-only changes, and verify the remote SHA before writing COMPLETE.json. No COMPLETE.json has been created.

Results root: `/Huang_group/zyyz/home_dir/tmp/gvcrt-decoder-latent-restoration-v1/experiments/gvcrt_decoder_latent_restoration_v1_trend_300`.

Large/local artifacts: `local_artifacts/`, currently containing only the 8 reused evaluation streams. Their absolute paths, sizes and SHA256 are in `large_file_manifest.json`; they are ignored by Git. No research mechanism or bitrate-saving claim is made.

## Requested continuation recheck

At 2026-10-27T05:31:11.035089+00:00, the requested external GPU/process query again failed before execution because automatic approval review returned HTTP 502 Bad Gateway. The existing gvc-rt environment still reported CUDA unavailable in the sandbox, and the remote branch query still failed on github.com DNS resolution. No checkpoint files or training cache were found. The latest external waiter receipt is preserved in `continuation_recheck.json`; it continues to report WAITING. No new GPU pipeline was launched and no optimizer update was performed. GPU access and publication remain blocked; COMPLETE.json remains absent.

## GPU 1 and 4 authorization and execution check

The user explicitly authorized physical GPUs 1 and 4. The common setup guard, GPU queue, pipeline scheduler, effective configuration and protocol now restrict new launches to those two devices. Syntax checks and the CPU audit passed. With CUDA_VISIBLE_DEVICES=1,4, the existing gvc-rt environment still reported CUDA unavailable and zero visible CUDA devices; nvidia-smi could not connect to the driver. The requested external GPU/process query was again not executed because automatic approval review returned HTTP 502 Bad Gateway. Changing device selection therefore did not restore GPU access. No training step was performed and no pipeline was launched; all GPU checks, training, evaluation and remote publication remain pending. See gpu_1_4_check.json.

## Scheduling correction for GPUs 0 and 4

The user clarified that physical GPUs 0 and 4 are available. Updated setup, queue and scheduler to these devices. Removed the unnecessarily strict utilization <10% launch condition; the scheduler retains the 24,000 MiB free-memory check. The earlier WAITING state was caused by this utilization gate. The running launcher must restart to load the updated policy. restart_waiting_pipeline.py identifies only this experiment launcher by its exact script path and refuses to stop it when active tasks are recorded. This correction changes resource scheduling only, not the experimental training budget or model configuration.
