# Runtime guard

`runtime_guard.py` watches the existing experiment controller. It uses its own
single-instance lock and runs in a separate session. It never signals processes
and does not start another controller while `runner.lock` is held.

The guard permits at most one recovery of an interrupted controller whose last
persisted task status is `RUNNING`. Recovery also requires that all recorded
workers have exited and all task worker locks are free. It verifies the frozen
sources before invoking the unchanged `runner.py`, which restores its atomic
checkpoint and committed receipts. An explicit `FAILED`, `BLOCKED`, `PARTIAL`,
or `COMPLETE` status stops the guard without retrying the task.

The existing foreground `runner.py | tee ...` process was left running. `tee`
records output but does not detach a process from its terminal. The added guard
provides recovery if that terminal interrupts the controller. See
`audits/runtime_guard_installation.json` for the actual PID, session, source
SHA256 and log path. Live guard state is in `ops/runtime_guard_status.json`.

This operational addition does not change the frozen model, training sources,
data plan, loss, calibration, budgets or evaluation rules. The empty-validity
region code revision and checkpoint migration are separately preserved in
`audits/source_revision_empty_gt_valid_region.json`. The resumed step32 and
checkpoint/receipt checks are recorded in `audits/resume_inspection_20261109.json`.

The local FFmpeg supports `xstack` without its newer `fill` option. This was
observed while exporting the step100 viewing preview. `ffmpeg_compat.py`
translates only the exact seven-panel task layout to `hstack`/`vstack` with an
explicit black eighth tile. The successful preview composition verifies that
layout; its model outputs and measured-frame hashes are unchanged.

`final_video_compat_guard.py` leaves the current runner and training processes
untouched. If the final montage later fails with this exact verified FFmpeg
error, after all four branches have completed, it archives the failure and
resumes the unchanged runner once with a task-private FFmpeg executable in its
PATH. The executable is generated from the guard's recorded source. It never
modifies the system FFmpeg, signals a process, edits frozen research sources,
replays completed optimizer updates, or retries an unrelated failure.
