# Execution record

This run overrides the original v6.34 long budget with the user's v6.34a trend-screen instruction. Existing v6.34 step-100/250 model states are reused only after checkpoint/index hash and complete-state checks. The old 5000-step driver PIDs in its last persisted state were absent from the live process table at inspection, so no process was signalled and no unrelated task was touched.

No training is launched in v6.34a. Fixed evaluation budget: six reserved U-Long training-pool holdout sources × three QPs × (original plus N/P at steps 100 and 250), with N/P step-0 aliases, 90 logical points total.
