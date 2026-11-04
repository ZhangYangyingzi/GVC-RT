# Online development stopping revision v3

This mid-training revision inherits source-group-isolation-v2 and native_geometry_v1. It was introduced after A committed additional595 / absolute5595 and B additional0 / absolute5000. The old protocol, source files, checkpoints and receipts remain unchanged. The original frozen plans and data hashes are unchanged.

12710 additional updates is now a safety cap, with no required minimum coverage. LR remains1e-4 and losses, precision, optimizer moments, clipping, frozen modules and TCG architecture remain fixed. Only tcg_low/mid/out137760 parameters train, with detached history and no BPTT or gates.

The preferred runtime assigns A physical GPUs4,5 and B6,7. Each physical rank computes two of the four original logical samples using four preserved logical RNG streams. Gradients SUM across ranks and divide once by4, followed by one clip1 and optimizer.step. Independent four-card/reference and two-card/accumulation probes check losses, gradients, parameter deltas, optimizer and RNG without committing formal updates. Existing relative-L2<1e-3 applies without relaxation. A failed probe selects four-card alternating validation segments.

All training and helper processes exit at rendezvous before validation. The fixed32 development videos, native64-frame windows and QP0/5/9 are unchanged. The common parent5000 baseline is evaluated first. At500,1000,...,12500 and the12710 cap, complete development evaluations produce equal-video FloLPIPS means F_q, and M=mean_q(F_q(s)/F_q(parent5000)). Baseline denominators must be finite and positive. Invalid or incomplete evaluations never count for patience.

Each arm independently tracks raw best including parent5000, improvement anchor initially1, and bad-check count initially0. M<anchor-0.005 resets the count and anchor; otherwise count increases. Raw best updates whenever M decreases, retaining the earlier checkpoint on exact ties. Three consecutive non-substantive complete checks stop that arm. Baseline does not count. States are atomic and replayed on resume. A500 is evaluated retrospectively from its saved checkpoint; committed595 is not rolled back.

Same-budget K_common is the largest intersection of completed planned validation steps, selected by step rather than score gap. Separate per-arm raw-best development states may have unequal budgets. After both arms stop, freeze parent0,parent5000,A/B common and A/B dev-best selections, deduplicate identical model states, then evaluate the actual unique-state Cartesian product on32 new-test plus13 historical-reference sequences and QP0/5/9. Test results never affect stopping or model selection.

Real RANS fresh encoding and independent decoding, persisted-byte rates, per-frame/per-transition/source/model hashes, existing PWC sampler definitions and isolated publication requirements remain unchanged. Training coverage and allocated-card wall time are reported as actually observed. No conclusion about convergence, effectiveness or generalization is made.

Active entry: launch_v3.py. Read-only status: v3_status.py. Atomic boundary pause: control_v3.json pause_requested=true. Resume: the fixed Python interpreter with -B -u launch_v3.py --resume, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=4,5,6,7. The launch/controller locks prevent a second runner.
