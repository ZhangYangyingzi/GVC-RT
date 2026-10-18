# Data and reuse audit

The fixed six-video evaluation manifest is reused byte-for-byte from v6.34a. The training rows follow the parent shared plan at the saved per-branch cursor; no new windows, sources, or crops are selected. The parent data isolation audit is included as a read-only source copy. It verifies the training pool against the listed evaluation and historical exclusion manifests; it does not claim exhaustive content-level near-duplicate screening.

The 90 earlier evaluation points were accepted only after checking each receipt and all recorded artifacts for path, byte count, modification timestamp, and SHA256. This continuation creates only the two step-500 model endpoints (36 encodes).
