# V6.34b fixed continuation protocol

This run resumes each v6.34 parent branch from its own latest complete transaction and stops at cumulative adaptation step 500. N resumes from 475; P resumes from 368. The parent experiment remains read-only. No training data, crop, QP, loss, learning rate, optimizer, EMA, RNG, buffer, codec, precision, or decoder semantics change. The inherited training plan is reused in its original order; each branch continues at its saved cursor.

The 6-video U-Long evaluation manifest and QP 0/5/9 are copied byte-for-byte from v6.34a. The 90 previous points are reused only after receipt and artifact size, modification time, and SHA256 checks. The only planned new encodes are N500/P500: 2 methods × 6 videos × 3 QPs = 36. Combined coverage is 126 logical points, with step-zero aliases not counted as additional encodes.

The parent source writes long-budget status fields as RUNNING after interruption. This continuation writes only to the new directory and uses a continuation driver capped at 500. Parent code, configuration, checkpoints, receipts and reports are not edited.

This is a short, single-seed trend screen. It does not claim convergence or broader generalization. No checkpoint selection or budget extension is performed.
