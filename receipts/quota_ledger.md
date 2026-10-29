# Kaggle GPU quota used by this project

Copied from the shared quota ledger (one Kaggle account, ~30 GPU-hours/week, shared with another project whose lines are left out). Format: session | project | estimate | launched (IST) | actual.

- ZDS probes x6 (T4, L4, TPU; two rounds) | speedrun | 10/29 | actual: ~6 x 1 min. T4 = 2x Tesla T4 OK; L4 request falls back to 2x T4; TPU request runs on CPU.
- ZDS smoke zds-smoke-s1 v1 | speedrun | launched 10/29 00:55 IST | actual: 1.7 min, died (dataset auto-extracted; fixed)
- ZDS smoke zds-smoke-s1 v2 (300 steps, 2x T4, 3 h cap) | speedrun | est ~1.5 h | launched 10/29 01:00 IST | actual: 2.68 h (stopped at step 266 by 3 h cap; val matches upstream)
- ZDS full zds-full-s1 (1450 steps, 2x T4, 12 h cap; chain s2 follows) | speedrun | est ~12 h + ~3 h | launched 10/29 04:33 IST | actual: 11.69 h, steps 0-1109, val 3.416 @1000
- ZDS full zds-full-s2 (chain from s1 at step 1110) | speedrun | auto | launched 10/29 16:18 IST | actual: 3.70 h, steps 1110-1449, FINAL val 3.2698 @1450 (target 3.28 HIT)
- ZDS rescore zds-rescore-s2 (eval-only: re-score s2 final ckpt, fp16 + fp32, 2x T4, 2 h cap) | speedrun | est ~0.5 h | launched 10/29 20:52 IST | actual: 0.32 h. fp16 re-score 3.2698315 (= logged, diff 0.0), fp32 3.2698309 (-5.8e-7). Final ckpt sha256 5bc6bac3...

Total: about 6 min of probes + 2.71 h smoke (v1 1.7 min, v2 2.68 h) + 15.39 h full run (11.69 h + 3.70 h) + 0.32 h re-score = **about 18.5 h**. Exactly one full-run attempt.
