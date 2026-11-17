# Chart sources

`charts/make_charts.py` reads every number on the charts from the files below. Nothing is typed into the script.
Prices were checked on 2026-10-29 against the primary posts (fetched, quoted below).

## Our run (charts 1 and 2, and the $0 / T4-hours row of chart 3)
- `receipts/zds-full-s1/run/log.jsonl` and `receipts/zds-full-s2/run/log.jsonl` (`event: val`, step records,
  `event: resume`, `event: exit`). s2's log starts with a byte-identical copy of s1's (`receipts/verify.py` checks it), so the
  full curve is read from s2 and the session boundary is the `resume` record.
- `receipts/zds-full-s{1,2}/summary.json` `elapsed_h`: Kaggle session hours (11.692 + 3.696 = 15.388 h; × 2 GPUs
  = 30.78 T4-hours). Chart 2's x-axis is the trainer's own whole-process clock (`gpu_seconds_total`, 2 GPUs × wall
  time since process start, carried across the resume); it ends at 30.7 T4-hours, a little under the session total
  because it doesn't count the notebook's data download.
- Cost $0: both kernels ran on Kaggle's free tier (`receipts/zds-full-s{1,2}/kernel-metadata.json`: `machine_shape
  NvidiaTeslaT4`, `is_private true`; data from Hugging Face). No card on file, no bill.
- Not from these logs but used in the post: the first Kaggle attempt (smoke v1) died 1.7 min in because Kaggle
  auto-extracts uploaded `.tar.gz` datasets (fixed before the smoke run in `receipts/zds-smoke-s1`).

## Upstream record #18 (chart 1, and the derived row of chart 3)
- `charts/data/upstream_record18_31d6c427.txt`: verbatim copy of
  [modded-nanogpt `records/track_1_short/2025-01-04_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt`](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_1_short/2025-01-04_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt)
  at commit `4ea6b937`, sha256 `a746f5f876e870224067b135f43cac5606bbbb9d3f131203fc3f37cd13de7cf0` (MIT licence,
  Keller Jordan). Val loss every 125 steps on a 1,390-step schedule; final 3.2785, train time 204,345 ms (3.4 min) on
  8×H100.
- The 3.28 target: [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt#readme), "attains 3.28
  cross-entropy loss on the FineWeb validation set", chosen to match llm.c's GPT-2 (small) reproduction.

## Cost ladder (chart 3)
| key | label | year | cost_usd | rate_usd_per_hr | note | source |
|---|---|---|---|---|---|---|
| llmc | llm.c GPT-2 (124M) | 2024 | 20 | 14 | 8×A100 rented, ~90 min; 10B tokens; its val loss set the speedrun's target | https://github.com/karpathy/llm.c/discussions/481 |
| nanochat | nanochat speedrun | 2025 | 100 | 24 | 8×H100 rented, 3h51m; a ~560M chat model, a different job ($92.4 actual) | https://github.com/karpathy/nanochat/discussions/1 |
| this | This run, Kaggle free 2×T4 | 2026 | 0 |  | free T4-hours, no card on file | receipts/zds-full-s{1,2} |

All three prices verified against the primary posts on 2026-10-29 (quotes below).

Quotes (WebFetch, 2026-10-29):
- **llm.c**, "Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20", Andrej Karpathy, 2024-05-28: "on one 8X A100
  80GB SXM node takes ~90 minutes … on Lambda this node goes for ~$14/hr, so the total cost of reproducing this model
  today is about $20." It trains on 10B FineWeb tokens and is judged on HellaSwag (29.9 vs GPT-2's 29.4). The speedrun's
  3.28 target comes from this run's val loss.
- **nanochat**, "Introducing nanochat: The best ChatGPT that $100 can buy.", Andrej Karpathy, 2025-10-13: "This is
  costing us about ~$24/hr" for an 8×H100 box on Lambda; "Up to and including SFT, the whole thing ran in 3h51m";
  reported total $92.4. It trains a ~560M-parameter, 20-layer chat model (pretraining + midtraining + SFT), a bigger and
  different job from a 3.28 val-loss run. The chart shows the $100 headline and says so.

Not like-for-like, and the chart says so: llm.c and nanochat are rented-GPU prices for different jobs. The fair rented
comparison for *this* job is record #18 itself. The chart adds that row as **derived, not quoted**: record #18's logged
train time (3.4 min) × nanochat's quoted 8×H100 rate ($24/hr) ≈ $1.4. It excludes compile and startup time, so the
real rental would be somewhat higher. Nobody published that price; it is our arithmetic.
