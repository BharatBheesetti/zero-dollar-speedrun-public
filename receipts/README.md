# Receipts

Every number in the headline claim, and the file and field that proves it. `python3 receipts/verify.py`
re-derives them all from this folder (stdlib only) and exits 1 if any doesn't hold.

> In a single run on Kaggle's free 2×T4 tier, costing $0, our fp16 port of the modded-nanogpt GPT-2-small speedrun
> model (record #18, 2025-01-04 SoftCap) reached 3.2698 FineWeb validation loss on the standard 10.49M-token val
> set, below the speedrun's 3.28 target. It took 1,450 steps and 760M training tokens (4.3% more than the 8×H100
> record's 729M) and 15.4 Kaggle session-hours (30.8 T4-hours) across two sessions (11.7 h + 3.7 h) joined by a
> checkpoint resume.

## Number → proof

| Claim | Value | File | Field |
|---|---|---|---|
| single run | 1 full attempt | `quota_ledger.md` | the ZDS lines: one `zds-full` series, no other full attempt |
| Kaggle 2×T4 | `NvidiaTeslaT4`, 2 × Tesla T4 | `zds-full-s{1,2}/kernel-metadata.json`; `zds-full-s{1,2}/summary.json`; `zds-full-s{1,2}/kernel.log` | `machine_shape`; `n_gpu` = 2, `gpu` = "Tesla T4"; nvidia-smi prints `Tesla T4, 15360 MiB` twice |
| free tier, $0 | private kernels on a free account | `zds-full-s{1,2}/kernel-metadata.json` | `is_private` true; no paid accelerator exists on the account (Kaggle bills nothing) |
| fp16 port of record #18 | fp16 + GradScaler, upstream `4ea6b937`, record #18 | `zds-full-s2/run/log.jsonl` line 1 (`start`) | `amp` "fp16", `scaler` "on", `upstream`, `record` |
| **3.2698** | 3.2698315009167063 | `zds-full-s2/run/log.jsonl` line 1506 | `event` "val", `step` 1450, `val_loss` |
| standard val set | upstream's file and token slice | same line; line 1 (`start`) | `val_set_is_upstream` true, `val_tokens_sha256`; `val_file_sha256` (pinned values in `val_set.json`) |
| 10.49M-token val set | 10,485,760 tokens, 160 × 65,536 | same line; line 1 | `val_tokens`, `val_seq_len`; `val_sequences` 160 |
| below 3.28 | 3.2698 ≤ 3.28 | as above | |
| 1,450 steps | steps 0..1449 | `zds-full-s2/run/log.jsonl` | one step record per step; line 1 `total_steps` 1450 |
| 1,449 optimizer updates | one skip at step 1006 | `zds-full-s2/run/log.jsonl` line 1043 | `skipped` true, `scale` 65536.0, `nonfinite_fwd` false; `overflow_skips` stays 1 to the end |
| 760M training tokens | 760,217,600 | `zds-full-s2/run/log.jsonl` lines 1505, 1506 | step 1449 `tokens`; final val `tokens` |
| 4.3% more than 729M | 760,217,600 / 728,760,320 − 1 = 4.32% | upstream's 1,390 steps × 524,288 tokens | (upstream log, record #18) |
| 15.4 session-hours | 11.692 + 3.696 = 15.388 | `zds-full-s1/summary.json`, `zds-full-s2/summary.json` | `elapsed_h` (notebook wall-clock from first cell to summary) |
| 30.8 T4-hours | 2 × 15.388 = 30.78 | as above | `n_gpu` × `elapsed_h`. The trainer's own count is 30.70 h (`zds-full-s2/run/log.jsonl` line 1508, `exit` `gpu_seconds_total` 110,503) |
| two sessions, 11.7 h + 3.7 h | | as above | `elapsed_h`. Each was capped at 12 h |
| joined by a checkpoint resume | s1 stopped on its time budget at step 1109; s2 resumed at 1110 | `zds-full-s1/summary.json`; `zds-full-s1/run/ckpt/ckpt_0001110.json`; `zds-full-s2/run/log.jsonl` line 1153 | `exit_reason` "time_budget"; sidecar `next_step` 1110, `tokens_seen` 581,959,680, `gpu_seconds` 80,443.69 = the `resume` record's values. s2's log lines 1–1152 are byte-identical to s1's log. `zds-full-s2/kernel.log` shows the copy of `ckpt_0001110.pt` |
| no repeated or skipped data across the resume | | `zds-full-s{1,2}/run/ckpt/*.json` | `data_position`: `shard_idx`, `pos` = (step − 190·shard_idx) × 524,288; `wraps` 0 |
| 275.7M params incl. embeddings | 275,742,772 | `zds-full-s2/run/log.jsonl` line 1 | `params` |
| the val'd weights are the saved weights | step 1450, reason final | `zds-full-s2/run/ckpt/ckpt_0001451.json` | `step` 1450, `reason` "final", `tokens_seen` 760,217,600 |
| re-scored outside training | fp16 3.2698315009167063 (diff 0.0); fp32 3.2698309167 (diff −5.8e−7) | `rescore.json` (= `zds-rescore-s2/rescore.json`) | `fp16.result.passes[0].val_loss`, `fp32.result.passes[0].val_loss`, `minus_logged`. A fresh 2×T4 process loaded `ckpt_0001451.pt` from s2's output and ran the trainer's own `validate`; `val_set_is_upstream` true |
| the re-score used the training code | trainer sha256 `ff541245…8312` | `rescore.json` | `fp16.result.trainer_sha256` = `git show b3343e0:trainer/train_t4.py \| sha256sum` |
| checkpoint hashes | final `5bc6bac3…c4ae`, s1→s2 link `ckpt_0001110.pt` `6d6a7a4d…9225` | `rescore.json` | `fp16.result.ckpt_sha256`; `hashes["zds-full-s1/run/ckpt/ckpt_0001110.pt"]`. Model-only fp32 weights `zds_final_model_step1450.pt`, sha256 in `fp16.result.model_sha256`, stay in the private Kaggle output |
| code version per session | smoke `3b6f049`, s1 and s2 `b3343e0`, re-score `38fff7a` | `*/kernel.log` | `code git sha:` line (from the code dataset's `manifest.json`; no `-dirty`) |

## Files

| Path | What |
|---|---|
| `zds-smoke-s1/` | 300-step smoke on upstream's exact 1,390-step schedule (val at 125/250 vs upstream). Stopped at step 266 by its 3 h cap |
| `zds-full-s1/` | full run, session 1: steps 0–1109 |
| `zds-full-s2/` | full run, session 2: steps 1110–1449, final val |
| `zds-rescore-s2/` | eval-only kernel: the final checkpoint scored again in a fresh process |
| `*/run/log.jsonl` | the trainer's log: one record per step, plus `start`, `val`, `checkpoint`, `resume`, `exit` |
| `*/run/train_*.out` | the trainer's stdout |
| `*/run/ckpt/*.json` | checkpoint sidecars (the `.pt` files are 3 GB each and stay on Kaggle) |
| `*/summary.json` | the notebook's summary cell |
| `*/kernel.log` | Kaggle's execution log (`kaggle kernels logs`) |
| `*/kernel-metadata.json`, `*/*.ipynb` | what was pushed (`kaggle kernels pull -m`): machine shape, privacy, sources, the notebook with its embedded run config |
| `rescore.json` | the re-score result (copy of `zds-rescore-s2/rescore.json`) |
| `val_set.json` | the val set's pinned hashes and where each run logged its own |
| `quota_ledger.md` | every Kaggle GPU session this project used, from the shared quota ledger |
| `autochain.log` | the unattended watcher that pulled s1 and pushed s2 |
| `verify.py` | re-derives every number above |

Upstream's log for comparison is not copied here: modded-nanogpt @ `4ea6b937`,
`records/track_1_short/2025-01-04_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt` (3.2785 at step 1390).
