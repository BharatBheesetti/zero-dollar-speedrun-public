# PORTING: modded-nanogpt record #18 → `trainer/train_t4.py` (T4 / SM75)

**Source.** github.com/KellerJordan/modded-nanogpt @ `4ea6b937337a4889b8cfe3f38a93d120048d8f71`,
`records/track_1_short/2025-01-04_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt` (record #18, "SoftCap").
It reached val 3.2785 after 1390 steps × 524,288 tokens (728.8M tokens) on 8×H100. Why #18 and not the current
record (#92 "ANVIL2"): see `research/01-recipe.md`. #92 needs FA3, FP8 and a 65B-parameter n-gram table.

**Run length: 1450 steps, not 1390** (`--num-iters`, default 1450; review M1 finding 1). Upstream's own 80 reruns
of #18 (`2025-01-04_SoftCap/README.md`) average 3.2791 ± 0.0019 and only 55/80 (69%) reach ≤ 3.28; the 3.2785 log
is a lucky draw, and the record was accepted on the 80-run mean. We get one run, and our deviations (fp16, fp32
embeddings, 8K pieces) move the mean by an unknown amount. Record #21's README puts one step at ~0.00015 nats late
in training, so +60 steps buys ~−0.009: an expected ~3.270 before port losses. Cost: 760.2M tokens instead of
728.8M (+4.3%), still inside the 8 downloaded train shards (190 steps each = 1520). The schedule keeps upstream's
shape: the LR cooldown is the last 40% of the new total (steps 870–1450) and the window schedule is linear over the
new total; the Muon momentum warmup stays at upstream's fixed 300 steps. `--num-iters 1390` reproduces upstream's
schedule exactly (the S1 smoke uses it, so its val at steps 125/250 compares 1:1 with upstream's 4.3686/3.9427).
At 1450 the window schedule already differs from step 103 (each window step comes a little later), so the full
run's val points are near, not exact, comparisons; the val log says which via `schedule_same_as_upstream`.
Report it as "a single run, 760M tokens vs upstream's 729M", never as record-equivalent.

**What is verified, and where.** "CPU-tested" means `make test` covers it (tiny config: 12 layers, d=96).
Everything GPU-specific is marked **verify on T4**. We have not run on a GPU yet.

## Feature by feature

| # | Upstream #18 feature | Status | What we do / why |
|---|---|---|---|
| 1 | 12 layers, d=768, 6 heads × 128, vocab 50257→50304 | kept | Same shapes. `make dry-shape`: 275.7M params, 121.2M without the four 50304×768 embedding tables |
| 2 | Layer 7 has no attention | kept | |
| 3 | ReLU² MLP, 4× width, zero-init `c_proj` | kept | **verify on T4**: fp16 overflows if a pre-activation exceeds ~255.9. Overflows show up as skipped steps (`overflow_skips` in the log) |
| 4 | QK-norm (RMS over the head dim) | kept | Computed in fp32 |
| 5 | Half-truncated RoPE, base 1024, `chunk(2)` layout | kept | cos/sin computed on the fly in fp32, not cached. Same values |
| 6 | Zero-init `lm_head`, untied embed/head | kept | |
| 7 | 3 value embeddings, U-net pattern `012 … 012` | kept | |
| 8 | Attention `lambdas` (v/ve mix), block `lambdas` (x/x0 mix), U-net `skip_weights` | kept | |
| 9 | Logit softcap `15·tanh(z/15)`, CE in fp32 | kept | CE runs in row chunks with activation checkpointing, so [rows, 50304] fp32 logits stay small. Mean over the step's tokens, as upstream |
| 10 | bf16 activations (autocast-free bf16 casts) | **changed** | fp16 `torch.autocast` with fp32 master weights. RMSNorm, rotary and CE run in fp32, and the residual stream stays fp32. T4 has no bf16 tensor cores |
| 11 | `bf16_embeds=True` (embedding tables stored bf16, Adam on bf16) | **changed** | fp32 tables and fp32 Adam state. Costs memory; avoids fp16 optimizer-state overflow (research/02-t4 §4). This also changes the dynamics, not just memory (review M1 finding 11): late in the cooldown (lr_mult → 1/556) upstream's Adam updates fall below half a bf16 ulp, so its tables effectively freeze, while ours keep moving. The effect could go either way. `--round-embeds-bf16` rounds the tables to bf16 after each Adam step (off by default; the cheap thing to try if run 1 misses) |
| 12 | FlexAttention: document-causal `mask_mod` over a block-level causal sliding window (128-token blocks) | **changed** | `windowed_attention()`: torch SDPA with a boolean mask, queries in chunks of `--attn-chunk` (1024), each attending to its own keys plus the `window` keys before it. The mask is upstream's exactly (`j<=i`, same document, `i//128 − j//128 < window_blocks`). CPU-tested against a dense reference of that mask. No block skipping, so attention FLOPs are chunk×(chunk+window), not window-only. **verify on T4**: SDPA must pick the memory-efficient kernel (not math) with this bool mask, and the speed has to be measured |
| 13 | Window schedule 128 → 1792 tokens, linear, in whole blocks | kept | CPU-tested against upstream's formula |
| 14 | Batch: 8 ranks × one 65,536-token sequence per step | **changed** | Same 524,288 tokens per step, split the same way (contiguous 64K sequences). Each sequence runs as `--micro-tokens` pieces (default 8192; 16384 if it fits T4 memory) with gradient accumulation. Gradient = mean over the step's tokens, same as upstream. CPU-tested: the pieces are exactly the tokens a verbatim copy of upstream's `DistributedDataLoader` gives each rank (`test_pieces_are_upstreams_tokens`). **Difference:** attention cannot see across a piece boundary. Measured on `fineweb_train_000001` (review M1 finding 3), averaged over the window schedule, the scored tokens with a truncated multi-layer receptive field are 12.0% at 8K pieces, 5.5% at 16K. Two fixes, both off by default so the smoke can A/B them: `--doc-ctx 2048` prepends the unscored tokens from the 128-aligned start of the document containing the piece's first token, capped at 2048. The doc mask hides earlier documents and the alignment keeps upstream's block grid, so such a piece is **exact at every layer** (CPU-tested against a whole-sequence forward, `test_doc_ctx_is_exact`). On real data 84% of non-first pieces become exact, for +9.8% tokens forwarded at 8K pieces or +4.9% at 16K. `--train-ctx N` (fixed N tokens; 1792 makes only layer 1 exact) costs ~19% at N=1792. Re-read context is logged as `ctx_tokens` and is not counted in `tokens_seen`. Trade-off: the loss cost of truncation is unmeasured and could be the same size as our margin (see Run length), while `--doc-ctx` costs ~5–10% more compute; the default stays off until the smoke A/B, then turn on `--doc-ctx 2048` for the record run if the smoke shows the ~10% cost (or ~5% at 16K pieces) is affordable. **verify on T4**: `make dry-shape DEVICE=cuda ARGS="--micro-tokens 16384 --doc-ctx 2048"` (dry-shape forwards micro + ctx tokens) |
| 15 | Data loader: one contiguous stream over `fineweb_train_*.bin`, wraps cyclically | kept (no wrap) | Same token order and shard-advance rule. Upstream wraps to shard 0 when the shards run out; we refuse to start if `--num-iters` (or `--stop-after-step`) needs more steps than the shards hold, and error out if they run out, unless `--allow-wrap` (then `event=data_wrap` is logged). The checkpoint stores `(shard, position)` plus the shard's file name and byte size; a resume where that shard index is a different file is refused (review M1 findings 5, 6). Read with `np.memmap`; same header checks. Real-data check (`tests/test_data.py`, skipped without `data/`): both shard headers carry upstream's magic 20240520 and version 1 and match the file size; the first 200 tokens of the val shard and of the first train shard decode with tiktoken `gpt2` to English (with a random-token control that fails the same check); the val slice's sha256 equals the pinned upstream value. (The old evidence, step-0 loss 10.8258 = ln 50304, proved nothing: with a zero-init `lm_head` it holds for any tokens.) |
| 16 | Muon: lr 0.05, Nesterov, momentum 0.85→0.95 over 300 steps, 5 quintic NS steps (3.4445, −4.7750, 2.0315), lr × max(1, rows/cols)^0.5 | kept | NS runs in **fp32** (upstream bf16). Momentum buffer fp32. Upstream shards NS across ranks and all-gathers the updates. We run it on every rank (replicated), which is simpler. Replicated, it costs about twice research/02-t4's split estimate: ~0.25–0.4 s/step, 2–3% (review M1 finding 13). fp16 NS on tensor cores would be safe (norm-scaled input) and nearly free, if it matters. Each rank keeps its own optimizer copy; at every checkpoint the ranks compare per-parameter |sums| and the run exits with code 4 if they drifted (`--rank-check-tol`, CPU-tested with 2 gloo ranks: identical). **verify on T4**: NS share of step time |
| 17 | Adam: embed lr 0.6, head 0.008, scalars 0.04; betas (0.8, 0.95); eps 1e-8; no weight decay | kept | Written out by hand (same formula as `torch.optim.Adam`) so its state lives in our checkpoint. `fused=True` dropped |
| 18 | LR: constant, then linear decay over the last 40% | kept | Update of step `it` uses `lr(it)`, as upstream's LambdaLR does. The 40% is of `--num-iters` (default 1450, see above) |
| 19 | DDP (bucketed, overlapped all-reduce) | **changed** | One flat fp32 all-reduce per optimizer step, after accumulation and before unscale, so every rank sees the same inf check. CPU-tested: 2 gloo ranks match 1 rank to ≤1e-5 over 20 steps (`tests/test_paths.py`, which also runs the fp16 + GradScaler path). **verify on T4**: ~1.1 GB per step over PCIe on 2×T4. If slow, add an fp16 compression hook or overlap |
| 20 | `torch.compile(model)` and compiled NS | **changed** | Off by default (`--compile` to try). On SM75, Triton matmuls run without tensor cores. Our matmuls already go to cuBLAS, so compile would only fuse pointwise ops. **verify on T4**: on vs off |
| 21 | Validation: first 10,485,760 val tokens as 160 independent 65,536-token sequences, final window | kept | Exact. Each val sequence is one forward (`--val-chunk` 65536). A smaller `--val-chunk` stays exact: each chunk gets a receptive-field prefix (CPU-tested) |
| 22 | `val_loss_every = 125` | changed | Trainer default 0 (final only). The Kaggle lane sets it: smoke 125 (on upstream's 1390-step schedule, so steps 125/250 compare 1:1 with 4.3686/3.9427), full 250. Re-plan if the full run trails upstream by more than ~0.02 at step 500 |
| 23 | Timer reset at step 10, `torch.empty(...).backward()` workaround, code-in-log | dropped | Not needed. The jsonl log records upstream commit, record and torch version |
| 24 | Save state only at the end | **replaced** | Atomic checkpoints (below) |

## Added (not in upstream)
- **GradScaler (fp16).** `torch.amp.GradScaler`, init scale 2^16. Each piece's loss is `sum/M` to keep fp16
  gradients in range. Grads are unscaled, then checked for inf/NaN by `GradScaler` across all parameters, then
  rescaled to the step mean. On overflow **both** optimizers skip the step and the scale halves (CPU-tested: weights
  and Adam step counts unchanged). The skip count is logged every step. **verify on T4**: how often early
  steps skip.
- **Overflow guard** (review M1 finding 2). A forward overflow (e.g. a ReLU² input above ~255.9) makes the loss
  non-finite at any loss scale, so every later step would skip while the run exits 0 hours later. Now: after
  `--max-consecutive-skips` (default 20) skipped steps in a row the trainer logs `event=abort` and exits with code
  3, **without** checkpointing; periodic checkpoints are also held back during a skip streak, so the newest
  checkpoint is always the last good state. Each step record has `nonfinite_fwd` (forward loss non-finite) apart
  from `skipped` (any non-finite gradient). `event=scale_alarm` is logged when the scale drops below
  `--scale-floor` (default 1). Every `--amax-every` (25) steps the log has `c_fc_amax` (max |ReLU² input| per
  layer). `--mlp-prescale` computes `c_proj((relu(x)/4)²)·16` in fp32 (exact powers of 2) for headroom up to
  ~1023; off by default. With `--scaler off` a non-finite gradient also skips the step. CPU-tested
  (`test_overflow_streak_aborts_nonzero` injects inf losses and checks exit code 3, the alarm and the checkpoint).
- **Crash-proof checkpoints.** Every `--ckpt-every-min` (default 20), on SIGTERM/SIGINT, on `--time-budget-min` and
  at `--stop-after-step`. Contents: model, Adam + Muon state, GradScaler, data position, all RNG states, step,
  tokens, GPU-seconds, wall-clock and per-session provider segments. Written to a temp file, fsynced, then
  renamed. The JSON sidecar (with the byte size) is written last and is the commit marker. A `.pt` without a
  matching sidecar is ignored, and leftover temp files are deleted. `--resume` loads the newest valid checkpoint.
  `make test` kills a run with `kill -9` at a random step in 15–45 and checks that the resumed per-step losses
  equal the uninterrupted run's (≤1e-6; on CPU they are bit-identical). **verify on T4**: GPU resume is close but
  not bitwise (cuBLAS/SDPA nondeterminism). Checkpoint size is ~3 GB for the full config (params 1.1 GB, Adam
  1.5 GB, Muon 0.34 GB); time the write to `/kaggle/working`, and note that 2 kept = ~6 GB.
  SIGTERM caveat (review M1 finding 14): torchrun's SIGTERM-to-SIGKILL grace is short, so finishing a step plus a
  ~3 GB write may not fit; the run then falls back to the last periodic checkpoint (≤20 min lost, never corrupt).
  `--time-budget-min` is the primary stop, and the lane's watchdog signals the workers directly, not torchrun.
  `--resume` loads the checkpoint to CPU memory first (no ~3 GB transient on the GPU).
- **Claims bookkeeping** (review M1 finding 7). The `start` record logs the full args, the train file names and
  sizes, the step capacity of the shards, and the val set: file name, byte size, file sha256, token count and the
  sha256 of the tokens actually scored (`[0, 10,485,761)`), next to `val_set_is_upstream`, which is true only when
  both sha256s equal the pinned upstream values (`VAL_FILE_SHA256`, `VAL_SLICE_SHA256`), the count is
  10,485,760 and sequences are 65,536 tokens. Every `val` record repeats the val token sha256 and count, and at
  upstream's logged steps adds `upstream_val_loss`, `val_minus_upstream` and `schedule_same_as_upstream`. Time:
  `compute_s`/`gpu_seconds` are train steps only; `wall_total_s`/`gpu_seconds_total` are whole-session wall-clock
  from process start (startup, validation, checkpoints included; `val_s`, `ckpt_s` broken out), carried across
  resumes. Use `gpu_seconds_total` for any "free GPU-hours" claim. A killed session's time after its last
  checkpoint and the lane's data download are not in it; the lane's `summary.json` `elapsed_h` has the true
  session length.
- `--dry-shape`: full config, one forward/backward + optimizer step on random tokens, prints params and peak
  memory. CPU (1024 tokens, fp32): 5.4 GB peak RSS. **verify on T4**: `make dry-shape DEVICE=cuda` at
  `--micro-tokens` 8192 and 16384, to pick the largest piece that fits in 15 GB.
- `--synthetic` data and `--tiny` config for tests. `--canon-mask on` also reports val loss with record #91's
  canonical-pair mask (needs tiktoken). It is off by default and not part of #18, so disclose it if used.

## Verify on T4 (first GPU session)
1. `make dry-shape DEVICE=cuda` at micro 8192/16384, each with and without `--doc-ctx 2048`: peak memory and
   fwd/bwd time.
2. Confirm the SDPA backend is memory-efficient, not math (profile one step).
3. 300-step smoke on 2×T4: tokens/s, NS share, all-reduce time, `overflow_skips`, ReLU² amax.
4. `--compile` on vs off.
5. `--doc-ctx 0` vs `2048` (and `--micro-tokens 8192` vs `16384`) over ~300 steps: step time and val at 250. The
   loss difference is likely below the run-to-run noise at 300 steps, so decide on cost and on the exactness
   argument above.
6. Checkpoint write time and size on Kaggle disk. Do one kill + `--resume` on GPU.
7. One full validation pass: time and memory at `--val-chunk 65536`.

## Optional later tricks (fp16/SM75-safe; not in the default trainer)
Research order (research/01-recipe.md §6), once the #18 port reaches ≤3.28: batch-size schedule (#46), multi-token
prediction (#53), 377K-row hashed bigram embedding (#62), NorMuon + cautious weight decay (#41, #43), tail
weight averaging (#92). Canonical masking at val (#91) is already available behind `--canon-mask on`.

Before the pivot to #18, commit `eff86c8` (`trainer/train_t4.py` at that SHA) held a plain-torch port of most of
the current record #92 (ANVIL2), minus FP8, FA3, CUDA graphs and Triton kernels. It has the 11-layer
mixed-width topology, paired-head attention, partial key offset, XSA, attention and value gates, MUDD dense
skips, smear, YaRN, MTP + prefix-token loss, the ANVIL optimizer (twin-rail momentum, six-map cascade in fp32,
lane equalizer, cautious decay), Adam on odd steps, an embed/head tie-then-split, a hashed bigram+trigram table
shrunk to 754,560 rows with its row-sparse beta1=0 Adam, the stage/batch/window schedule, and tail averaging.
It ran the tiny CPU config end to end (loss fell). It was **not** resume-tested and is not maintained. Mine it
for the tricks above rather than running it as is. Never port: FP8, FA3, FlexAttention, Triton `tl.dot`
kernels, the 84.6M-row n-gram table.
