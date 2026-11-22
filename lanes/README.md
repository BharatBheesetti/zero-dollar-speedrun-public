# Kaggle lane (L1)

`lanes/kaggle_push.py` ships `trainer/` + `scripts/` to Kaggle as a private dataset, runs `trainer/train_t4.py` on
GPU T4 x2 under `torchrun`, and pulls the logs back. Offline commands (`pack`, `kernel`, `dryrun`) need no
credentials. Every live command refuses to run unless `ZDS_KAGGLE_LEDGER_OK=1` is set.

## Before every live session
1. Credentials: `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`) in `~/.config/zds/env` (override the
   path with `ZDS_KAGGLE_ENV_FILE`). Without that file: the same variables from the environment, else
   `~/.kaggle/kaggle.json`. They are passed only to the `kaggle` CLI subprocess and never printed.
2. Kaggle GPU quota is ~30 h/week. Log the session in your quota ledger: kernel, est hours, launched. Set
   `ZDS_LEDGER=/path/to/ledger.md` to have `lanes/autochain.py` append chained sessions to it. Then export
   `ZDS_KAGGLE_LEDGER_OK=1` in that shell.

## Command sequence
```sh
make kaggle-dryrun                                                       # offline: pack, render, validate
export ZDS_KAGGLE_LEDGER_OK=1                                            # only after the ledger line
.venv/bin/python lanes/kaggle_push.py quota                              # GPU/TPU hours left this week

# 1. smoke: 300 steps (~1.5 GPU-h est.), 3 h cap. Uploads the code dataset, pushes zds-smoke-s1.
.venv/bin/python lanes/kaggle_push.py smoke
.venv/bin/python lanes/kaggle_push.py status --mode smoke
.venv/bin/python lanes/kaggle_push.py pull --mode smoke                 # -> out/kaggle/zds-smoke-s1/summary.json
#    summary.json: tokens_per_s_median, peak_gpu_mem_mib_nvidia_smi, overflow_skips_*, step_s_median,
#    val_vs_upstream (smoke runs upstream's 1390-step schedule with val every 125: step 250 should be ~3.9427)
#    A/B variants: add e.g. --trainer-arg=--doc-ctx --trainer-arg=2048 or --trainer-arg=--micro-tokens --trainer-arg=16384

# 2. full: to the target (1450 steps, trainer default; see trainer/PORTING.md "Run length"). Stops itself 20 min
#    before the 12 h cap. Pushes zds-full-s1.
.venv/bin/python lanes/kaggle_push.py full --skip-dataset
#    The smoke checkpoint is NOT a prefix of the 1450-step run (the window schedule differs from step 103; the
#    trainer refuses the resume). Continuing it means running upstream's 1390 steps:
#    .venv/bin/python lanes/kaggle_push.py full --skip-dataset --resume-from <user>/zds-smoke-s1 \
#        --trainer-arg=--num-iters --trainer-arg=1390

# 3. chain: only if s1 ended without "done": true. Attaches zds-full-s1's output and resumes its newest checkpoint.
.venv/bin/python lanes/kaggle_push.py status --mode full --session 1     # must be complete first
.venv/bin/python lanes/kaggle_push.py chain --session 2 --skip-dataset  # -> zds-full-s2
.venv/bin/python lanes/kaggle_push.py pull --mode full --session 2      # repeat with --session 3... until done

# 4. re-score: eval-only kernel zds-rescore-s2 attaches s2's (and s1's) output, scores the
#    final checkpoint with scripts/rescore.py in fp16 and fp32, hashes the joining checkpoints. ~0.5 h, 2 h cap.
.venv/bin/python lanes/kaggle_push.py rescore                           # uploads the code (rescore.py), pushes
.venv/bin/python lanes/kaggle_push.py pull --kernel zds-rescore-s2      # -> out/kaggle/zds-rescore-s2/rescore.json
.venv/bin/python lanes/kaggle_push.py files zds-full-s2                 # list a kernel's outputs (no download)
.venv/bin/python lanes/kaggle_push.py meta zds-full-s2 --dir D          # its pushed notebook + kernel-metadata.json
```
`--skip-dataset` reuses the uploaded code. Drop it when `trainer/` changed. Each run command logs its git SHA in
the dataset's `manifest.json`.

## What a session does
Kernel `zds-<mode>-s<N>` is private, with `machine_shape: NvidiaTeslaT4` (T4 x2), internet on, the dataset
`<user>/zds-code` attached, and for resumes `kernel_sources: ["<user>/<previous kernel>"]`. The notebook:
1. Untars the code into `/kaggle/tmp`.
2. Runs `scripts/get_data.py` to fetch the FineWeb val shard plus 2 (smoke) or 8 (full) train shards into
   `/kaggle/tmp/fineweb10B`.
3. If resuming, copies the newest committed checkpoint (`.json` sidecar size matches the `.pt`) and `log.jsonl`
   from the attached output into `/kaggle/working/run/`. If it finds none, it refuses to start from step 0.
4. Runs `torchrun --standalone --nproc_per_node=2 trainer/train_t4.py --provider kaggle --resume
   --ckpt-every-min 20 --time-budget-min <cap - 20 min - setup time>`. A watchdog SIGTERMs the workers
   10 min before the cap (they checkpoint on SIGTERM) and SIGKILLs them at 4 min. Kaggle keeps no output from a
   session that hits the cap.
5. Writes `/kaggle/working/summary.json`.

Trainer note: `--resume` is a plain flag ("newest valid checkpoint if any"). There is no `--resume auto` value.

## Does this account get TPU v5e-8 or L4?
The metadata key is `machine_shape`, same as `--accelerator` on `kaggle kernels push`. The values are
`TpuV5E8`, `NvidiaL4` and `NvidiaL4X1`. kaggle-cli `docs/kernels.md` lists them as of Sep 2026, but "some are only
available to participants of specific competitions". `kaggle quota` shows TPU hours if the account has any. One
line, logged in your quota ledger first, costs a few quota minutes:
```sh
for a in TpuV5E8 NvidiaL4X1; do ZDS_KAGGLE_LEDGER_OK=1 .venv/bin/python lanes/kaggle_push.py probe --accelerator $a; done
```
If the push errors on an accelerator, the account can't use it. Otherwise `kaggle kernels output
<user>/zds-probe-<acc> -p out/kaggle/` gives `probe.json` with `nvidia-smi -L` or `/dev/accel*`.

## Files
- `kaggle_push.py`: the CLI. Denylist: `.venv/`, `data/`, `ckpt/`, `out/`, `build/`, `.git/`,
  `__pycache__/`, anything that resolves under `~/.config`, and key/token/env file names. `*.pt` and `*.bin` are
  also denied.
- `kaggle_notebook.py`: the notebook cell sources. The run config is embedded in the first cell.
- `templates/*.tmpl`: dataset and kernel metadata.
- Tests: `tests/test_kaggle_lane.py`. Offline only. It covers the metadata, the denylist, notebook parsing,
  resume wiring (a real tiny checkpoint is restored, then resumed), the ledger gate, and the train and summary cells
  on CPU.
