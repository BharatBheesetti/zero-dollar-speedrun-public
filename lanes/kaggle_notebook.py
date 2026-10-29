"""Cell sources for the Kaggle notebook that runs trainer/train_t4.py on 2xT4.

lanes/kaggle_push.py calls build(cfg) and writes build/kernel/kernel.ipynb. The run config (mode, steps, data
shards, resume source, time cap) is embedded in the first code cell, so changing the mode needs a new kernel push,
not a new dataset version. Kept as Python so the notebook is reviewable in a diff.

What the notebook does:
 1. untar zds_code.tar.gz (trainer/ + scripts/) from the attached private dataset into scratch;
 2. download FineWeb10B GPT-2 shards with scripts/get_data.py into /kaggle/tmp (not /kaggle/working: the
    20 GB output limit is for checkpoints and logs);
 3. if a previous kernel is attached (kernel_sources, --resume-from), copy its newest committed checkpoint and
    its log.jsonl into /kaggle/working/run/ so `--resume` continues it;
 4. torchrun --nproc_per_node=<gpus> trainer/train_t4.py --provider kaggle --resume --ckpt-every-min 20
    --time-budget-min <what is left of the session minus a margin>, with a watchdog that SIGTERMs the trainer
    (it checkpoints on SIGTERM) before Kaggle's hard cap, because a session killed at the cap saves no output;
 5. summary.json: steps, tokens/s, peak GPU memory (nvidia-smi samples), fp16 overflow skips, val loss, reason.
"""
from __future__ import annotations

import json

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


md("""# Zero-Dollar Speedrun: GPT-2 (124M) on Kaggle 2xT4

Pushed by `lanes/kaggle_push.py`. Code comes from the attached private dataset (`zds_code.tar.gz`), data from
Hugging Face (`kjj0/fineweb10B-gpt2`), the checkpoint of the previous session from the attached kernel output.
Everything that must survive the session goes to `/kaggle/working/run/` (checkpoints, `log.jsonl`) and
`/kaggle/working/summary.json`.""")

code("""import glob, json, os, shutil, signal, statistics, subprocess, sys, tarfile, time
T0 = time.time()
CFG = json.loads(__CFG_JSON__)
print(json.dumps(CFG, indent=1))
WORK = "/kaggle/working"
INPUT = CFG.get("input_root", "/kaggle/input")      # datasets + attached kernel outputs, read-only
RUN = os.path.join(WORK, "run")
CKPT = os.path.join(RUN, "ckpt")
SCRATCH = CFG.get("scratch", "/kaggle/tmp")
SRC = os.path.join(SCRATCH, "zds_src")
DATA = os.path.join(SCRATCH, "fineweb10B")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout or "no nvidia-smi")
import torch
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "x", N_GPU, "|", shutil.disk_usage(SCRATCH))
assert torch.cuda.is_available(), "GPU not attached: enable_gpu/machine_shape in kernel-metadata.json"
os.chdir(WORK)""")

md("## 1. Unpack the trainer from the attached dataset")
UNPACK_SRC = """# Kaggle auto-extracts uploaded archives: the dataset shows up either as zds_code.tar.gz or as a zds_code/ dir.
tarballs = sorted(glob.glob(os.path.join(INPUT, "**", "zds_code.tar.gz"), recursive=True))
extracted = sorted(glob.glob(os.path.join(INPUT, "**", "zds_code", "trainer", "train_t4.py"), recursive=True))
assert tarballs or extracted, ("zds code not found under /kaggle/input (neither zds_code.tar.gz nor zds_code/): "
                               "is the dataset attached (dataset_sources)? " + repr(glob.glob(os.path.join(INPUT, "**"), recursive=True)[:40]))
shutil.rmtree(SRC, ignore_errors=True)
if tarballs:
    os.makedirs(SRC)
    with tarfile.open(tarballs[0]) as tf:
        try:
            tf.extractall(SRC, filter="data")
        except TypeError:                  # python < 3.11.4 has no filter=
            tf.extractall(SRC)
    code_root = os.path.dirname(tarballs[0])
else:
    code_dir = os.path.dirname(os.path.dirname(extracted[0]))
    shutil.copytree(code_dir, SRC)
    code_root = os.path.dirname(code_dir)
manifest = glob.glob(os.path.join(code_root, "manifest.json"))
if manifest:
    print("code git sha:", json.load(open(manifest[0])).get("git_sha"))
print(sorted(os.listdir(SRC)), os.listdir(os.path.join(SRC, "trainer")))"""
code(UNPACK_SRC)

md("## 2. FineWeb10B shards (val + N train) into scratch")
code("""t = time.time()
subprocess.run([sys.executable, "-u", "scripts/get_data.py", str(CFG["train_shards"]), "--dir", DATA],
               cwd=SRC, check=True)
print(f"data ready in {time.time() - t:.0f} s:", sorted(os.listdir(DATA)))""")

md("## 3. Resume: copy the newest committed checkpoint of the attached previous session")
code("""def committed_ckpts(root):
    out = []
    for js in glob.glob(os.path.join(root, "**", "ckpt_*.json"), recursive=True):
        pt = js[:-5] + ".pt"
        try:
            meta = json.load(open(js))
            if os.path.getsize(pt) == meta["bytes"]:
                out.append((meta["next_step"], js, pt, meta))
        except (OSError, ValueError, KeyError):
            pass
    return sorted(out, reverse=True, key=lambda r: r[0])

RESUMED_FROM = None
if CFG.get("resume_from"):
    prev = committed_ckpts(INPUT)
    assert prev, (f"resume_from={CFG['resume_from']} but no committed ckpt_*.json/.pt pair under {INPUT}; "
                  "did the previous session finish and save its output? Refusing to restart from step 0.")
    step, js, pt, meta = prev[0]
    have = committed_ckpts(CKPT)
    if have and have[0][0] >= step:
        print(f"working dir already holds ckpt at next_step {have[0][0]} >= {step}; not copying")
    else:
        t = time.time()
        shutil.copy(pt, os.path.join(CKPT, os.path.basename(pt)))
        shutil.copy(js, os.path.join(CKPT, os.path.basename(js)))      # sidecar last: commits the pair
        print(f"copied {pt} (next_step {step}, {meta['bytes'] / 2**30:.2f} GiB) in {time.time() - t:.0f} s")
    prev_log = os.path.join(os.path.dirname(os.path.dirname(js)), "log.jsonl")
    if os.path.exists(prev_log) and not os.path.exists(os.path.join(RUN, "log.jsonl")):
        shutil.copy(prev_log, os.path.join(RUN, "log.jsonl"))         # one continuous log across sessions
        print("continuing log", prev_log)
    RESUMED_FROM = {"source": CFG["resume_from"], "path": pt, "next_step": step, "segments": meta.get("segments")}
else:
    print("fresh run (no resume_from)")""")

md("## 4. Train: torchrun on every GPU, self-stop before the session cap")
code("""CAP_S = CFG["session_cap_h"] * 3600                      # min(12 h, push timeout)
DEADLINE = T0 + CAP_S - CFG["margin_min"] * 60           # trainer checkpoints and exits by here
SIGTERM_AT = T0 + CAP_S - CFG["sigterm_min"] * 60        # watchdog, if a step or the final val overruns
KILL_AT = T0 + CAP_S - CFG["kill_min"] * 60
budget_min = (DEADLINE - time.time()) / 60
assert budget_min > 5, f"only {budget_min:.1f} min left for training"
cmd = ["torchrun", "--standalone", f"--nproc_per_node={max(N_GPU, 1)}", "trainer/train_t4.py",
       "--device", CFG.get("device", "cuda"), "--provider", "kaggle", "--resume",
       "--data-dir", DATA, "--out-dir", RUN, "--ckpt-dir", CKPT,
       "--ckpt-every-min", str(CFG["ckpt_every_min"]), "--time-budget-min", f"{budget_min:.1f}"]
if CFG.get("stop_after_step", -1) >= 0:
    cmd += ["--stop-after-step", str(CFG["stop_after_step"])]
if CFG.get("val_every"):
    cmd += ["--val-every", str(CFG["val_every"])]
cmd += [str(x) for x in CFG.get("trainer_args", [])]
print(" ".join(cmd), flush=True)
env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
out_path = os.path.join(RUN, f"train_{time.strftime('%Y%m%d-%H%M%S')}.out")
out = open(out_path, "w")
proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=out, stderr=subprocess.STDOUT, text=True,
                        start_new_session=True)
peak_mib, last_line, termed = {}, None, False
while proc.poll() is None:
    time.sleep(CFG.get("poll_s", 20))
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        for row in q.strip().splitlines():
            i, used = (int(x) for x in row.split(","))
            peak_mib[i] = max(peak_mib.get(i, 0), used)
    except (OSError, ValueError):
        pass
    lines = open(out_path).read().splitlines()
    if lines and lines[-1] != last_line:
        last_line = lines[-1]
        print(f"[{(time.time() - T0) / 3600:5.2f} h] {last_line}", flush=True)
    if time.time() > SIGTERM_AT and not termed:
        # signal the workers, not torchrun: torchrun would SIGKILL them 30 s later, possibly mid-checkpoint
        try:
            kids = [int(x) for x in open(f"/proc/{proc.pid}/task/{proc.pid}/children").read().split()]
        except OSError:
            kids = []
        print(f"[watchdog] SIGTERM to workers {kids or [proc.pid]}: they checkpoint after the current step", flush=True)
        for pid in kids or [proc.pid]:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        termed = True
    if time.time() > KILL_AT:
        print("[watchdog] SIGKILL: keeping the last committed checkpoint", flush=True)
        os.killpg(proc.pid, signal.SIGKILL)
        break
RC = proc.wait()
out.close()
print(f"trainer exit {RC}; last 25 lines:")
print("\\n".join(open(out_path).read().splitlines()[-25:]))""")

md("## 5. Summary")
code("""recs = []
if os.path.exists(os.path.join(RUN, "log.jsonl")):
    for line in open(os.path.join(RUN, "log.jsonl")):
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
session = [r for r in recs if r.get("time", 0) >= T0]
steps = [r for r in session if "train_loss" in r]
vals = [r for r in recs if r.get("event") == "val"]
exits = [r for r in session if r.get("event") == "exit"]
tps = [r["tokens_per_s"] for r in steps[10:]] or [r["tokens_per_s"] for r in steps]
ck = committed_ckpts(CKPT)
summary = {
    "mode": CFG["mode"], "kernel": CFG["kernel"], "resumed_from": RESUMED_FROM, "trainer_exit": RC,
    "watchdog_sigterm": termed, "elapsed_h": round((time.time() - T0) / 3600, 3), "n_gpu": N_GPU,
    "gpu": torch.cuda.get_device_name(0) if N_GPU else "cpu", "torch": torch.__version__,
    "first_step": steps[0]["step"] if steps else None, "last_step": steps[-1]["step"] if steps else None,
    "steps_this_session": len(steps), "tokens_per_s_median": statistics.median(tps) if tps else None,
    "step_s_median": statistics.median([r["step_s"] for r in steps]) if steps else None,
    "overflow_skips_total": steps[-1]["overflow_skips"] if steps else None,
    "overflow_skips_this_session": sum(bool(r.get("skipped")) for r in steps),
    "peak_gpu_mem_mib_nvidia_smi": peak_mib, "last_train_loss": steps[-1]["train_loss"] if steps else None,
    "last_val": vals[-1] if vals else None, "exit_reason": exits[-1]["reason"] if exits else None,
    "done": any(r.get("event") == "checkpoint" and r.get("reason") == "final" for r in recs),
    "newest_ckpt_next_step": ck[0][0] if ck else None, "gpu_seconds": steps[-1]["gpu_seconds"] if steps else None,
    "gpu_seconds_total": steps[-1].get("gpu_seconds_total") if steps else None,   # incl. val + checkpoints
    "val_set_is_upstream": next((r.get("val_set_is_upstream") for r in recs if r.get("event") == "start"), None),
    "val_vs_upstream": [{k: r.get(k) for k in ("step", "val_loss", "upstream_val_loss", "val_minus_upstream",
                                               "schedule_same_as_upstream")} for r in vals if "upstream_val_loss" in r],
}
json.dump(summary, open(os.path.join(WORK, "summary.json"), "w"), indent=1)
shutil.rmtree(SRC, ignore_errors=True)
print(json.dumps(summary, indent=1))
assert RC == 0 or termed, f"trainer failed with exit code {RC}"
""")


# ---------------------------------------------------------------------------------------------
# Re-score notebook (review M3 finding 4): score the saved final weights again, outside the training run.
# ---------------------------------------------------------------------------------------------
RESCORE_CELLS: list[tuple[str, str]] = [("markdown", """# Zero-Dollar Speedrun: re-score the saved final weights

Pushed by `lanes/kaggle_push.py rescore`. Loads the final checkpoint from the attached training session's output
and scores it on the upstream val set in a fresh process with `scripts/rescore.py` (the trainer's own model and
`validate`): once in fp16 autocast as in training, once in fp32. Also hashes the checkpoints that join the two
sessions. Writes `/kaggle/working/rescore.json`.""")]
RESCORE_CELLS.append(("code", """import glob, hashlib, json, os, shutil, signal, subprocess, sys, tarfile, time
T0 = time.time()
CFG = json.loads(__CFG_JSON__)
print(json.dumps(CFG, indent=1))
WORK = "/kaggle/working"
INPUT = CFG.get("input_root", "/kaggle/input")
SCRATCH = CFG.get("scratch", "/kaggle/tmp")
SRC = os.path.join(SCRATCH, "zds_src")
DATA = os.path.join(SCRATCH, "fineweb10B")
os.makedirs(DATA, exist_ok=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout or "no nvidia-smi")
import torch
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "x", N_GPU)
assert torch.cuda.is_available(), "GPU not attached"
os.chdir(WORK)"""))
RESCORE_CELLS.append(("code", UNPACK_SRC))
RESCORE_CELLS.append(("code", """t = time.time()
subprocess.run([sys.executable, "-u", "scripts/get_data.py", "0", "--dir", DATA], cwd=SRC, check=True)   # val shard only
print(f"val shard ready in {time.time() - t:.0f} s:", sorted(os.listdir(DATA)))"""))
RESCORE_CELLS.append(("code", """def find(rel):
    \"\"\"An attached kernel output file by its tail path, e.g. zds-full-s2/run/ckpt/ckpt_0001451.pt.\"\"\"
    hits = [p for p in glob.glob(os.path.join(INPUT, "**", os.path.basename(rel)), recursive=True) if p.endswith("/" + rel)]
    return sorted(hits)[0] if hits else None

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()

CKPT = find(CFG["ckpt"])
assert CKPT, f"{CFG['ckpt']} not found under {INPUT}: " + repr(glob.glob(os.path.join(INPUT, "**"), recursive=True)[:60])
side = json.load(open(CKPT[:-3] + ".json"))
assert os.path.getsize(CKPT) == side["bytes"], "checkpoint size does not match its sidecar (not committed)"
print("final checkpoint:", CKPT, side["bytes"], "bytes, step", side["step"], side["reason"])
HASHES = {}
for rel in CFG.get("hash_files", []):
    p = find(rel)
    if p is None:
        HASHES[rel] = None
        continue
    t = time.time()
    js = p[:-3] + ".json"
    HASHES[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p),
                   "sidecar": json.load(open(js)) if os.path.exists(js) else None}
    print(rel, HASHES[rel]["sha256"], f"{time.time() - t:.0f} s")"""))
RESCORE_CELLS.append(("code", """CAP_S = CFG["session_cap_h"] * 3600

def rescore(amp, extra, margin_min):
    out_json = os.path.join(WORK, f"rescore_{amp}.json")
    out_log = os.path.join(WORK, f"rescore_{amp}.out")
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={max(N_GPU, 1)}", "scripts/rescore.py", "--ckpt", CKPT,
           "--data-dir", DATA, "--device", CFG.get("device", "cuda"), "--amp", amp, "--expect", repr(CFG["expect"]),
           "--out", out_json] + extra + [str(x) for x in CFG.get("rescore_args", [])]
    print(" ".join(cmd), flush=True)
    timeout = CAP_S - (time.time() - T0) - margin_min * 60
    with open(out_log, "w") as log:
        proc = subprocess.Popen(cmd, cwd=SRC, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                env=dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4"))
        try:
            rc = proc.wait(timeout=max(timeout, 60))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = "timeout"
    print(f"[{amp}] exit {rc}; tail:\\n" + "\\n".join(open(out_log).read().splitlines()[-12:]), flush=True)
    return rc, (json.load(open(out_json)) if os.path.exists(out_json) else None)

RC16, R16 = rescore("fp16", ["--hash", "--save-model", os.path.join(WORK, CFG["model_file"])], margin_min=25)
RC32, R32 = rescore("fp32", [], margin_min=8)"""))
RESCORE_CELLS.append(("code", """man = glob.glob(os.path.join(INPUT, "**", "manifest.json"), recursive=True)
summary = {
    "kernel": CFG["kernel"], "sources": CFG["sources"], "ckpt": CFG["ckpt"], "expect": CFG["expect"],
    "code_git_sha": json.load(open(man[0])).get("git_sha") if man else None,
    "fp16": {"exit": RC16, "result": R16}, "fp32": {"exit": RC32, "result": R32}, "hashes": HASHES,
    "ckpt_sidecar": side, "n_gpu": N_GPU, "gpu": torch.cuda.get_device_name(0) if N_GPU else "cpu", "torch": torch.__version__,
    "elapsed_h": round((time.time() - T0) / 3600, 3),
}
json.dump(summary, open(os.path.join(WORK, "rescore.json"), "w"), indent=1)
shutil.rmtree(SRC, ignore_errors=True)
print(json.dumps({k: v for k, v in summary.items() if k not in ("ckpt_sidecar",)}, indent=1)[:6000])
assert RC16 == 0, f"fp16 re-score failed: {RC16}"
"""))


def build(cfg: dict, cells_src: list[tuple[str, str]] | None = None) -> dict:
    cfg_src = repr(json.dumps(cfg, sort_keys=True))
    cells = []
    for kind, src in (CELLS if cells_src is None else cells_src):
        src = src.replace("__CFG_JSON__", cfg_src)
        c = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}
