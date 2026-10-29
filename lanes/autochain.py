#!/usr/bin/env python
"""Detached watcher: polls zds-full-sN; when it completes, pulls it; if not done and it checkpointed,
logs the next session in the quota ledger at $ZDS_LEDGER (if set) and pushes chain --session N+1. Stops on done, error, or MAX_SESSIONS.
Log: out/autochain.log.   Usage: nohup setsid .venv/bin/python lanes/autochain.py 1 >/dev/null 2>&1 &"""
import json, os, subprocess, sys, time, datetime
from pathlib import Path
sys.path.insert(0, "lanes")
import kaggle_push as k
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "out" / "autochain.log"; LOG.parent.mkdir(exist_ok=True)
LEDGER = os.environ.get("ZDS_LEDGER")   # optional: quota ledger to append each chained session to
MAX_SESSIONS = 3
def log(m):
    with LOG.open("a") as f: f.write(f"{datetime.datetime.now():%m/%d %H:%M:%S} {m}\n")
def run(args, extra_env=None):
    env, _ = k.kaggle_env()
    r = subprocess.run(args, env=dict(env, **(extra_env or {})), capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout + r.stderr)
env, user = k.kaggle_env()
n = int(sys.argv[1])
log(f"watching zds-full-s{n}")
while True:
    slug = f"{user}/zds-full-s{n}"
    rc, out = run([".venv/bin/kaggle", "kernels", "status", slug])
    st = out.strip().splitlines()[-1] if out.strip() else "?"
    if not any(x in st.upper() for x in ("COMPLETE", "ERROR", "CANCEL")):
        time.sleep(300); continue
    log(f"s{n}: {st}")
    rc, out = run([".venv/bin/python", "lanes/kaggle_push.py", "pull", "--mode", "full", "--session", str(n)], {"ZDS_KAGGLE_LEDGER_OK": "1"})
    log(f"pull rc={rc}")
    try:
        s = json.loads((ROOT / "out" / "kaggle" / f"zds-full-s{n}" / "summary.json").read_text())
    except Exception as e:
        log(f"no summary ({e}); stopping"); break
    log(f"s{n} summary: last_step={s.get('last_step')} done={s.get('done')} exit={s.get('exit_reason')} "
        f"last_val={(s.get('last_val') or {}).get('val_loss')} elapsed_h={s.get('elapsed_h')} skips={s.get('overflow_skips_total')}")
    if s.get("done") or "ERROR" in st.upper() or "CANCEL" in st.upper() or not s.get("newest_ckpt_next_step"):
        log("stopping (done, error, cancel, or no checkpoint)"); break
    if n >= MAX_SESSIONS:
        log("MAX_SESSIONS reached; stopping"); break
    if LEDGER:
        with open(LEDGER, "a") as f:
            f.write(f"- ZDS full zds-full-s{n+1} (chain from s{n} at step {s.get('newest_ckpt_next_step')}) | speedrun | auto | launched {datetime.datetime.now():%m/%d %H:%M} IST | actual: tbd\n")
    rc, out = run([".venv/bin/python", "lanes/kaggle_push.py", "chain", "--session", str(n + 1), "--skip-dataset"], {"ZDS_KAGGLE_LEDGER_OK": "1"})
    log(f"chain s{n+1} rc={rc}: {out.strip().splitlines()[-1] if out.strip() else ''}")
    if rc != 0: break
    n += 1
    time.sleep(120)
log("autochain exit")
