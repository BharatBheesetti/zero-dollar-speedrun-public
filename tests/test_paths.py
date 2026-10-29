"""The 2-rank (gloo) DDP path and the fp16 + GradScaler path, on the tiny CPU config (review M1 finding 9)."""
import math
import os
import subprocess
import sys

import pytest

from conftest import TINY_ARGS, TRAINER, env, read_log, train_losses, trainer_cmd


def test_two_ranks_match_one_rank(run_a, tmp_path):
    threads = str(max(1, int(os.environ.get("ZDS_THREADS", "8")) // 2))
    args = list(TINY_ARGS)
    args[args.index("--threads") + 1] = threads
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", TRAINER, *args,
           "--out-dir", str(tmp_path), "--stop-after-step", "19"]
    r = subprocess.run(cmd, env=dict(env(), OMP_NUM_THREADS=threads), capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and "Address already in use" in r.stderr:
        pytest.skip("no free local port for torchrun")
    assert r.returncode == 0, r.stderr[-3000:]
    ref = {s: v[0] for s, v in train_losses(run_a[1]).items()}
    log = read_log(tmp_path / "log.jsonl")
    got = {s: v[0] for s, v in train_losses(log).items()}
    assert sorted(got) == list(range(20))
    worst = max(abs(got[s] - ref[s]) for s in got)
    assert worst <= 1e-5, worst
    ck = [x for x in log if x.get("event") == "checkpoint"][-1]
    assert ck["rank_weights_identical"] and ck["rank_weights_max_rel_diff"] == 0.0
    assert [x for x in log if x.get("event") == "start"][0]["world_size"] == 2


def test_fp16_gradscaler_path(run_a, tmp_path):
    r = subprocess.run(trainer_cmd(tmp_path, "--amp", "fp16", "--stop-after-step", "9"), env=env(),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    ref = {s: v[0] for s, v in train_losses(run_a[1]).items()}
    steps = [x for x in read_log(tmp_path / "log.jsonl") if "train_loss" in x]
    assert len(steps) == 10 and not any(x["skipped"] for x in steps) and steps[-1]["scale"] == 2.0 ** 16
    assert all(math.isfinite(x["train_loss"]) and abs(x["train_loss"] - ref[x["step"]]) < 0.02 for x in steps)
