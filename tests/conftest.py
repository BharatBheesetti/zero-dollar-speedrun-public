"""Shared fixtures: the trainer module, and one uninterrupted tiny CPU run reused by several tests."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, "trainer", "train_t4.py")
sys.path.insert(0, os.path.join(ROOT, "trainer"))

# Tiny CPU config shared by every run in the suite: synthetic data, fp32, GradScaler on (exercises its state).
TINY_ARGS = ["--tiny", "--synthetic", "--device", "cpu", "--scaler", "on", "--threads",
             os.environ.get("ZDS_THREADS", "8"), "--ckpt-every-min", "1000000"]


def trainer_cmd(out_dir, *extra):
    return [sys.executable, TRAINER, *TINY_ARGS, "--out-dir", str(out_dir), *extra]


def env():
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", os.environ.get("ZDS_THREADS", "8"))
    return e


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def train_losses(records):
    """step -> train_loss; a re-run step (after a resume) must agree, so keep every value."""
    out = {}
    for r in records:
        if "train_loss" in r:
            out.setdefault(r["step"], []).append(r["train_loss"])
    return out


@pytest.fixture(scope="session")
def run_a(tmp_path_factory):
    """60 uninterrupted steps (the whole tiny schedule) plus the final validation."""
    out = tmp_path_factory.mktemp("run_a")
    subprocess.run(trainer_cmd(out), check=True, env=env(), timeout=600)
    return out, read_log(out / "log.jsonl")
