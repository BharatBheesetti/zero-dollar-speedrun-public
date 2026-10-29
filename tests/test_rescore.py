"""scripts/rescore.py: a fresh process scores a saved checkpoint and gets the training run's logged val loss."""
import json
import os
import runpy
import subprocess
import sys

import pytest
import torch

import train_t4 as T
from conftest import ROOT, env

RESCORE = os.path.join(ROOT, "scripts", "rescore.py")


def logged_final_val(run_a):
    out, log = run_a
    val = [r for r in log if r.get("event") == "val"][-1]
    ckpt = os.path.join(out, "ckpt", f"ckpt_{val['step'] + 1:07d}.pt")
    assert os.path.exists(ckpt)
    return val, ckpt


def test_rescore_matches_logged_val(run_a, tmp_path):
    val, ckpt = logged_final_val(run_a)
    res, model = tmp_path / "rescore.json", tmp_path / "model.pt"
    r = subprocess.run([sys.executable, RESCORE, "--ckpt", ckpt, "--tiny", "--device", "cpu", "--amp", "fp32",
                        "--expect", str(val["val_loss"]), "--hash", "--save-model", str(model), "--out", str(res)],
                       env=env(), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(res.read_text())
    # same weights, same code, same CPU: bit-identical
    assert out["passes"][0]["val_loss"] == val["val_loss"], (out["passes"], val["val_loss"])
    assert out["val_step"] == val["step"] and out["ckpt_done"] and out["sidecar"]["reason"] == "final"
    assert out["val_tokens_sha256"] == val["val_tokens_sha256"] and out["val_tokens"] == val["val_tokens"]
    assert len(out["ckpt_sha256"]) == 64 and out["trainer_sha256"] and out["params"] > 0
    # the model-only file holds the same weights as the checkpoint
    m = torch.load(model, map_location="cpu", weights_only=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert m["step"] == val["step"] and m["args"] == ck["args"]
    assert all(torch.equal(m["model"][k], v.float()) for k, v in ck["model"].items())


def test_rescore_two_ranks(run_a, tmp_path):
    """The Kaggle path: torchrun splits the val sequences across ranks and all-reduces the sums."""
    val, ckpt = logged_final_val(run_a)
    res = tmp_path / "rescore.json"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", RESCORE,
           "--ckpt", ckpt, "--tiny", "--device", "cpu", "--amp", "fp32", "--threads", "2", "--out", str(res)]
    r = subprocess.run(cmd, env=dict(env(), OMP_NUM_THREADS="2"), capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and "Address already in use" in r.stderr:
        pytest.skip("no free local port for torchrun")
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(res.read_text())
    assert out["world_size"] == 2
    assert abs(out["passes"][0]["val_loss"] - val["val_loss"]) <= 1e-6


def test_rescore_ignores_arch_flags(run_a, tmp_path):
    """Architecture comes from the checkpoint: a conflicting --num-iters on the command line has no effect."""
    val, ckpt = logged_final_val(run_a)
    passes = runpy.run_path(RESCORE)["main"](
        ["--ckpt", ckpt, "--tiny", "--device", "cpu", "--amp", "fp32", "--num-iters", "999", "--threads", "4"])
    assert passes[0]["val_loss"] == val["val_loss"]
    assert T.FULL["num_iters"] == 1450  # the trainer module was not mutated
