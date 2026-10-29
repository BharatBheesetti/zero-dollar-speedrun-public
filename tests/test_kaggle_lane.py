"""Kaggle lane (lanes/kaggle_push.py), offline only: nothing here contacts Kaggle or reads real credentials."""
import ast
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lanes"))
import kaggle_push as kp  # noqa: E402

LIVE = [["dataset"], ["smoke"], ["full"], ["chain", "--session", "2"], ["status"], ["pull"], ["quota"],
        ["probe", "--accelerator", "TpuV5E8"], ["rescore"], ["files", "zds-full-s2"], ["meta", "zds-full-s2", "--dir", "x"]]


@pytest.fixture(autouse=True)
def no_creds_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("ZDS_KAGGLE_ENV_FILE", str(tmp_path / "no-such-env"))
    monkeypatch.delenv("ZDS_KAGGLE_LEDGER_OK", raising=False)

    def boom(*a, **k):
        raise AssertionError("tried to call the kaggle CLI")
    monkeypatch.setattr(kp, "run_kaggle", boom)


@pytest.fixture
def built(tmp_path):
    b = tmp_path / "build"
    kp.main(["--build-dir", str(b), "dryrun"])
    return b


def code_cells(nb):
    return [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"]


def test_dryrun_renders_valid_metadata(built):
    dm = json.loads((built / "dataset" / "dataset-metadata.json").read_text())
    assert dm["id"] == "USERNAME/zds-code" and len(dm["licenses"]) == 1 and 6 <= len(dm["title"]) <= 50
    for mode, slug in (("smoke", "zds-smoke-s1"), ("full", "zds-full-s1"), ("chain", "zds-full-s2")):
        km = json.loads((built / f"kernel-{mode}" / "kernel-metadata.json").read_text())
        assert km["id"] == f"USERNAME/{slug}" and km["title"] == slug
        assert km["is_private"] == "true" and km["enable_gpu"] == "true" and km["enable_internet"] == "true"
        assert km["machine_shape"] == "NvidiaTeslaT4"
        assert km["dataset_sources"] == ["USERNAME/zds-code"]
        assert km["kernel_type"] == "notebook" and km["code_file"] == "kernel.ipynb"


def test_denylist_holds(tmp_path, built):
    names = tarfile.open(built / "dataset" / kp.TARBALL).getnames()
    assert "trainer/train_t4.py" in names and "scripts/get_data.py" in names
    assert all(n.split("/")[0] in kp.PACK_DIRS for n in names), names
    assert not any(part in kp.DENY_DIRS for n in names for part in n.split("/")[:-1])
    for rel in (".venv/bin/python", "data/fineweb10B/fineweb_val_000000.bin", "ckpt/ckpt_0000010.pt",
                "out/run/log.jsonl", "build/dataset/x.tar.gz", "trainer/__pycache__/t.pyc", "scripts/.env",
                "scripts/kaggle.json", "trainer/env", "trainer/w.pt"):
        assert kp.deny_reason(ROOT / rel, rel=rel), rel
    assert kp.deny_reason(Path("~/.config/zds/env").expanduser())       # path check only, never opened
    link = tmp_path / "innocent.py"
    link.symlink_to(Path("~/.config/zds-test-nonexistent.py").expanduser())
    assert kp.deny_reason(link, rel="innocent.py") == "under ~/.config"
    assert kp.deny_reason(ROOT / "trainer" / "train_t4.py", rel="trainer/train_t4.py") is None
    # --add extras go through the same denylist, file by file
    extra = tmp_path / "extra"
    (extra / "data").mkdir(parents=True)
    (extra / "ok.py").write_text("x = 1\n")
    (extra / ".env").write_text("SECRET=1\n")
    (extra / "data" / "shard.npy").write_text("no")
    (extra / "id_rsa").write_text("no")
    b2 = tmp_path / "b2"
    kp.main(["--build-dir", str(b2), "pack", "--add", str(extra)])
    names = tarfile.open(b2 / "dataset" / kp.TARBALL).getnames()
    assert "extra/ok.py" in names
    assert not any(n.endswith((".env", "shard.npy", "id_rsa")) for n in names), names
    skipped = json.loads((b2 / "dataset" / "manifest.json").read_text())["skipped_by_denylist"]
    assert {"extra/.env", "extra/data/shard.npy", "extra/id_rsa"} <= {s["path"] for s in skipped}


def test_notebook_json_parses(built):
    for mode in ("smoke", "full", "chain"):
        nb = json.loads((built / f"kernel-{mode}" / "kernel.ipynb").read_text())
        assert nb["nbformat"] == 4
        cells = code_cells(nb)
        for src in cells:
            ast.parse(src)
        allsrc = "\n".join(cells)
        for need in ("torchrun", "--nproc_per_node", "trainer/train_t4.py", '"--provider", "kaggle"', '"--resume"',
                     "--time-budget-min", "--ckpt-every-min", "/kaggle/working", "/kaggle/tmp", "scripts/get_data.py"):
            assert need in allsrc, (mode, need)
        cfg_line = cells[0].splitlines()[2]
        assert cfg_line.startswith("CFG = json.loads(")
        cfg = json.loads(ast.literal_eval(cfg_line[len("CFG = json.loads("):-1]))
        assert cfg == json.loads((built / f"kernel-{mode}" / "run_config.json").read_text())
        assert cfg["margin_min"] == 20 and cfg["ckpt_every_min"] == 20 and cfg["session_cap_h"] <= 12
        if mode == "smoke":
            assert cfg["stop_after_step"] == 299 and cfg["session_cap_h"] <= 3     # 300 steps, bounded quota
            # val at 125 and 250 on upstream's exact schedule: parity check against upstream's 3.9427 at 250
            assert cfg["val_every"] == 125 and cfg["trainer_args"][:2] == ["--num-iters", "1390"]
        else:
            assert cfg["stop_after_step"] == -1 and cfg["train_shards"] >= 8      # 1450 steps = 760.2M tokens
            assert "--num-iters" not in cfg["trainer_args"]


def test_resume_from_wiring(built, tmp_path):
    chain = json.loads((built / "kernel-chain" / "kernel-metadata.json").read_text())
    cfg = json.loads((built / "kernel-chain" / "run_config.json").read_text())
    assert chain["kernel_sources"] == ["USERNAME/zds-full-s1"] == [cfg["resume_from"]]
    for mode in ("smoke", "full"):
        assert json.loads((built / f"kernel-{mode}" / "kernel-metadata.json").read_text())["kernel_sources"] == []
    b = tmp_path / "b"
    kp.main(["--build-dir", str(b), "kernel", "--mode", "full", "--session", "3", "--resume-from", "someone/zds-smoke-s1"])
    km = json.loads((b / "kernel" / "kernel-metadata.json").read_text())
    assert km["id"] == "USERNAME/zds-full-s3" and km["kernel_sources"] == ["someone/zds-smoke-s1"]
    with pytest.raises(SystemExit):
        kp.main(["--build-dir", str(b), "kernel", "--mode", "chain", "--session", "1"])
    with pytest.raises(SystemExit):
        kp.main(["--build-dir", str(b), "kernel", "--mode", "full", "--resume-from", "no-slash"])


def test_resume_cell_restores_a_real_checkpoint(built, tmp_path):
    """Make a real tiny checkpoint, lay it out like an attached kernel output, run the notebook's resume cell,
    then check the trainer continues from it."""
    prev = tmp_path / "input" / "zds-full-s1" / "run"
    trainer = [sys.executable, str(ROOT / "trainer" / "train_t4.py"), "--tiny", "--synthetic", "--device", "cpu",
               "--threads", "2", "--ckpt-every-min", "1000000"]
    env = dict(os.environ, OMP_NUM_THREADS="2")
    subprocess.run(trainer + ["--out-dir", str(prev), "--stop-after-step", "4"], check=True, env=env, timeout=600)
    (prev / "ckpt" / "ckpt_0000999.pt").write_bytes(b"torn write")                   # uncommitted: must be ignored
    (prev / "ckpt" / "ckpt_0000999.json").write_text(json.dumps({"next_step": 999, "bytes": 1}))
    (tmp_path / "input" / "zds-code").mkdir()

    work = tmp_path / "working" / "run"
    nb = json.loads((built / "kernel-chain" / "kernel.ipynb").read_text())
    resume_src = next(s for s in code_cells(nb) if "def committed_ckpts" in s)
    ns = {"CFG": {"resume_from": "USERNAME/zds-full-s1"}, "INPUT": str(tmp_path / "input"),
          "RUN": str(work), "CKPT": str(work / "ckpt")}
    os.makedirs(ns["CKPT"])
    exec("import glob, json, os, shutil, time\n" + resume_src, ns)
    assert ns["RESUMED_FROM"]["next_step"] == 5
    assert sorted(os.listdir(ns["CKPT"])) == ["ckpt_0000005.json", "ckpt_0000005.pt"]
    assert (work / "log.jsonl").exists()

    subprocess.run(trainer + ["--out-dir", str(work), "--resume", "--provider", "kaggle", "--stop-after-step", "6"],
                   check=True, env=env, timeout=600)
    recs = [json.loads(line) for line in open(work / "log.jsonl")]
    resume = [r for r in recs if r.get("event") == "resume"]
    assert resume and resume[-1]["next_step"] == 5
    assert [r["step"] for r in recs if "train_loss" in r] == [0, 1, 2, 3, 4, 5, 6]     # one continuous log

    # an attached source with no committed checkpoint must refuse rather than silently restart at step 0
    empty = {"CFG": {"resume_from": "x/y"}, "INPUT": str(tmp_path / "input" / "zds-code"),
             "RUN": str(tmp_path / "w2"), "CKPT": str(tmp_path / "w2" / "ckpt")}
    os.makedirs(empty["CKPT"])
    with pytest.raises(AssertionError, match="Refusing to restart"):
        exec("import glob, json, os, shutil, time\n" + resume_src, empty)


@pytest.mark.parametrize("argv", LIVE, ids=lambda a: a[0])
def test_live_commands_refuse_without_ledger(argv, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        kp.main(["--build-dir", str(tmp_path / "b"), *argv])
    msg = str(e.value)
    assert "ZDS_KAGGLE_LEDGER_OK=1" in msg and "GPU-quota ledger" in msg and "ZDS_LEDGER" in msg
    assert not (tmp_path / "b").exists()                     # refused before packing anything


def test_ledger_ok_without_creds_still_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("ZDS_KAGGLE_LEDGER_OK", "1")
    with pytest.raises(SystemExit, match="credentials missing"):
        kp.main(["--build-dir", str(tmp_path / "b"), "status"])


def test_train_and_summary_cells_run_end_to_end_on_cpu(built, tmp_path, monkeypatch):
    """The notebook's resume, train (real torchrun command line) and summary cells, on CPU with the tiny preset."""
    import shutil
    import time

    import torch
    src = tmp_path / "src"                                   # the summary cell deletes SRC: never point it at the repo
    shutil.copytree(ROOT / "trainer", src / "trainer", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    work = tmp_path / "working"
    nb = json.loads((built / "kernel-full" / "kernel.ipynb").read_text())
    cells = code_cells(nb)
    cfg = json.loads((built / "kernel-full" / "run_config.json").read_text())
    cfg.update(device="cpu", poll_s=1, stop_after_step=5, val_every=0,
               trainer_args=["--tiny", "--synthetic", "--threads", "2"])
    ns = {"CFG": cfg, "T0": time.time(), "N_GPU": 0, "torch": torch, "SRC": str(src), "DATA": str(tmp_path / "d"),
          "WORK": str(work), "RUN": str(work / "run"), "CKPT": str(work / "run" / "ckpt"), "INPUT": str(tmp_path / "in")}
    os.makedirs(ns["CKPT"])
    pre = "import glob, json, os, shutil, signal, statistics, subprocess, sys, tarfile, time\n"
    for marker in ("def committed_ckpts", "CAP_S =", "summary = {"):
        exec(pre + next(c for c in cells if marker in c), ns)
    assert ns["RC"] == 0
    s = json.loads((work / "summary.json").read_text())
    assert s["first_step"] == 0 and s["last_step"] == 5 and s["steps_this_session"] == 6
    assert s["exit_reason"] == "stop_after_step" and s["newest_ckpt_next_step"] == 6
    assert s["tokens_per_s_median"] > 0 and s["overflow_skips_this_session"] >= 0
    start = [json.loads(line) for line in open(work / "run" / "log.jsonl")][0]
    assert start["event"] == "start" and start["provider"] == "kaggle"
    assert not src.exists()


def test_rescore_kernel_end_to_end_on_cpu(tmp_path, monkeypatch, run_a):
    """The eval-only kernel: metadata attaches both training sessions; its cells find the final checkpoint in the
    attached output, hash the session-joining checkpoints, and re-score (fp16 + fp32) with scripts/rescore.py."""
    import shutil
    import time

    import torch
    b = tmp_path / "b"
    kp.main(["--build-dir", str(b), "rescore-render"])
    d = b / "kernel-rescore"
    km = json.loads((d / "kernel-metadata.json").read_text())
    assert km["id"] == "USERNAME/zds-rescore-s2" and km["is_private"] == "true" and km["machine_shape"] == "NvidiaTeslaT4"
    assert km["kernel_sources"] == ["USERNAME/zds-full-s2", "USERNAME/zds-full-s1"]
    cells = code_cells(json.loads((d / "kernel.ipynb").read_text()))
    cfg = json.loads((d / "run_config.json").read_text())
    assert cfg["ckpt"].endswith("ckpt_0001451.pt") and cfg["expect"] == 3.2698315009167063

    out, log = run_a                                          # a tiny finished run stands in for zds-full-s2
    val = [r for r in log if r.get("event") == "val"][-1]
    inp = tmp_path / "in" / "notebooks" / "USERNAME"
    s2 = inp / "zds-full-s2" / "run" / "ckpt"
    s2.mkdir(parents=True)
    for f in (out / "ckpt").iterdir():
        shutil.copy(f, s2 / f.name)
    ck = sorted(p.name for p in s2.glob("*.pt"))[-1]
    src = tmp_path / "src"
    for sub in ("trainer", "scripts"):
        shutil.copytree(ROOT / sub, src / sub, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    work = tmp_path / "working"
    work.mkdir()
    cfg.update(device="cpu", ckpt=f"zds-full-s2/run/ckpt/{ck}", expect=val["val_loss"],
               hash_files=[f"zds-full-s2/run/ckpt/{ck}", "zds-full-s1/run/ckpt/missing.pt"],
               rescore_args=["--tiny", "--threads", "2"])
    ns = {"CFG": cfg, "T0": time.time(), "N_GPU": 0, "torch": torch, "SRC": str(src), "WORK": str(work),
          "DATA": str(tmp_path / "d"), "INPUT": str(tmp_path / "in")}
    pre = "import glob, hashlib, json, os, shutil, signal, subprocess, sys, tarfile, time\n"
    for marker in ("def find", "CAP_S =", "man = glob"):
        exec(pre + next(c for c in cells if marker in c), ns)
    s = json.loads((work / "rescore.json").read_text())
    assert s["fp16"]["exit"] == 0 and s["fp32"]["exit"] == 0
    assert s["fp32"]["result"]["passes"][0]["val_loss"] == val["val_loss"]          # CPU fp32: bit-identical
    assert abs(s["fp16"]["result"]["passes"][0]["val_loss"] - val["val_loss"]) < 1e-2
    assert len(s["fp16"]["result"]["ckpt_sha256"]) == 64 and (work / cfg["model_file"]).exists()
    assert s["hashes"][f"zds-full-s2/run/ckpt/{ck}"]["sha256"] == s["fp16"]["result"]["ckpt_sha256"]
    assert s["hashes"]["zds-full-s1/run/ckpt/missing.pt"] is None
    assert not src.exists()


def test_creds_fallback_to_kaggle_json(monkeypatch, tmp_path):
    """No ZDS_KAGGLE_ENV_FILE and no ~/.config/zds/env: the environment, then ~/.kaggle/kaggle.json."""
    monkeypatch.delenv("ZDS_KAGGLE_ENV_FILE")
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert kp.kaggle_env(required=False) == (None, "USERNAME")
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text('{"username": "alice", "key": "k1"}')
    env, user = kp.kaggle_env()
    assert user == "alice" and env["KAGGLE_KEY"] == "k1"
    monkeypatch.setenv("KAGGLE_USERNAME", "bob")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t2")
    env, user = kp.kaggle_env()
    assert user == "bob" and env["KAGGLE_API_TOKEN"] == "t2"
