"""Crash-proof resume: kill -9 at a random step, resume, and reproduce the uninterrupted run's losses."""
import os
import random
import signal
import subprocess
import time

from conftest import env, read_log, train_losses, trainer_cmd


def test_kill9_resume_matches_uninterrupted(run_a, tmp_path):
    _, log_a = run_a
    ref = {s: v[0] for s, v in train_losses(log_a).items()}
    seed = int(os.environ.get("ZDS_KILL_SEED", time.time_ns() % 1_000_000))
    kill_at = random.Random(seed).randint(15, 45)
    print(f"kill seed {seed}: kill -9 after step {kill_at}")
    out = tmp_path / "run_b"
    log = out / "log.jsonl"
    cmd = trainer_cmd(out, "--ckpt-every-steps", "5", "--resume")
    proc = subprocess.Popen(cmd, env=env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 600
    killed_after = None
    while time.time() < deadline and proc.poll() is None:
        if log.exists():
            steps = [r["step"] for r in read_log(log) if "train_loss" in r]
            if steps and steps[-1] >= kill_at:
                os.kill(proc.pid, signal.SIGKILL)
                killed_after = steps[-1]
                break
        time.sleep(0.01)
    proc.wait()
    assert killed_after is not None, f"run B never reached step {kill_at}: {proc.stderr.read().decode()[-2000:]}"
    assert proc.returncode == -signal.SIGKILL
    # resume to the end
    subprocess.run(cmd, env=env(), check=True, timeout=600)
    recs = read_log(log)
    resumes = [r for r in recs if r.get("event") == "resume"]
    assert len(resumes) == 1 and 0 < resumes[0]["next_step"] <= killed_after + 1, resumes
    got = train_losses(recs)
    assert sorted(got) == list(range(60))
    worst = max(abs(v - ref[s]) for s, vals in got.items() for v in vals)
    assert worst <= 1e-6, f"max |loss_B - loss_A| = {worst}"
    # steps after the resume point really were recomputed, and match
    resumed_from = resumes[0]["next_step"]
    assert all(len(got[s]) >= 1 for s in range(resumed_from, 60))
    val_a = [r["val_loss"] for r in log_a if r.get("event") == "val"]
    val_b = [r["val_loss"] for r in recs if r.get("event") == "val"]
    assert val_b and abs(val_b[-1] - val_a[-1]) <= 1e-6, (val_a, val_b)
