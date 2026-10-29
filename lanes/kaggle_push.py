#!/usr/bin/env python3
"""Kaggle lane (job L1): trainer tarball -> private dataset -> 2xT4 kernel -> outputs back.

Offline (never contact Kaggle, need no credentials):
    lanes/kaggle_push.py pack                      # trainer/ + scripts/ -> build/dataset/ (zds_code.tar.gz + metadata)
    lanes/kaggle_push.py kernel --mode smoke       # render build/kernel/{kernel.ipynb,kernel-metadata.json}
    lanes/kaggle_push.py dryrun                    # pack + render smoke/full/chain + validate everything

Live (refuse unless ZDS_KAGGLE_LEDGER_OK=1, i.e. the session is logged in your GPU-quota ledger):
    lanes/kaggle_push.py dataset                   # upload build/dataset (create, or new version)
    lanes/kaggle_push.py smoke                     # pack + dataset + push zds-smoke-s1: 300 steps, tok/s, mem, overflows
    lanes/kaggle_push.py full [--resume-from U/K]  # pack + dataset + push zds-full-s1: train to the target (self-stops ~20 min before 12 h)
    lanes/kaggle_push.py chain --session 2         # push zds-full-s2 with zds-full-s1's output attached; resumes its ckpt
    lanes/kaggle_push.py status [--mode full --session 2]
    lanes/kaggle_push.py pull   [--mode full --session 2] [--with-ckpt]   # -> out/kaggle/<kernel-slug>/
    lanes/kaggle_push.py quota                     # weekly GPU/TPU hours left (`kaggle quota`)
    lanes/kaggle_push.py rescore                   # eval-only kernel zds-rescore-s2: re-score s2's final checkpoint
    lanes/kaggle_push.py files zds-full-s2         # list a kernel's output files (no download)
    lanes/kaggle_push.py meta zds-full-s2 --dir D  # pull its pushed notebook + kernel-metadata.json
    lanes/kaggle_push.py probe --accelerator TpuV5E8   # 5-min no-op kernel: is this accelerator usable by the account?

Credentials come from ~/.config/zds/env (KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN; override the
path with ZDS_KAGGLE_ENV_FILE). Without that file (and without ZDS_KAGGLE_ENV_FILE):
the same variables from the environment, else Kaggle's own ~/.kaggle/kaggle.json. They go into the kaggle CLI
subprocess env and are never printed. Offline commands render "USERNAME" when the file is absent or incomplete.

Kernel naming: one kernel per session, zds-<mode>-s<N> (chain sessions continue the "full" series), so the previous
session is a fixed `kernel_sources` entry whose output Kaggle mounts read-only under /kaggle/input/.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

LANES = Path(__file__).resolve().parent
ROOT = LANES.parent
sys.path.insert(0, str(LANES))
import kaggle_notebook  # noqa: E402

TEMPLATES = LANES / "templates"
DEFAULT_BUILD = ROOT / "build"
DATASET_SLUG = "zds-code"
TARBALL = "zds_code.tar.gz"
LEDGER_VAR = "ZDS_KAGGLE_LEDGER_OK"
MACHINE_SHAPE = "NvidiaTeslaT4"          # kaggle-cli docs/kernels_metadata.md: "NvidiaTeslaT4 (GPU T4 x2)"
KNOWN_ACCELERATORS = ("NvidiaTeslaT4", "NvidiaTeslaA100", "NvidiaL4", "NvidiaL4X1", "TpuV5E8", "TpuV6E8",
                      "NvidiaH100", "NvidiaRtxPro6000")   # kaggle-cli docs/kernels.md, "as of Sep 2026"

# What goes into the tarball: these repo dirs, from the working tree, minus the denylist.
PACK_DIRS = ("trainer", "scripts")
DENY_NAMES = {".env", "env", "kaggle.json", "credentials", "credentials.json", "id_rsa", "id_ed25519", ".netrc"}
DENY_GLOBS = ("*.key", "*.pem", "*.p12", "*.pfx", ".env.*", "*.secret", "*.token", "*.pt", "*.bin", "*.pyc",
              "*.part", "*.tmp.*")
DENY_DIRS = (".venv", "data", "ckpt", "out", "build", ".git", "__pycache__", ".pytest_cache")
CONFIG_ROOT = Path("~/.config").expanduser().resolve()

# Run modes. full: the trainer's default 1450 steps x 524,288 tokens = 760.2M tokens -> 8 train shards of 100M
# (190 steps each = 1520). smoke: upstream's exact 1390-step schedule, so val at steps 125 and 250 compares 1:1 with
# upstream's log (4.3686, 3.9427); the trainer logs upstream_val_loss / val_minus_upstream in those val records.
MODES = {
    "smoke": dict(stop_after_step=299, train_shards=2, val_every=125, timeout_h=3.0,   # 300 steps = 157M tokens
                  trainer_args=["--num-iters", "1390"]),
    "full": dict(stop_after_step=-1, train_shards=8, val_every=250, timeout_h=12.0),
}
SESSION_CAP_H = 12.0      # Kaggle GPU session maximum
MARGIN_MIN = 20           # trainer's --time-budget-min ends this long before the cap (task: ~20 min)
SIGTERM_MIN = 10          # notebook watchdog SIGTERMs the workers (they checkpoint) this long before the cap
KILL_MIN = 4              # ... and SIGKILLs them here, leaving time for the summary cell
CKPT_EVERY_MIN = 20


# ---------------------------------------------------------------------------------------------
# guard + credentials
# ---------------------------------------------------------------------------------------------
def require_ledger(what: str) -> None:
    if os.environ.get(LEDGER_VAR) != "1":
        raise SystemExit(
            f"refusing `{what}`: it contacts Kaggle and spends Kaggle GPU quota (~30 h/week).\n"
            f"First log this session in your GPU-quota ledger (set ZDS_LEDGER=/path/to/ledger.md to have "
            f"autochain append to it), then rerun with {LEDGER_VAR}=1.")


def env_file() -> Path:
    return Path(os.environ.get("ZDS_KAGGLE_ENV_FILE", "~/.config/zds/env")).expanduser()


def fallback_creds() -> dict[str, str]:
    """Without ZDS_KAGGLE_ENV_FILE and without the default env file: KAGGLE_USERNAME + KAGGLE_KEY (or
    KAGGLE_API_TOKEN) from the environment, else Kaggle's own token file ~/.kaggle/kaggle.json."""
    if "ZDS_KAGGLE_ENV_FILE" in os.environ:
        return {}
    env = {k: os.environ[k] for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN") if os.environ.get(k)}
    if env.get("KAGGLE_USERNAME") and (env.get("KAGGLE_KEY") or env.get("KAGGLE_API_TOKEN")):
        return env
    kj = Path("~/.kaggle/kaggle.json").expanduser()
    try:
        j = json.loads(kj.read_text())
        return {"KAGGLE_USERNAME": j["username"], "KAGGLE_KEY": j["key"]}
    except (OSError, ValueError, KeyError):
        return {}


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines (export prefix, quotes, # comments tolerated). Values are never printed."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def kaggle_env(required: bool = True) -> tuple[dict[str, str] | None, str]:
    path = env_file()
    env = read_env_file(path) or fallback_creds()
    user = env.get("KAGGLE_USERNAME")
    key, token = env.get("KAGGLE_KEY"), env.get("KAGGLE_API_TOKEN")
    if not user or not (key or token):
        if required:
            raise SystemExit(f"kaggle credentials missing: put KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN) in "
                             f"{path} (chmod 600; the file is {'absent' if not path.exists() else 'present but incomplete'}), "
                             "or use Kaggle's token file ~/.kaggle/kaggle.json (Settings -> API -> Create New Token)")
        return None, "USERNAME"
    e = dict(os.environ, KAGGLE_USERNAME=user)
    if key:
        e["KAGGLE_KEY"] = key
    if token:
        e["KAGGLE_API_TOKEN"] = token
    e.pop("KAGGLE_CONFIG_DIR", None)
    return e, user


def kaggle_cli() -> list[str]:
    exe = Path(sys.executable).parent / "kaggle"
    if exe.exists():
        return [str(exe)]
    if shutil.which("kaggle"):
        return [shutil.which("kaggle")]
    raise SystemExit("kaggle CLI not found: .venv/bin/pip install kaggle")


def run_kaggle(args: list[str], env: dict, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("$ kaggle " + " ".join(args), flush=True)
    return subprocess.run(kaggle_cli() + args, env=env, check=check, text=True, capture_output=capture)


# ---------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------
def deny_reason(path: Path, rel: str | None = None) -> str | None:
    """Why a file must not be packed, or None. `rel` is the repo-relative path when known."""
    name = path.name
    parts = [x for x in (rel or path.as_posix()).split("/") if x]
    if name in DENY_NAMES:
        return f"name {name!r} is denied"
    for g in DENY_GLOBS:
        if fnmatch.fnmatch(name, g):
            return f"matches {g}"
    for d in DENY_DIRS:
        if d in parts[:-1]:
            return f"under {d}/"
    try:
        rp = path.resolve()                 # follows symlinks: a link into ~/.config is still denied
        if rp == CONFIG_ROOT or CONFIG_ROOT in rp.parents:
            return "under ~/.config"
        if rp != path.absolute() and ROOT in rp.parents:
            rrel = rp.relative_to(ROOT).as_posix()
            if any(d in rrel.split("/")[:-1] for d in DENY_DIRS):
                return f"symlink into {rrel}"
    except (OSError, ValueError):
        pass
    return None


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def git_sha() -> str:
    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--", *PACK_DIRS)
    return sha + ("-dirty" if dirty else "")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(tmpl: Path, dst: Path, **kw) -> dict:
    text = tmpl.read_text()
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    left = re.findall(r"\{[a-z_]+\}", text)
    if left:
        raise SystemExit(f"unrendered placeholders in {tmpl.name}: {left}")
    meta = json.loads(text)
    dst.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def collect(extra: list[str] | None = None) -> tuple[list[tuple[Path, str]], list[tuple[str, str]]]:
    """(file, arcname) pairs to ship and (path, why) pairs skipped by the denylist."""
    files, skipped = [], []
    roots = [(ROOT / d, d) for d in PACK_DIRS] + [(Path(x).absolute(), Path(x).name) for x in (extra or [])]
    for base, arc in roots:
        if not base.exists():
            raise SystemExit(f"nothing to pack at {base}")
        why = deny_reason(base, rel=arc + ("/x" if base.is_dir() else ""))
        if why:
            skipped.append((str(base), why))
            continue
        cands = [base] if base.is_file() else sorted(q for q in base.rglob("*") if q.is_file() or q.is_symlink())
        for q in cands:
            rel = arc if q == base else f"{arc}/{q.relative_to(base).as_posix()}"
            why = deny_reason(q, rel=rel)
            if why:
                skipped.append((rel, why))
            elif q.is_file():
                files.append((q, rel))
    return files, skipped


def pack(a) -> Path:
    _, user = kaggle_env(required=False)
    d = Path(a.build_dir) / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    files, skipped = collect(getattr(a, "add", None))
    tar_path = d / TARBALL
    with tarfile.open(tar_path, "w:gz") as tf:
        for p, arc in files:
            tf.add(p, arcname=arc, recursive=False)
    sha = git_sha()
    render(TEMPLATES / "dataset-metadata.json.tmpl", d / "dataset-metadata.json", username=user,
           dataset_slug=DATASET_SLUG, dataset_title="Zero-Dollar Speedrun code", git_sha=sha)
    manifest = {"git_sha": sha, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "files": {arc: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p, arc in files},
                "skipped_by_denylist": [{"path": p_, "why": w} for p_, w in skipped]}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for path, why in skipped:
        if "__pycache__" not in path and not path.endswith(".pyc"):
            print(f"  skipped (denylist): {path}  [{why}]")
    print(f"packed {len(files)} files ({tar_path.stat().st_size / 1e3:.0f} kB, code {sha}) -> {d}")
    return d


# ---------------------------------------------------------------------------------------------
# kernel render
# ---------------------------------------------------------------------------------------------
def kernel_slug(mode: str, session: int) -> str:
    series = "full" if mode in ("full", "chain") else mode
    return f"zds-{series}-s{session}"


def resolve_resume(a, user: str) -> str | None:
    if getattr(a, "resume_from", None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[a-z0-9-]+", a.resume_from):
            raise SystemExit(f"--resume-from must be <user>/<kernel-slug>, got {a.resume_from!r}")
        return a.resume_from
    if a.mode == "chain":
        if a.session < 2:
            raise SystemExit("chain needs --session >= 2 (it resumes from session N-1) or an explicit --resume-from")
        return f"{user}/{kernel_slug('full', a.session - 1)}"
    return None


def run_config(a, user: str) -> dict:
    base = MODES["smoke" if a.mode == "smoke" else "full"]
    timeout_h = min(a.timeout_h or base["timeout_h"], SESSION_CAP_H)
    cfg = dict(mode=a.mode, kernel=f"{user}/{kernel_slug(a.mode, a.session)}", resume_from=resolve_resume(a, user),
               stop_after_step=base["stop_after_step"] if a.steps is None else a.steps - 1,
               train_shards=a.train_shards or base["train_shards"],
               val_every=base["val_every"] if a.val_every is None else a.val_every,
               session_cap_h=timeout_h, margin_min=MARGIN_MIN, sigterm_min=SIGTERM_MIN, kill_min=KILL_MIN,
               ckpt_every_min=CKPT_EVERY_MIN, scratch="/kaggle/tmp",
               trainer_args=list(base.get("trainer_args", [])) + list(a.trainer_arg or []))
    return cfg


def render_kernel(a, user: str) -> Path:
    d = Path(a.build_dir) / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    cfg = run_config(a, user)
    (d / "kernel.ipynb").write_text(json.dumps(kaggle_notebook.build(cfg), indent=1) + "\n")
    slug = kernel_slug(a.mode, a.session)
    meta = render(TEMPLATES / "kernel-metadata.json.tmpl", d / "kernel-metadata.json", username=user,
                  kernel_slug=slug, kernel_title=slug, dataset_slug=DATASET_SLUG,
                  machine_shape=getattr(a, "accelerator", None) or MACHINE_SHAPE)
    if cfg["resume_from"]:
        meta["kernel_sources"] = [cfg["resume_from"]]
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (d / "run_config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"rendered {meta['id']} ({a.mode}, cap {cfg['session_cap_h']} h, resume_from={cfg['resume_from']}) -> {d}")
    return d


RESCORE = dict(slug="zds-rescore-s2", sources=("zds-full-s2", "zds-full-s1"), timeout_h=2.0,
               ckpt="zds-full-s2/run/ckpt/ckpt_0001451.pt", expect=3.2698315009167063,
               hash_files=["zds-full-s1/run/ckpt/ckpt_0001110.pt", "zds-full-s2/run/ckpt/ckpt_0001435.pt"],
               model_file="zds_final_model_step1450.pt")


def render_rescore(a, user: str) -> Path:
    """Eval-only kernel: attach the training sessions' outputs, re-score the final checkpoint (review M3 #4)."""
    d = Path(a.build_dir) / "kernel-rescore"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    R = RESCORE
    sources = [f"{user}/{s}" for s in R["sources"]]
    cfg = dict(mode="rescore", kernel=f"{user}/{R['slug']}", sources=sources, ckpt=R["ckpt"], expect=R["expect"],
               hash_files=R["hash_files"], model_file=R["model_file"], session_cap_h=R["timeout_h"],
               scratch="/kaggle/tmp")
    nb = kaggle_notebook.build(cfg, kaggle_notebook.RESCORE_CELLS)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse(c["source"])
    (d / "kernel.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    meta = render(TEMPLATES / "kernel-metadata.json.tmpl", d / "kernel-metadata.json", username=user,
                  kernel_slug=R["slug"], kernel_title=R["slug"], dataset_slug=DATASET_SLUG, machine_shape=MACHINE_SHAPE)
    meta["kernel_sources"] = sources
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (d / "run_config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"rendered {meta['id']} (rescore {R['ckpt']}, sources {sources}) -> {d}")
    return d


def cmd_rescore(a) -> None:
    require_ledger("rescore")
    if not a.skip_dataset:
        pack(a)
        errs = validate_dataset(Path(a.build_dir) / "dataset")
        if errs:
            raise SystemExit("dataset validation failed:\n  " + "\n  ".join(errs))
        cmd_dataset(a)
    env, user = kaggle_env()
    d = render_rescore(a, user)
    run_kaggle(["kernels", "push", "-p", str(d), "-t", str(int(RESCORE["timeout_h"] * 3600)),
                "--accelerator", MACHINE_SHAPE], env)
    print(f"pushed {user}/{RESCORE['slug']}. Then: lanes/kaggle_push.py pull --kernel {RESCORE['slug']}")


# ---------------------------------------------------------------------------------------------
# validation (no network)
# ---------------------------------------------------------------------------------------------
def validate_dataset(d: Path) -> list[str]:
    errs = []
    dm = json.loads((d / "dataset-metadata.json").read_text())
    for k in ("title", "id", "licenses"):
        if k not in dm:
            errs.append(f"dataset-metadata missing {k}")
    if not (6 <= len(dm.get("title", "")) <= 50):
        errs.append("dataset title must be 6-50 chars")
    if not (6 <= len(dm.get("id", "").split("/")[-1]) <= 50):
        errs.append("dataset slug must be 6-50 chars")
    if len(dm.get("licenses", [])) != 1:
        errs.append("exactly one license")
    sub = dm.get("subtitle")
    if sub and not (20 <= len(sub) <= 80):
        errs.append("subtitle must be 20-80 chars")
    with tarfile.open(d / TARBALL) as tf:
        names = tf.getnames()
    for need in ("trainer/train_t4.py", "scripts/get_data.py"):
        if need not in names:
            errs.append(f"tarball lacks {need}")
    for n in names:
        why = deny_reason(Path(n), rel=n)
        if why:
            errs.append(f"tarball holds denied {n} [{why}]")
    return errs


def validate_kernel(d: Path) -> tuple[list[str], dict]:
    errs = []
    km = json.loads((d / "kernel-metadata.json").read_text())
    if km.get("language") != "python" or km.get("kernel_type") != "notebook":
        errs.append("kernel-metadata language/kernel_type")
    if km.get("is_private") != "true":
        errs.append("kernel must be private")
    if km.get("enable_gpu") != "true" or km.get("enable_internet") != "true":
        errs.append("kernel-metadata needs enable_gpu and enable_internet = 'true'")
    if km.get("machine_shape") not in KNOWN_ACCELERATORS:
        errs.append(f"machine_shape {km.get('machine_shape')!r} not in {KNOWN_ACCELERATORS}")
    if not km.get("dataset_sources"):
        errs.append("kernel-metadata has no dataset_sources")
    slug = km["id"].split("/")[-1]
    if not (5 <= len(km.get("title", "")) <= 50) or km["title"].lower().replace(" ", "-") != slug:
        errs.append("kernel title must be 5-50 chars and slugify to the id's slug")
    nb = json.loads((d / km["code_file"]).read_text())
    if nb.get("nbformat") != 4:
        errs.append("notebook nbformat != 4")
    srcs = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        srcs.append(src)
        try:
            ast.parse(src)
        except SyntaxError as e:
            errs.append(f"notebook cell {i}: {e}")
    allsrc = "\n".join(srcs)
    for need in ("torchrun", "--nproc_per_node", "--provider", "--resume", "--time-budget-min", "--ckpt-every-min",
                 "/kaggle/working", "scripts/get_data.py"):
        if need not in allsrc:
            errs.append(f"notebook never mentions {need}")
    cfg = json.loads((d / "run_config.json").read_text())
    if cfg["resume_from"] and km.get("kernel_sources") != [cfg["resume_from"]]:
        errs.append("resume_from is not wired into kernel_sources")
    if cfg["session_cap_h"] > SESSION_CAP_H:
        errs.append("session cap above 12 h")
    return errs, km


def dryrun(a) -> None:
    pack(a)
    errs = validate_dataset(Path(a.build_dir) / "dataset")
    _, user = kaggle_env(required=False)
    rendered = {}
    for mode, session in (("smoke", 1), ("full", 1), ("chain", 2)):
        b = argparse.Namespace(**{**vars(a), "mode": mode, "session": session, "resume_from": None})
        kd = render_kernel(b, user)
        e, km = validate_kernel(kd)
        errs += [f"[{mode}] {x}" for x in e]
        dst = Path(a.build_dir) / f"kernel-{mode}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(kd, dst)
        rendered[mode] = km
    # leave build/kernel as the smoke render (what `smoke` would push first)
    shutil.rmtree(Path(a.build_dir) / "kernel")
    shutil.copytree(Path(a.build_dir) / "kernel-smoke", Path(a.build_dir) / "kernel")
    if errs:
        raise SystemExit("validation failed:\n  " + "\n  ".join(errs))
    for mode, km in rendered.items():
        print(f"validated {mode:5s}: {km['id']} machine_shape={km['machine_shape']} gpu={km['enable_gpu']} "
              f"internet={km['enable_internet']} private={km['is_private']} kernel_sources={km['kernel_sources']}")
    print("dry run ok (no Kaggle contact)")


# ---------------------------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------------------------
def dataset_ready(env, ref: str) -> str:
    r = run_kaggle(["datasets", "status", ref], env, check=False, capture=True)
    return (r.stdout + r.stderr).strip().lower() if r.returncode == 0 else "missing"


def cmd_dataset(a) -> None:
    require_ledger("dataset")
    env, user = kaggle_env()
    d = Path(a.build_dir) / "dataset"
    if not (d / "dataset-metadata.json").exists():
        raise SystemExit("nothing packed: run `pack` first")
    dm = json.loads((d / "dataset-metadata.json").read_text())
    if dm["id"].split("/")[0] != user:
        raise SystemExit(f"packed for {dm['id'].split('/')[0]!r} but credentials are for another user: re-run pack")
    ref = f"{user}/{DATASET_SLUG}"
    st = dataset_ready(env, ref)
    if "missing" in st or "not found" in st or "404" in st:
        run_kaggle(["datasets", "create", "-p", str(d), "-q"], env)                      # private by default
    else:
        msg = getattr(a, "message", None) or f"zds {git_sha()} {time.strftime('%Y-%m-%d %H:%M')}"
        run_kaggle(["datasets", "version", "-p", str(d), "-m", msg, "-q"], env)
    for _ in range(40):                                                                   # <= 10 min
        st = dataset_ready(env, ref)
        if "ready" in st:
            print(f"dataset {ref}: ready")
            return
        time.sleep(15)
    print(f"dataset {ref}: still '{st}' after 10 min; the kernel may start on the previous version")


def push_kernel(a) -> None:
    env, user = kaggle_env()
    kd = render_kernel(a, user)
    e, km = validate_kernel(kd)
    if e:
        raise SystemExit("kernel validation failed:\n  " + "\n  ".join(e))
    cfg = json.loads((kd / "run_config.json").read_text())
    run_kaggle(["kernels", "push", "-p", str(kd), "-t", str(int(cfg["session_cap_h"] * 3600)),
                "--accelerator", km["machine_shape"]], env)
    slug = km["id"]
    print(f"pushed {slug}. Next: `lanes/kaggle_push.py status --mode {a.mode} --session {a.session}`, then `pull`.")


def cmd_run(a) -> None:
    require_ledger(a.mode)
    if not a.skip_dataset:
        pack(a)
        errs = validate_dataset(Path(a.build_dir) / "dataset")
        if errs:
            raise SystemExit("dataset validation failed:\n  " + "\n  ".join(errs))
        cmd_dataset(a)
    push_kernel(a)


def cmd_status(a) -> None:
    require_ledger("status")
    env, user = kaggle_env()
    run_kaggle(["kernels", "status", f"{user}/{kernel_slug(a.mode, a.session)}"], env, check=False)


def cmd_pull(a) -> None:
    require_ledger("pull")
    env, user = kaggle_env()
    slug = a.kernel or kernel_slug(a.mode, a.session)
    dst = ROOT / "out" / "kaggle" / (a.run or slug)
    dst.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", f"{user}/{slug}", "-p", str(dst), "-o"]
    if not a.with_ckpt:
        args += ["--file-pattern", r"^(?!.*\.pt$).*"]     # checkpoints are GBs; the next session reads them on Kaggle
    run_kaggle(args, env)
    r = run_kaggle(["kernels", "logs", f"{user}/{slug}"], env, check=False, capture=True)
    if r.returncode == 0:
        (dst / "kernel.log").write_text(r.stdout)
    print(f"pulled into {dst}: {sorted(p.name for p in dst.iterdir())}")
    for s in dst.rglob("summary.json"):
        print(s.read_text())


def cmd_files(a) -> None:
    """List a kernel's output files (names + sizes) without downloading them."""
    require_ledger("files")
    env, user = kaggle_env()
    ref = a.kernel if "/" in a.kernel else f"{user}/{a.kernel}"
    run_kaggle(["kernels", "files", ref, "--page-size", "200"], env, check=False)


def cmd_meta(a) -> None:
    """Pull a kernel's pushed source + kernel-metadata.json (`kaggle kernels pull -m`) into --dir."""
    require_ledger("meta")
    env, user = kaggle_env()
    ref = a.kernel if "/" in a.kernel else f"{user}/{a.kernel}"
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    run_kaggle(["kernels", "pull", ref, "-p", str(a.dir), "-m"], env)
    print(f"pulled {ref} source + metadata -> {a.dir}: {sorted(p.name for p in Path(a.dir).iterdir())}")


def cmd_quota(a) -> None:
    require_ledger("quota")
    env, _ = kaggle_env()
    run_kaggle(["quota"], env, check=False)


def cmd_probe(a) -> None:
    """Push a ~1-minute kernel on `--accelerator` that prints what devices it got. Costs a few quota minutes."""
    require_ledger("probe")
    if a.accelerator not in KNOWN_ACCELERATORS:
        raise SystemExit(f"--accelerator must be one of {KNOWN_ACCELERATORS}")
    env, user = kaggle_env()
    d = Path(a.build_dir) / "probe"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    slug = f"zds-probe-{a.accelerator.lower()}"
    src = ("import glob, json, os, shutil, subprocess\n"
           "def sh(c):\n"
           "    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()[-2000:]\n"
           "    except Exception as e: return repr(e)\n"
           "out = {'nvidia_smi_path': shutil.which('nvidia-smi'), 'nvidia_smi': sh('nvidia-smi -L'),\n"
           "       'dev_nvidia': glob.glob('/dev/nvidia*'), 'dev_accel': glob.glob('/dev/accel*'), 'dev_vfio': glob.glob('/dev/vfio/*'),\n"
           "       'tpu_env': {k: v for k, v in os.environ.items() if 'TPU' in k or 'XLA' in k},\n"
           "       'cpu_count': os.cpu_count(), 'mem': sh('free -g | head -2')}\n"
           "try:\n"
           "    import torch; out['torch'] = torch.__version__; out['cuda'] = torch.cuda.is_available()\n"
           "    out['cuda_devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
           "except Exception as e: out['torch_err'] = repr(e)\n"
           "try:\n"
           "    import jax; out['jax_devices'] = [str(x) for x in jax.devices()]\n"
           "except Exception as e: out['jax_err'] = repr(e)[:300]\n"
           "print(json.dumps(out, indent=1)); json.dump(out, open('/kaggle/working/probe.json', 'w'))\n")
    (d / "probe.py").write_text(src)
    meta = dict(id=f"{user}/{slug}", title=slug, code_file="probe.py", language="python", kernel_type="script",
                is_private="true", enable_gpu="true" if not a.accelerator.startswith("Tpu") else "false",
                enable_tpu="true" if a.accelerator.startswith("Tpu") else "false", enable_internet="false",
                machine_shape=a.accelerator, dataset_sources=[], competition_sources=[], kernel_sources=[],
                model_sources=[])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    r = run_kaggle(["kernels", "push", "-p", str(d), "-t", "300", "--accelerator", a.accelerator], env, check=False)
    print(f"push exit {r.returncode}. An error naming the accelerator = not available to this account. "
          f"Otherwise wait ~3 min: kaggle kernels status {user}/{slug}; kaggle kernels output {user}/{slug} -p out/kaggle/{slug}")


# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.environ.get("ZDS_KAGGLE_BUILD", str(DEFAULT_BUILD)))
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add_pack(p):
        p.add_argument("--add", action="append", help="extra file/dir for the tarball (denylist still applies)")

    def add_sel(p, mode_default="full"):
        p.add_argument("--mode", choices=["smoke", "full", "chain"], default=mode_default)
        p.add_argument("--session", type=int, default=1, help="session number N in zds-<mode>-s<N>")

    def add_run(p):
        p.add_argument("--session", type=int, default=1, help="session number N in the kernel slug zds-<mode>-s<N>")
        p.add_argument("--resume-from", default=None, help="<user>/<kernel-slug> whose output holds the checkpoint")
        p.add_argument("--steps", type=int, default=None, help="train this many steps from 0 then stop (smoke: 300)")
        p.add_argument("--train-shards", type=int, default=None, help="FineWeb train shards to download")
        p.add_argument("--val-every", type=int, default=None)
        p.add_argument("--timeout-h", type=float, default=0, help="session cap in hours (<= 12; smoke default 3)")
        p.add_argument("--accelerator", default=MACHINE_SHAPE, choices=KNOWN_ACCELERATORS)
        p.add_argument("--trainer-arg", action="append", help="extra flag passed to train_t4.py (repeatable; use =)")

    p = sp.add_parser("pack"); add_pack(p)
    p = sp.add_parser("kernel", help="render only"); add_run(p)
    p.add_argument("--mode", choices=["smoke", "full", "chain"], default="smoke")
    p = sp.add_parser("dryrun"); add_pack(p)
    p.set_defaults(steps=None, train_shards=None, val_every=None, timeout_h=0, accelerator=MACHINE_SHAPE, trainer_arg=None)
    p = sp.add_parser("dataset"); p.add_argument("-m", "--message", default=None)
    for mode in ("smoke", "full", "chain"):
        p = sp.add_parser(mode); add_run(p); add_pack(p)
        p.add_argument("--skip-dataset", action="store_true", help="reuse the uploaded code dataset as is")
        p.add_argument("-m", "--message", default=None)
        p.set_defaults(mode=mode, **({"session": 2} if mode == "chain" else {}))
    p = sp.add_parser("status"); add_sel(p)
    p = sp.add_parser("pull"); add_sel(p); p.add_argument("--run", default=None)
    p.add_argument("--kernel", default=None, help="pull this kernel slug instead of zds-<mode>-s<N> (e.g. zds-rescore-s2)")
    p.add_argument("--with-ckpt", action="store_true", help="also download the .pt checkpoints (GBs)")
    sp.add_parser("quota")
    p = sp.add_parser("rescore", help="eval-only kernel: re-score zds-full-s2's final checkpoint"); add_pack(p)
    p.add_argument("--skip-dataset", action="store_true"); p.add_argument("-m", "--message", default=None)
    p = sp.add_parser("rescore-render", help="offline: render the rescore kernel only")
    p = sp.add_parser("files", help="list a kernel's output files"); p.add_argument("kernel", help="<slug> or <user>/<slug>")
    p = sp.add_parser("meta", help="pull a kernel's source + kernel-metadata.json")
    p.add_argument("kernel", help="<slug> or <user>/<slug>"); p.add_argument("--dir", required=True)
    p = sp.add_parser("probe"); p.add_argument("--accelerator", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "pack":
        pack(a)
    elif a.cmd == "kernel":
        render_kernel(a, kaggle_env(required=False)[1])
    elif a.cmd == "dryrun":
        dryrun(a)
    elif a.cmd == "dataset":
        cmd_dataset(a)
    elif a.cmd in ("smoke", "full", "chain"):
        cmd_run(a)
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "pull":
        cmd_pull(a)
    elif a.cmd == "quota":
        cmd_quota(a)
    elif a.cmd == "rescore":
        cmd_rescore(a)
    elif a.cmd == "rescore-render":
        render_rescore(a, kaggle_env(required=False)[1])
    elif a.cmd == "files":
        cmd_files(a)
    elif a.cmd == "meta":
        cmd_meta(a)
    elif a.cmd == "probe":
        cmd_probe(a)


if __name__ == "__main__":
    main()
