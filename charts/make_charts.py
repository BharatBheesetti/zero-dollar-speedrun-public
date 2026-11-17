#!/usr/bin/env python3
"""Regenerate every chart in charts/ from the run logs. No numbers are typed in here.

Inputs (see charts/SOURCES.md):
  receipts/zds-full-s1/{summary.json,run/log.jsonl}
  receipts/zds-full-s2/{summary.json,run/log.jsonl}     (s2's log starts with a copy of s1's)
  charts/data/upstream_record18_31d6c427.txt             (verbatim modded-nanogpt record #18 log)
  charts/SOURCES.md                                      (cost-ladder table)

Usage:  .venv/bin/python charts/make_charts.py [--runs receipts] [--out charts]     (needs matplotlib)
Writes: val_vs_step, val_vs_t4_hours, cost_ladder (.svg and .png each).
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Palette: dataviz reference instance, light mode (validated: blue/orange adjacent CVD dE 24.7, contrast >= 3:1).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OURS = "#2a78d6"      # slot 1
UPSTREAM = "#eb6834"  # slot 2
CONTEXT = "#c3c2b7"   # de-emphasis gray for train-loss context and rented-GPU bars

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "font.weight": "normal",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "svg.fonttype": "none",
})


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(runs):
    s1 = json.load(open(os.path.join(runs, "zds-full-s1", "summary.json")))
    s2 = json.load(open(os.path.join(runs, "zds-full-s2", "summary.json")))
    log = read_jsonl(os.path.join(runs, "zds-full-s2", "run", "log.jsonl"))
    resume_idx = next(i for i, r in enumerate(log) if r.get("event") == "resume")
    # s1 part of the log = everything before the resume; the s1 exit record ends it.
    s1_exit = [r for r in log[:resume_idx] if r.get("event") == "exit"][-1]
    resume = log[resume_idx]
    vals = [r for r in log if r.get("event") == "val"]
    steps = [r for r in log if "train_loss" in r and "event" not in r]
    final_val = vals[-1]
    start = next(r for r in log if r.get("event") == "start")
    skips = [r["step"] for r in steps if r.get("skipped")]
    return dict(s1=s1, s2=s2, log=log, s1_exit=s1_exit, resume=resume, vals=vals, steps=steps,
                final_val=final_val, start=start, skips=skips)


def load_upstream(path):
    vals, train_ms = [], None
    pat = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms")
    total = None
    for line in open(path):
        m = pat.match(line.strip())
        if m:
            vals.append((int(m.group(1)), float(m.group(3))))
            total = int(m.group(2))
            train_ms = int(m.group(4))
    return dict(vals=vals, total_steps=total, train_ms=train_ms)


def load_target(readme_sources):
    # The target is quoted in SOURCES.md from the upstream README ("attains 3.28 cross-entropy loss").
    m = re.search(r"attains ([\d.]+)\s+cross-entropy loss", open(readme_sources).read().replace("\n", " "))
    return float(m.group(1))


def load_cost_table(path):
    rows, lines = [], open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("| key |"))
    hdr = [c.strip() for c in lines[i].strip("|").split("|")]
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows.append(dict(zip(hdr, cells)))
    return rows


def rolling(xs, n):
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= n:
            acc -= xs[i - n]
        out.append(acc / min(i + 1, n))
    return out


def save(fig, out, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def title(fig, ax, head, sub):
    ax.set_title(head, loc="left", fontsize=14, color=INK, fontweight="bold", pad=28)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom", ha="left")


def legend(ax, anchor):
    # Legend keys are plain 2px line segments (no markers), so they don't read as dash-dot.
    hs, ls = ax.get_legend_handles_labels()
    keys = [matplotlib.lines.Line2D([], [], color=h.get_color(), lw=2) for h in hs]
    ax.legend(keys, ls, loc="upper right", bbox_to_anchor=anchor, frameon=False, fontsize=9.5, labelcolor=INK)


def end_label(ax, x, y, text, color, dx=8, dy=0, ha="left"):
    ax.plot([x], [y], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=10, color=INK)


def chart_val_vs_step(run, up, target, out):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ours = [(v["step"], v["val_loss"]) for v in run["vals"] if v["step"] > 0]
    ups = [(s, l) for s, l in up["vals"] if s >= ours[0][0]]
    lo = min(l for _, l in ours + ups)
    hi = max(l for s, l in ours + ups if s >= ours[0][0])
    ax.axhline(target, color=INK_2, lw=1, zorder=1)
    ax.text(0.01, target, f"target {target:g}", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=10, color=INK_2)
    rs = run["resume"]["next_step"]
    ax.axvline(rs, color=AXIS, lw=1, zorder=1)
    ax.text(rs, 0.98, f"  session 2 resumes from checkpoint\n  (step {rs})", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9.5, color=INK_2)
    ax.plot(*zip(*ups), color=UPSTREAM, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.5,
            label=f"modded-nanogpt record #18, 8×H100, bf16 ({up['total_steps']:,}-step schedule)")
    ax.plot(*zip(*ours), color=OURS, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.5,
            label=f"This port, Kaggle free 2×T4, fp16 ({run['start']['total_steps']:,}-step schedule)")
    fs, fl = ours[-1]
    us, ul = ups[-1]
    end_label(ax, fs, fl, f"{fl:.4f}", OURS, dx=8, dy=-8)
    end_label(ax, us, ul, f"{ul:.4f}", UPSTREAM, dx=-8, dy=-10, ha="right")
    ax.set_xlim(ours[0][0] - 60, max(fs, us) + 90)
    ax.set_ylim(lo - 0.04, hi + 0.06)
    ax.set_xlabel("training step (524,288 tokens each)")
    ax.set_ylabel("FineWeb val loss (standard 10.49M-token set)")
    legend(ax, (1.0, 0.86))
    first = ours[0][0]
    title(fig, ax, "Same model, hardware it was never meant for",
          f"Val loss by step, from step {first}. Single run. Ours was scored every {first} steps, "
          f"upstream every {ups[1][0] - ups[0][0]}.")
    save(fig, out, "val_vs_step")


def chart_val_vs_hours(run, target, out):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    st = run["steps"]
    th = [r["gpu_seconds_total"] / 3600 for r in st]
    tl = rolling([r["train_loss"] for r in st], 25)
    k = next(i for i, r in enumerate(st) if r["step"] >= run["vals"][1]["step"])  # skip the warm-up spike
    ax.plot(th[k:], tl[k:], color=CONTEXT, lw=1.5, label="train loss (25-step mean)", zorder=2)
    vh = [(v["gpu_seconds_total"] / 3600, v["val_loss"]) for v in run["vals"] if v["step"] > 0]
    ax.plot(*zip(*vh), color=OURS, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label="val loss", zorder=3)
    ax.axhline(target, color=INK_2, lw=1, zorder=1)
    ax.text(0.005, target, f"target {target:g}", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=10, color=INK_2)
    b = run["s1_exit"]["gpu_seconds_total"] / 3600
    ax.axvline(b, color=AXIS, lw=1, zorder=1)
    s1h, s2h = run["s1"]["elapsed_h"], run["s2"]["elapsed_h"]
    ax.text(b, 0.98, f"Kaggle session 1 ends ({s1h:.1f} h)  ", transform=ax.get_xaxis_transform(), ha="right",
            va="top", fontsize=9.5, color=INK_2)
    ax.text(b, 0.98, f"  session 2 ({s2h:.1f} h)\n  resumes from checkpoint", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9.5, color=INK_2)
    fx, fy = vh[-1]
    end_label(ax, fx, fy, f"{fy:.4f}", OURS, dx=-8, dy=-12, ha="right")
    lo = min(min(tl[k:]), fy)
    hi = max(max(tl[k:]), max(v for _, v in vh))
    ax.set_ylim(lo - 0.05, hi + 0.12)
    ax.set_xlim(0, fx * 1.04)
    ax.set_xlabel(f"cumulative T4 GPU-hours, trainer's clock (2 GPUs × wall time; ends at {fx:.1f}, "
                  f"excludes data download)")
    ax.set_ylabel("loss")
    legend(ax, (1.0, 0.84))
    title(fig, ax, f"{2 * (s1h + s2h):.1f} free T4-hours to the target",
          f"One run, two Kaggle sessions ({s1h:.1f} h + {s2h:.1f} h = {s1h + s2h:.1f} session-hours). "
          f"Cost: $0.")
    save(fig, out, "val_vs_t4_hours")


def chart_cost_ladder(run, up, rows, out):
    by = {r["key"]: r for r in rows}
    up_min = up["train_ms"] / 60000
    rate = float(by["nanochat"]["rate_usd_per_hr"])
    derived = rate * up["train_ms"] / 3_600_000
    t4h = 2 * (run["s1"]["elapsed_h"] + run["s2"]["elapsed_h"])
    bars = [
        (f"{by['nanochat']['label']} ({by['nanochat']['year']})", float(by["nanochat"]["cost_usd"]), CONTEXT,
         "${:.0f}".format(float(by["nanochat"]["cost_usd"])), by["nanochat"]["note"]),
        (f"{by['llmc']['label']} ({by['llmc']['year']})", float(by["llmc"]["cost_usd"]), CONTEXT,
         "${:.0f}".format(float(by["llmc"]["cost_usd"])), by["llmc"]["note"]),
        (f"Record #18 on rented 8×H100", derived, CONTEXT, f"≈${derived:.1f}",
         f"derived: {up_min:.1f} min × ${rate:.0f}/hr; train loop only"),
        (f"{by['this']['label']} ({by['this']['year']})", float(by["this"]["cost_usd"]), OURS,
         "$0", f"{t4h:.1f} {by['this']['note']}"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ys = list(range(len(bars)))[::-1]
    for y, (lab, v, c, txt, note) in zip(ys, bars):
        ax.barh(y, v, height=0.34, color=c, zorder=2)
        if v == 0:
            ax.plot([0], [y], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=3, clip_on=False)
        ax.annotate(txt, (v, y), xytext=(8, 0), textcoords="offset points", va="center", ha="left", fontsize=11,
                    color=INK, fontweight="bold")
        ax.text(0.3, y - 0.36, note, va="center", ha="left", fontsize=9.5, color=INK_2)
    ax.set_yticks(ys, [b[0] for b in bars], color=INK, fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(b[1] for b in bars) * 1.15)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("what the run cost to rent (USD)")
    title(fig, ax, "What a GPT-2-class training run costs",
          "Quoted prices from the authors' own posts (charts/SOURCES.md). Different jobs; not like-for-like.")
    save(fig, out, "cost_ladder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "receipts"))
    ap.add_argument("--upstream", default=os.path.join(HERE, "data", "upstream_record18_31d6c427.txt"))
    ap.add_argument("--sources", default=os.path.join(HERE, "SOURCES.md"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    run = load_run(a.runs)
    up = load_upstream(a.upstream)
    target = load_target(a.sources)
    rows = load_cost_table(a.sources)
    chart_val_vs_step(run, up, target, a.out)
    chart_val_vs_hours(run, target, a.out)
    chart_cost_ladder(run, up, rows, a.out)
    fv = run["final_val"]
    print(f"final val {fv['val_loss']:.4f} @ step {fv['step']} ({fv['tokens']:,} tokens); "
          f"upstream {up['vals'][-1][1]} @ {up['vals'][-1][0]}; skips at {run['skips']}; "
          f"session h {run['s1']['elapsed_h']} + {run['s2']['elapsed_h']}; "
          f"trainer T4-h {run['final_val']['gpu_seconds_total'] / 3600:.2f}")


if __name__ == "__main__":
    main()
