#!/usr/bin/env python3
"""Re-derive every number in the headline claim from the files in receipts/. Stdlib only.

    python3 receipts/verify.py        # prints each number with its source; exit 1 if any check fails
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_TOKENS = 1390 * 524_288                 # record #18: 1,390 steps x 524,288 tokens = 728,760,320
VAL_FILE_SHA256 = "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640"
VAL_SLICE_SHA256 = "ce6ad9a79700ee93c1e3edd25ecdcb06e279a958b5a3841c1dd538ef8819e375"
FAILS = []


def p(*parts):
    return os.path.join(HERE, *parts)


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def js(path):
    with open(path) as f:
        return json.load(f)


def check(name, value, ok, source):
    print(f"{'ok ' if ok else 'BAD'} {name:<44} {value!s:<28} {source}")
    if not ok:
        FAILS.append(name)


s1_log, s2_log = jl(p("zds-full-s1/run/log.jsonl")), jl(p("zds-full-s2/run/log.jsonl"))
s1_sum, s2_sum = js(p("zds-full-s1/summary.json")), js(p("zds-full-s2/summary.json"))
start = s2_log[0]
vals = [r for r in s2_log if r.get("event") == "val"]
final = vals[-1]
steps = {}
for r in s2_log:
    if "train_loss" in r:
        steps[r["step"]] = r

# the result
check("final val loss", round(final["val_loss"], 4), final["step"] == 1450 and final["val_loss"] <= 3.28,
      "zds-full-s2/run/log.jsonl: last event=val, val_loss")
check("val set is upstream's", final["val_set_is_upstream"],
      final["val_set_is_upstream"] is True and final["val_tokens_sha256"] == VAL_SLICE_SHA256,
      "same record: val_set_is_upstream, val_tokens_sha256")
check("val tokens", f"{final['val_tokens']:,}", final["val_tokens"] == 10_485_760 and final["val_seq_len"] == 65536,
      "same record: val_tokens, val_seq_len")
check("val file sha256", start["val_file_sha256"][:16] + "...", start["val_file_sha256"] == VAL_FILE_SHA256,
      "zds-full-s2/run/log.jsonl line 1 (start): val_file_sha256")
check("steps", len(steps), sorted(steps) == list(range(1450)), "zds-full-s2/run/log.jsonl: step records 0..1449")
skipped = sorted(s for s, r in steps.items() if r.get("skipped"))
check("fp16 overflow skips (steps)", skipped, skipped == [1006], "step records: skipped=true")
check("optimizer updates", 1450 - len(skipped), 1450 - len(skipped) == 1449, "1450 - skips")
check("training tokens", f"{final['tokens']:,}", final["tokens"] == 1450 * 524_288 == steps[1449]["tokens"],
      "final val record: tokens (= step 1449 tokens)")
more = final["tokens"] / UPSTREAM_TOKENS - 1
check("tokens vs 8xH100 record (728.8M)", f"+{100 * more:.1f}%", round(100 * more, 1) == 4.3,
      "760,217,600 / (1390 x 524,288) - 1")
check("params (incl. embedding tables)", f"{start['params']:,}", start["params"] == 275_742_772,
      "start record: params")

# time and hardware
h1, h2 = s1_sum["elapsed_h"], s2_sum["elapsed_h"]
check("session 1 hours", h1, round(h1, 1) == 11.7, "zds-full-s1/summary.json: elapsed_h")
check("session 2 hours", h2, round(h2, 1) == 3.7, "zds-full-s2/summary.json: elapsed_h")
check("Kaggle session-hours", round(h1 + h2, 2), round(h1 + h2, 1) == 15.4, "sum of elapsed_h")
check("T4-hours (2 GPUs per session)", round(2 * (h1 + h2), 2), round(2 * (h1 + h2), 1) == 30.8,
      "2 x session-hours (summary.json n_gpu)")
check("GPUs per session", (s1_sum["n_gpu"], s2_sum["n_gpu"], s2_sum["gpu"]),
      s1_sum["n_gpu"] == s2_sum["n_gpu"] == 2 and s2_sum["gpu"] == "Tesla T4", "summary.json: n_gpu, gpu")
for k in ("zds-full-s1", "zds-full-s2"):
    km = js(p(k, "kernel-metadata.json"))
    check(f"{k} machine + privacy", (km["machine_shape"], km["is_private"]),
          km["machine_shape"] == "NvidiaTeslaT4" and km["is_private"] in (True, "true"),
          f"{k}/kernel-metadata.json: machine_shape, is_private")
ex = [r for r in s2_log if r.get("event") == "exit"][-1]
check("trainer's own whole-process GPU-hours", round(ex["gpu_seconds_total"] / 3600, 2),
      ex["reason"] == "done", "zds-full-s2 log: last event=exit, gpu_seconds_total / 3600")

# the resume that joins the two sessions
n1 = len(s1_log)
check("s2 log starts with s1's log, unchanged", f"{n1} lines", s2_log[:n1] == s1_log, "compare the two log.jsonl")
res = s2_log[n1]
side = js(p("zds-full-s1/run/ckpt/ckpt_0001110.json"))
check("resume = s1's last checkpoint", (res["next_step"], res["tokens_seen"]),
      res["event"] == "resume" and res["next_step"] == side["next_step"] == 1110
      and res["tokens_seen"] == side["tokens_seen"] and res["gpu_seconds"] == side["gpu_seconds"],
      "s2 log resume record vs zds-full-s1/run/ckpt/ckpt_0001110.json")
check("s1 stopped on its time budget", s1_sum["exit_reason"], s1_sum["exit_reason"] == "time_budget",
      "zds-full-s1/summary.json: exit_reason")
fin = js(p("zds-full-s2/run/ckpt/ckpt_0001451.json"))
check("final checkpoint = the val'd state", (fin["step"], fin["reason"]),
      fin["step"] == 1450 and fin["reason"] == "final" and fin["tokens_seen"] == final["tokens"],
      "zds-full-s2/run/ckpt/ckpt_0001451.json")

# code version per session
for k in ("zds-smoke-s1", "zds-full-s1", "zds-full-s2", "zds-rescore-s2"):
    shas = sorted(set(re.findall(r"code git sha: ([0-9a-f]+(?:-dirty)?)", open(p(k, "kernel.log")).read())))
    check(f"{k} code git sha", shas, len(shas) == 1 and not shas[0].endswith("-dirty"), f"{k}/kernel.log")

# re-score outside training
if os.path.exists(p("rescore.json")):
    rs = js(p("rescore.json"))
    for amp in ("fp16", "fp32"):
        r = (rs.get(amp) or {}).get("result")
        if r:
            v = r["passes"][0]["val_loss"]
            check(f"re-score {amp} (fresh process)", f"{v:.4f} ({v - final['val_loss']:+.1e})",
                  r["val_set_is_upstream"] is True and v <= 3.28 and abs(v - final["val_loss"]) < 2e-3,
                  f"rescore.json: {amp}.result.passes[0].val_loss")
    r16 = rs["fp16"]["result"]
    check("re-scored checkpoint", (r16["ckpt"], r16["ckpt_step"]), r16["ckpt"] == fin["file"] and
          r16["ckpt_bytes"] == fin["bytes"], "rescore.json fp16.result.ckpt, ckpt_bytes vs final sidecar")
else:
    print("--  rescore.json not present yet")

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
