"""Smoke + unit tests: loss falls, checkpoints load, overflow steps are skipped, attention and the data
stream match their references."""
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import train_t4 as T
from conftest import train_losses

TINY = ["--tiny", "--synthetic", "--device", "cpu"]


# ---------------------------------------------------------------- end to end (shared tiny CPU run)
def test_loss_decreases(run_a):
    _, log = run_a
    losses = train_losses(log)
    assert sorted(losses) == list(range(60)), "expected one record per step 0..59"
    assert all(math.isfinite(v[0]) for v in losses.values())
    first = np.mean([losses[s][0] for s in range(5)])
    last = np.mean([losses[s][0] for s in range(55, 60)])
    assert last < first - 3.0, (first, last)
    val = [r for r in log if r.get("event") == "val"]
    assert len(val) == 1 and val[0]["step"] == 60 and val[0]["val_loss"] < math.log(4096) - 3.0


def test_log_proves_val_set_and_wall_clock(run_a):
    """Finding M1-7: the log names the val set (file, bytes, sha256s, token count) and counts val + checkpoint
    time in the session wall-clock, apart from compute-only time."""
    out, log = run_a
    start = [r for r in log if r.get("event") == "start"][0]
    assert start["val_file"] == "synthetic-stream3-000000" and start["val_tokens"] == 8192
    assert len(start["val_tokens_sha256"]) == 64 and start["val_set_is_upstream"] is False
    assert start["args"]["num_iters"] == 60 and start["train_files"][0][0] == "synthetic-stream2-000000"
    val = [r for r in log if r.get("event") == "val"][-1]
    assert val["val_tokens_sha256"] == start["val_tokens_sha256"] and val["val_tokens"] == 8192
    assert "upstream_val_loss" not in val  # tiny model: no upstream comparison
    assert val["val_s"] > 0 and val["wall_total_s"] >= val["compute_s"] + val["val_s"]
    ex = [r for r in log if r.get("event") == "exit"][-1]
    assert ex["reason"] == "done" and ex["ckpt_s"] > 0
    assert ex["wall_total_s"] >= ex["compute_s"] + ex["val_s"] + ex["ckpt_s"]
    assert ex["gpu_seconds_total"] == ex["wall_total_s"]  # world size 1
    _, meta, _ = T.load_latest(str(out / "ckpt"), "cpu")
    assert meta["wall_total_s"] >= meta["compute_s"] and meta["segments"][-1]["end_step"] == 60


def test_val_set_hashes_pin_upstream_val(tmp_path):
    """val_set_is_upstream needs the exact upstream file; the same token count from other data is not enough."""
    p = tmp_path / "fineweb_val_000000.bin"
    T.write_shard(str(p), np.arange(T.VAL_TOKENS_UPSTREAM + 1) % 50257)
    r = T.Runner(T.parse_args(["--device", "cpu", "--tiny", "--val-tokens", str(T.VAL_TOKENS_UPSTREAM),
                               "--val-files", str(p), "--train-files", str(p), "--out-dir", str(tmp_path)]))
    info = r.val_set_info()
    assert info["val_tokens"] == T.VAL_TOKENS_UPSTREAM and info["val_file_bytes"] == 1024 + 2 * (T.VAL_TOKENS_UPSTREAM + 1)
    assert info["val_set_is_upstream"] is False and len(info["val_file_sha256"]) == 64


def test_schedule_same_as_upstream():
    s = T.Schedule(T.parse_args(["--device", "cpu"]))  # 1450 steps: the window grows later from step 103 on
    assert s.same_as_upstream(102) and not s.same_as_upstream(103) and not s.same_as_upstream(250)
    assert T.Schedule(T.parse_args(["--device", "cpu", "--num-iters", "1390"])).same_as_upstream(1390)


def test_checkpoint_loads(run_a):
    out, _ = run_a
    ck, meta, _ = T.load_latest(str(out / "ckpt"), "cpu")
    assert ck is not None and ck["done"] and meta["next_step"] == 61 and meta["provider"] == "laptop"
    assert meta["tokens_seen"] == ck["tokens_seen"] == 60 * 2048 and meta["gpu_seconds"] > 0
    for key in ("model", "opt", "scaler", "loader", "rng"):
        assert key in ck
    assert ck["scaler"]["scale"] > 0 and len(ck["opt"]["muon"]) > 0 and len(ck["opt"]["adam"]) > 0
    a = T.parse_args(TINY)
    model = T.GPT(a.vocab_size, a.num_layers, a.model_dim)
    model.load_state_dict(ck["model"])
    model.eval()
    src = T.SyntheticSource(a.vocab_size, a.vocab_size - 1, a.seed, 4096, stream=3)
    toks = torch.from_numpy(src.shard(0)[:513].astype(np.int64))
    x, y = toks[:-1], toks[1:]
    seg = torch.cumsum((x == a.vocab_size - 1).long(), 0)
    with torch.no_grad():
        h = model.hidden(x, seg, dict(dtype=torch.float32, window_blocks=14, block=16, attn_chunk=128))
        ce, _ = T.ce_per_token(model, h, y, 256)
    assert ce.mean().item() < math.log(4096) - 3.0


def test_overflow_step_is_skipped(tmp_path):
    """A non-finite gradient must skip the update (weights and optimizer state untouched) and halve the scale."""
    r = T.Runner(T.parse_args(TINY + ["--scaler", "on", "--out-dir", str(tmp_path)]))
    r.train_step(0)
    before = {n: p.detach().clone() for n, p in r.model.named_parameters()}
    adam_steps = {id(p): s["step"] for p, s in r.opt.adam.items()}
    scale = r.scaler.get_scale()
    h = r.model.blocks[-1].register_forward_hook(lambda mod, inp, out: out * float("inf"))
    out = r.train_step(1)
    h.remove()
    assert out["skipped"] and r.overflow_skips == 1 and r.scaler.get_scale() == scale / 2
    for n, p in r.model.named_parameters():
        assert torch.equal(p, before[n]), n
    assert all(s["step"] == adam_steps[id(p)] for p, s in r.opt.adam.items())
    assert not r.train_step(2)["skipped"]


def test_uncommitted_checkpoints_are_ignored(tmp_path):
    d = str(tmp_path)
    torch.save({"x": 1}, os.path.join(d, "ckpt_0000009.pt"))  # no sidecar: never committed
    open(os.path.join(d, "ckpt_0000012.pt.tmp.123"), "wb").write(b"partial")  # killed mid-write
    good = T.save_checkpoint(d, {"next_step": 5, "v": torch.ones(3)}, {"next_step": 5}, keep=2)
    T.save_checkpoint(d, {"next_step": 7, "v": torch.ones(1000)}, {"next_step": 7}, keep=5)
    with open(os.path.join(d, "ckpt_0000007.pt"), "r+b") as f:  # committed, then truncated: size check rejects
        f.truncate(100)
    ck, meta, path = T.load_latest(d, "cpu")
    assert path == good and meta["next_step"] == 5 and torch.equal(ck["v"], torch.ones(3))
    assert not os.path.exists(os.path.join(d, "ckpt_0000012.pt.tmp.123"))  # leftovers are cleaned up


# ---------------------------------------------------------------- unit checks against references
def flex_reference(q, k, v, seg, window_blocks, block, scale):
    """Upstream's FlexAttention semantics as one dense mask: block-causal sliding window AND document_causal."""
    S = q.size(0)
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None]
    mask = (j <= i) & (seg[:, None] == seg[None]) & (i // block - j // block < window_blocks)
    return F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                          attn_mask=mask, scale=scale).transpose(0, 1)


@pytest.mark.parametrize("S,wb,block,chunk", [(512, 3, 16, 64), (300, 1, 16, 128), (256, 40, 16, 32),
                                              (1024, 14, 32, 256)])
def test_windowed_attention_matches_flex_mask(S, wb, block, chunk):
    g = torch.Generator().manual_seed(S + wb)
    q, k, v = (torch.randn(S, 3, 16, generator=g) for _ in range(3))
    seg = torch.cumsum((torch.rand(S, generator=g) < 0.01).long(), 0)
    ref = flex_reference(q, k, v, seg, wb, block, 0.25)
    out = T.windowed_attention(q, k, v, seg, wb, block, 0.25, chunk)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_chunked_validation_is_exact():
    """Validation may split a sequence into chunks with a receptive-field-sized prefix: same losses."""
    a = T.parse_args(TINY)
    torch.manual_seed(0)
    m = T.GPT(a.vocab_size, a.num_layers, a.model_dim).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    src = T.SyntheticSource(a.vocab_size, a.vocab_size - 1, 0, 4096, stream=3)
    t = torch.from_numpy(src.shard(0)[:1025].astype(np.int64))
    x, y = t[:-1], t[1:]
    seg = torch.cumsum((x == a.vocab_size - 1).long(), 0)
    rt = dict(dtype=torch.float32, window_blocks=2, block=16, attn_chunk=64)
    with torch.no_grad():
        full, _ = T.ce_per_token(m, m.hidden(x, seg, rt), y, 256)
        ctx = len(m.blocks) * 2 * 16
        h = m.hidden(x[512 - ctx:], seg[512 - ctx:], rt)[ctx:]
        part, _ = T.ce_per_token(m, h, y[512:], 256)
    torch.testing.assert_close(part, full[512:], atol=1e-5, rtol=1e-5)


def test_stream_loader_resume_and_shard_wrap():
    src = T.SyntheticSource(4096, 4095, 0, 10_000, stream=2, num_shards=2)
    wraps = []
    a = T.StreamLoader(src, allow_wrap=True, on_wrap=wraps.append)
    a.next_batch(2048)
    state = a.state_dict()
    assert state["shard_name"] == "synthetic-stream2-000000" and state["shard_bytes"] == 1024 + 20_000
    want = []
    for _ in range(12):  # crosses both shard boundaries (wraps to shard 0, like upstream, because allowed)
        toks, s = a.next_batch(2048)
        want.append((a.shard_idx, np.asarray(toks[s:s + 2049]).copy()))
    assert {i for i, _ in want} == {0, 1} and wraps == [1]
    b = T.StreamLoader(src, allow_wrap=True)
    b.load_state_dict(state)
    for idx, w in want:
        toks, s = b.next_batch(2048)
        assert b.shard_idx == idx and np.array_equal(np.asarray(toks[s:s + 2049]), w)


def test_stream_loader_refuses_wrap_and_foreign_shards(tmp_path):
    """Findings M1-5/6: running out of shards is an error unless allowed; a resume on another shard set too."""
    src = T.SyntheticSource(4096, 4095, 0, 10_000, stream=2, num_shards=2)
    assert T.steps_capacity(src, 2048) == 8
    a = T.StreamLoader(src)
    for _ in range(8):
        a.next_batch(2048)
    with pytest.raises(RuntimeError, match="out of train data"):
        a.next_batch(2048)
    for i in (1, 2):
        T.write_shard(str(tmp_path / f"fineweb_train_{i:06d}.bin"), src.shard(i - 1))
    files = T.FileSource(str(tmp_path / "fineweb_train_*.bin"))
    ld = T.StreamLoader(files)
    for _ in range(5):
        ld.next_batch(2048)
    state = ld.state_dict()
    assert state["shard_name"] == "fineweb_train_000002.bin" and state["shard_idx"] == 1
    os.remove(tmp_path / "fineweb_train_000001.bin")  # another session downloaded a different set
    other = T.StreamLoader(T.FileSource(str(tmp_path / "fineweb_train_*.bin")))
    with pytest.raises(SystemExit, match="resume refused"):
        other.load_state_dict(state)


def test_run_refuses_more_steps_than_data(tmp_path, monkeypatch):
    import signal
    monkeypatch.setattr(signal, "signal", lambda *a: None)  # keep pytest's own SIGINT/SIGTERM handlers
    r = T.Runner(T.parse_args(TINY + ["--num-iters", "2100", "--out-dir", str(tmp_path)]))  # capacity 4 x 511
    with pytest.raises(SystemExit, match="refusing to start"):
        r.run()
    r = T.Runner(T.parse_args(TINY + ["--num-iters", "2100", "--stop-after-step", "0", "--out-dir", str(tmp_path)]))
    r.run()  # a run stopped early only needs the steps it trains


def test_shard_roundtrip(tmp_path):
    toks = np.random.default_rng(0).integers(0, 50257, 1000).astype(np.uint16)
    p = str(tmp_path / "fineweb_train_000001.bin")
    T.write_shard(p, toks)
    assert np.array_equal(np.asarray(T.read_shard(p)), toks)


def test_schedules_match_upstream():
    s = T.Schedule(T.parse_args(["--device", "cpu", "--num-iters", "1390"]))
    assert s.lr(0) == 1.0 and s.lr(833) == 1.0 and abs(s.lr(1389) - (1 / 1390) / 0.4) < 1e-12
    assert s.window_blocks(0) == 1 and s.window_blocks(1390) == 14
    assert s.muon_momentum(0) == 0.85 and s.muon_momentum(300) == 0.95


def test_default_run_length_has_margin():
    """Finding M1-1: default 1450 steps (not upstream's 1390); same schedule shape, stretched."""
    a = T.parse_args(["--device", "cpu"])
    assert a.num_iters == 1450 and a.num_iters * a.batch_tokens == 760_217_600
    s = T.Schedule(a)
    cool = int(1450 * (1 - T.COOLDOWN_FRAC))  # 870: cooldown is the last 40% of the new total, as upstream
    assert s.lr(cool) == 1.0 and s.lr(cool + 1) < 1.0 and abs(s.lr(1449) - (1 / 1450) / 0.4) < 1e-12
    assert s.window_blocks(0) == 1 and s.window_blocks(1450) == 14
    assert s.momentum_warmup == 300  # upstream's fixed Muon warmup is not stretched
    assert 1450 <= 8 * 190  # still fits the 8 train shards the lane downloads (190 steps/shard)


def test_dry_shape_tiny():
    rep = T.main(TINY + ["--dry-shape"])
    assert rep["params"] > 0 and rep["tokens"] == 1024 and math.isfinite(rep["loss"]) and rep["peak_memory_mb"] > 0


def test_overflow_streak_aborts_nonzero(tmp_path):
    """Finding M1-2: a run whose every step overflows must exit non-zero after N skips, log why, raise a scale
    alarm on the way down, and leave the last good checkpoint as the newest one."""
    import subprocess
    from conftest import env, read_log, trainer_cmd
    cmd = trainer_cmd(tmp_path, "--ckpt-every-steps", "5", "--inject-overflow-from", "10",
                      "--max-consecutive-skips", "6", "--scale-floor", "4096")
    r = subprocess.run(cmd, env=env(), capture_output=True, text=True, timeout=600)
    assert r.returncode == T.EXIT_OVERFLOW, (r.returncode, r.stderr[-2000:])
    assert "ABORT: 6 consecutive overflow-skipped steps (10..15" in r.stderr
    log = read_log(tmp_path / "log.jsonl")
    steps = {x["step"]: x for x in log if "train_loss" in x}
    assert max(steps) == 15 and not any(steps[i]["skipped"] for i in range(10))
    assert all(steps[i]["skipped"] and steps[i]["nonfinite_fwd"] for i in range(10, 16))
    assert steps[15]["consec_skips"] == 6
    alarms = [x for x in log if x.get("event") == "scale_alarm"]
    assert len(alarms) == 1 and alarms[0]["step"] == 14 and alarms[0]["scale"] == 2048.0  # 65536 / 2^5
    abort = [x for x in log if x.get("event") == "abort"]
    assert len(abort) == 1 and abort[0]["nonfinite_fwd_in_streak"] == 6
    assert [x["reason"] for x in log if x.get("event") == "exit"] == ["overflow_abort"]
    _, meta, _ = T.load_latest(str(tmp_path / "ckpt"), "cpu")
    assert meta["next_step"] == 10  # the step-14 periodic checkpoint was suppressed mid-streak
    assert "c_fc_amax" in steps[0] and len(steps[0]["c_fc_amax_layers"]) == 12


def test_mlp_prescale_is_the_same_math():
    torch.manual_seed(0)
    mlp = T.MLP(32)
    with torch.no_grad():
        mlp.c_proj.weight.normal_()
    x = 3 * torch.randn(64, 32)
    ref = mlp(x)
    mlp.prescale = True
    torch.testing.assert_close(mlp(x), ref, rtol=1e-6, atol=1e-5)


# ---------------------------------------------------------------- micro-batch pieces vs upstream's loader
class UpstreamDistributedDataLoader:
    """Record #18's DistributedDataLoader, verbatim except: tokens stay on the CPU (no pin_memory/cuda), and
    rank/world are arguments instead of os.environ."""

    def __init__(self, filename_pattern, rank, world_size):
        import glob
        self.rank = rank
        self.world_size = world_size
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = torch.from_numpy(np.asarray(T.read_shard(self.files[self.current_shard])).astype(np.int64))

    def next_batch(self, batch_size):
        assert batch_size % self.world_size == 0
        device_batch_size = batch_size // self.world_size
        # load next shard if necessary
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        pos = self.current_position + self.rank * device_batch_size
        device_batch_tokens = self.tokens[pos:pos+device_batch_size+1]
        # advance current position
        self.current_position += batch_size
        inputs = device_batch_tokens[:-1]
        targets = device_batch_tokens[1:]
        return inputs, targets


@pytest.mark.parametrize("world,micro,doc_ctx", [(1, 256, 0), (2, 256, 0), (2, 512, 256)])
def test_pieces_are_upstreams_tokens(tmp_path, world, micro, doc_ctx):
    """Finding M1-8: our per-rank pieces, concatenated, are exactly the inputs/targets upstream's loader gives
    each rank, step after step, across shard boundaries."""
    src = T.SyntheticSource(4096, 4095, 0, 10_000, stream=2, num_shards=3)
    for i in range(3):
        T.write_shard(str(tmp_path / f"fineweb_train_{i + 1:06d}.bin"), src.shard(i))
    pattern = str(tmp_path / "fineweb_train_*.bin")
    B, seq_len = 2048, 1024
    ours = T.StreamLoader(T.FileSource(pattern))
    ups = [UpstreamDistributedDataLoader(pattern, r, world) for r in range(world)]
    for step in range(11):  # 4 steps per 10,000-token shard: crosses two shard boundaries
        toks, start = ours.next_batch(B)
        for r in range(world):
            x_up, y_up = ups[r].next_batch(B)
            plan = T.piece_plan(toks, start, r, world, B, seq_len, micro, 16, 4095, doc_ctx=doc_ctx)
            x = np.concatenate([np.asarray(toks[s:s + micro]) for s, _ in plan])
            y = np.concatenate([np.asarray(toks[s + 1:s + micro + 1]) for s, _ in plan])
            assert np.array_equal(x, x_up.numpy()) and np.array_equal(y, y_up.numpy()), (step, r)
            for s, c in plan:  # context never reaches before the piece's own 64K-style sequence
                assert c % 16 == 0 and s - c >= start + ((s - start) // seq_len) * seq_len


def test_doc_ctx_lengths():
    bos, blk = 9, 16
    toks = np.zeros(4096, dtype=np.uint16)
    toks[[0, 300, 1100]] = bos
    plan = dict(T.piece_plan(toks, 0, 0, 1, 2048, 2048, 256, blk, bos, doc_ctx=512))
    assert plan[0] == 0                      # sequence start: nothing before it
    assert plan[256] == 256                  # doc 0 began at 0, within the cap: back to it
    assert plan[512] == 512 - 288            # doc at 300 -> its block starts at 288
    assert plan[1024] == 512                 # doc at 300 is > 512 back: capped (truncated piece)
    assert plan[1280] == 1280 - 1088         # doc at 1100 -> block 1088
    assert dict(T.piece_plan(toks, 0, 0, 1, 2048, 2048, 256, blk, bos, train_ctx=128))[1024] == 128


def test_doc_ctx_is_exact():
    """With --doc-ctx, every piece whose document began within the cap gets exactly the losses of a forward over
    the whole sequence (upstream); without context, some pieces do not."""
    a = T.parse_args(TINY)
    torch.manual_seed(0)
    m = T.GPT(a.vocab_size, a.num_layers, a.model_dim).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    bos = a.vocab_size - 1
    toks = T.SyntheticSource(a.vocab_size, bos, 0, 4096, stream=2).shard(0)[:1025]
    t = torch.from_numpy(toks.astype(np.int64))
    rt = dict(dtype=torch.float32, window_blocks=14, block=16, attn_chunk=128)

    def ce(lo, s, hi):
        x = t[lo:hi]
        seg = torch.cumsum((x == bos).long(), 0)
        with torch.no_grad():
            return T.ce_per_token(m, m.hidden(x, seg, rt)[s - lo:], t[s + 1:hi + 1], 256)[0]

    full = ce(0, 0, 1024)
    differs = 0
    for doc_ctx in (0, 512):
        for s, c in T.piece_plan(toks, 0, 0, 1, 1024, 1024, 256, 16, bos, doc_ctx=doc_ctx):
            got = ce(s - c, s, s + 256)
            if doc_ctx:
                assert (toks[s - c:s + 1] == bos).any() or s - c == 0  # every synthetic doc is < 512 tokens
                torch.testing.assert_close(got, full[s:s + 256], atol=1e-5, rtol=1e-5)
            else:
                differs += not torch.allclose(got, full[s:s + 256], atol=1e-5, rtol=1e-5)
    assert differs > 0


def test_round_embeds_bf16():
    r = T.Runner(T.parse_args(TINY + ["--round-embeds-bf16"]))
    r.train_step(0)
    for e in [r.model.embed, *r.model.value_embeds]:
        assert torch.equal(e.weight, e.weight.bfloat16().float())
    assert not torch.equal(r.model.lm_head.weight, r.model.lm_head.weight.bfloat16().float())
