#!/usr/bin/env python3
"""Zero-Dollar Speedrun: a single-file, T4-safe, crash-proof port of modded-nanogpt record #18.

Upstream: github.com/KellerJordan/modded-nanogpt @ 4ea6b937337a4889b8cfe3f38a93d120048d8f71,
records/track_1_short/2025-01-04_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt ("SoftCap", 2025-01-04:
3.2785 val loss after 1390 steps x 524,288 tokens = 728.8M tokens on 8xH100). trainer/PORTING.md lists every
upstream feature as kept / changed / dropped.

T4 (SM75) changes: no bf16 (fp16 autocast + torch.amp.GradScaler, fp32 master weights and optimizer state,
Muon's Newton-Schulz in fp32), no FlexAttention (torch SDPA with an exact block-sliding-window + document mask),
torch.compile off by default, gradient accumulation instead of 8 ranks. Added: atomic checkpoints of the full
training state every N minutes and on SIGTERM, and --resume that continues (bit-exactly on CPU) from the newest
valid checkpoint, so a run can hop across killed free-tier sessions.

  CPU test:   python trainer/train_t4.py --tiny --synthetic --device cpu
  One T4:     python trainer/train_t4.py --device cuda --provider kaggle --resume
  Two T4s:    torchrun --standalone --nproc_per_node=2 trainer/train_t4.py --provider kaggle --resume
"""
import time

_T_PROCESS = time.time()  # the session's wall-clock counter starts here, before the slow imports
import argparse
import glob
import hashlib
import json
import math
import os
import random
import signal
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_COMMIT = "4ea6b937337a4889b8cfe3f38a93d120048d8f71"
UPSTREAM_RECORD = "records/track_1_short/2025-01-04_SoftCap (record #18)"
BLOCK = 128  # FlexAttention block size upstream; the sliding window moves in whole blocks
# The val set (record #18): the first 10,485,760 tokens of fineweb_val_000000.bin (HF kjj0/fineweb10B-gpt2), read
# as tokens [0, 10,485,761) for the shifted targets. Pinned sha256s: of the whole file, and of that token slice
# (uint16 little-endian). Every run logs its own values next to these.
VAL_TOKENS_UPSTREAM = 10_485_760
VAL_FILE_SHA256 = "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640"
VAL_SLICE_SHA256 = "ce6ad9a79700ee93c1e3edd25ecdcb06e279a958b5a3841c1dd538ef8819e375"
# Upstream's val curve in the #18 log (step: val_loss), for parity checks while our schedule equals upstream's.
UPSTREAM_VAL = {0: 10.8258, 125: 4.3686, 250: 3.9427, 375: 3.7671, 500: 3.6528, 625: 3.5728, 750: 3.5205,
                875: 3.4715, 1000: 3.4056, 1125: 3.3537, 1250: 3.3076, 1375: 3.2793, 1390: 3.2785}
EXIT_OVERFLOW = 3  # exit code: too many consecutive fp16 overflow-skipped steps (no checkpoint written)
EXIT_RANK_DRIFT = 4  # exit code: the ranks' weights differ at a checkpoint (only rank 0's are saved)

# =============================================================================================
# Configuration
# =============================================================================================
# num_iters: upstream #18 ran 1390 steps, but its 80 reruns average 3.2791 +- 0.0019 and only 55/80 reach <= 3.28
# (records/track_1_short/2025-01-04_SoftCap/README.md). We get one run, so the default is 1450 steps (760.2M tokens,
# +4.3%): at ~0.00015 nats/step late in training that is ~-0.009 of margin for port losses. The LR cooldown (last
# 40%) and the window schedule are fractions of num_iters, so they stretch with it; --num-iters 1390 = upstream.
UPSTREAM_ITERS = 1390
FULL = dict(vocab_size=50257, model_dim=768, num_layers=12, num_iters=1450, batch_tokens=8 * 64 * 1024,
            seq_len=64 * 1024, micro_tokens=8192, block=BLOCK, attn_chunk=1024, val_tokens=10_485_760,
            val_chunk=64 * 1024, loss_chunk=4096, synthetic_shard_tokens=1 << 24)
# --tiny: same 12-layer topology at width 96, 2K-token steps, 16-token window blocks; for CPU tests.
TINY = dict(vocab_size=4096, model_dim=96, num_layers=12, num_iters=60, batch_tokens=2048, seq_len=1024,
            micro_tokens=1024, block=16, attn_chunk=128, val_tokens=8192, val_chunk=1024, loss_chunk=512,
            synthetic_shard_tokens=1 << 20)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--device", choices=["cpu", "cuda"], default=None, help="default: cuda if available")
    a("--tiny", action="store_true", help="small CPU test preset (12 layers, d=96, 1K-token sequences)")
    a("--synthetic", action="store_true", help="deterministic synthetic token shards instead of FineWeb")
    a("--data-dir", default="data/fineweb10B")
    a("--train-files", default=None, help="glob; default <data-dir>/fineweb_train_*.bin")
    a("--val-files", default=None, help="glob; default <data-dir>/fineweb_val_*.bin")
    for k in FULL:
        a("--" + k.replace("_", "-"), type=int, default=None)
    a("--train-ctx", type=int, default=0,
      help="unscored tokens prepended to each micro-batch so attention near its start sees the context it "
           "would see inside upstream's 64K sequence (multiple of --block; 0 = off)")
    a("--doc-ctx", type=int, default=0,
      help="doc-aligned context: prepend the unscored tokens from the block-aligned start of the document "
           "containing each micro-batch's first token, capped at N (multiple of --block, e.g. 2048; 0 = off). "
           "Exact at every layer when that document began within N tokens. Excludes --train-ctx")
    a("--val-every", type=int, default=0, help="validate every N steps (upstream 125; 0 = final only)")
    a("--max-consecutive-skips", type=int, default=20,
      help=f"exit with code {EXIT_OVERFLOW} after this many consecutive overflow-skipped steps, without "
           "checkpointing, so the newest checkpoint stays the last good one (0 = never)")
    a("--scale-floor", type=float, default=1.0, help="log a scale_alarm when the GradScaler scale drops below this")
    a("--amax-every", type=int, default=25,
      help="every N steps log max |c_fc output| per layer (the ReLU^2 input; fp16 overflows above ~255.9); "
           "0 = off; ignored with --compile")
    a("--mlp-prescale", action="store_true",
      help="compute c_proj((relu(x)/4)^2) * 16 in fp32: same math (powers of 2), but fp16 headroom for MLP "
           "pre-activations up to ~1023 instead of ~255.9")
    a("--rank-check-tol", type=float, default=1e-4,
      help=f"at each checkpoint compare every parameter's |sum| across ranks; exit with code {EXIT_RANK_DRIFT} if "
           "the largest relative difference exceeds this")
    a("--round-embeds-bf16", action="store_true",
      help="round the embedding tables to bf16 after each Adam step, like upstream's bf16_embeds (late in the "
           "cooldown upstream's bf16 tables stop moving; our fp32 ones keep moving). Off by default")
    a("--inject-overflow-from", type=int, default=-1, help=argparse.SUPPRESS)  # tests: inf loss from this step on
    a("--amp", choices=["fp16", "fp32"], default=None, help="default fp16 on cuda, fp32 on cpu")
    a("--scaler", choices=["on", "off"], default=None, help="GradScaler; default on iff --amp fp16")
    a("--init-scale", type=float, default=2.0 ** 16)
    a("--compile", action="store_true", help="torch.compile the model forward (off by default; see PORTING.md)")
    a("--sdpa-pad", choices=["auto", "on", "off"], default="auto", help="(no-op for #18: q, k, v widths match)")
    a("--seed", type=int, default=0)
    a("--threads", type=int, default=8)
    a("--deterministic", action="store_true")
    a("--out-dir", default="out/run")
    a("--log", default=None, help="jsonl log; default <out-dir>/log.jsonl")
    a("--ckpt-dir", default=None, help="default <out-dir>/ckpt")
    a("--ckpt-every-min", type=float, default=20.0)
    a("--ckpt-every-steps", type=int, default=0)
    a("--keep-ckpts", type=int, default=2)
    a("--resume", action="store_true", help="continue from the newest valid checkpoint, if any")
    a("--provider", default="laptop", help="kaggle|colab|lightning|laptop|... (logged per segment)")
    a("--allow-wrap", action="store_true",
      help="when the train shards run out, wrap to the first one (repeats data; logged as data_wrap). Default: "
           "refuse to start if the shards cannot supply the run, and error out if they run out")
    a("--time-budget-min", type=float, default=0, help="checkpoint and exit 0 after this much wall time")
    a("--stop-after-step", type=int, default=-1, help="checkpoint and exit after training this step")
    a("--canon-mask", choices=["on", "off"], default="off",
      help="also report val loss with record #91's canonical-pair mask (needs tiktoken; not part of #18)")
    a("--dry-shape", action="store_true",
      help="build the model, run one forward/backward + optimizer step on random tokens, print params and peak "
           "memory, exit (no data needed)")
    a("--dry-shape-tokens", type=int, default=None, help="sequence length for --dry-shape (default: 1024 on "
      "cpu, --micro-tokens on cuda)")
    a("--scale-step-consts", choices=["auto", "on", "off"], default="auto",
      help="scale the 300-step Muon momentum warmup to the run length (auto: on for --tiny; off for the full run, "
           "where upstream's fixed 300 steps is kept and only the cooldown and window schedule scale with "
           "--num-iters)")
    args = p.parse_args(argv)
    preset = TINY if args.tiny else FULL
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp is None:
        args.amp = "fp16" if args.device == "cuda" else "fp32"
    if args.scaler is None:
        args.scaler = "on" if args.amp == "fp16" else "off"
    if args.log is None:
        args.log = os.path.join(args.out_dir, "log.jsonl")
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(args.out_dir, "ckpt")
    if args.train_files is None:
        args.train_files = os.path.join(args.data_dir, "fineweb_train_*.bin")
    if args.val_files is None:
        args.val_files = os.path.join(args.data_dir, "fineweb_val_*.bin")
    assert args.train_ctx % args.block == 0, "--train-ctx must be a multiple of --block"
    assert args.doc_ctx % args.block == 0, "--doc-ctx must be a multiple of --block"
    assert not (args.train_ctx and args.doc_ctx), "--train-ctx and --doc-ctx are alternatives; pick one"
    return args


# Fields that must match between a checkpoint and the run resuming it.
ARCH_KEYS = ("vocab_size", "model_dim", "num_layers", "num_iters", "batch_tokens", "seq_len", "block",
             "synthetic", "seed", "scale_step_consts")


def next_multiple(v, n):
    return math.ceil(v / n) * n


# =============================================================================================
# Attention: exact block-sliding-window + document-causal attention on torch SDPA
# =============================================================================================
def windowed_attention(q, k, v, seg, window_blocks, block, scale, chunk):
    """Query i sees key j iff j <= i, seg[i] == seg[j] and i // block - j // block < window_blocks.

    This is upstream's FlexAttention mask (document_causal mask_mod over the block-level causal sliding
    window) for a sequence of any length. Queries go in chunks of `chunk`; each chunk attends to its own keys
    plus the window_blocks * block keys before it, under a boolean mask (SDPA memory-efficient kernel on SM75).
    q, k: [S, h, dq]; v: [S, h, dv] -> [S, h, dv].
    """
    S, h, _ = q.shape
    C = min(chunk, S)
    nb = -(-S // C)
    P = nb * C - S
    K = window_blocks * block
    dev = q.device
    qb = F.pad(q, (0, 0, 0, 0, 0, P)).view(nb, C, h, -1).transpose(1, 2)

    def windows(t):  # [S, h, d] -> [nb, h, C + K, d]
        return F.pad(t, (0, 0, 0, 0, K, P)).unfold(0, C + K, C).permute(0, 1, 3, 2)

    kb, vb = windows(k), windows(v)
    pos_k = torch.arange(-K, nb * C, device=dev).unfold(0, C + K, C)
    seg_k = torch.cat([torch.full((K,), -3, device=dev, dtype=seg.dtype), seg,
                       torch.full((P,), -2, device=dev, dtype=seg.dtype)]).unfold(0, C + K, C)
    pos_q = torch.arange(nb * C, device=dev).view(nb, C)
    seg_q = torch.cat([seg, torch.full((P,), -2, device=dev, dtype=seg.dtype)]).view(nb, C)
    pq, pk = pos_q[:, :, None], pos_k[:, None, :]
    mask = (pk <= pq) & (pq.div(block, rounding_mode="floor") - pk.div(block, rounding_mode="floor") < window_blocks) \
        & (seg_q[:, :, None] == seg_k[:, None, :])
    y = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask[:, None], scale=scale)
    return y.transpose(1, 2).reshape(nb * C, h, -1)[:S]


# =============================================================================================
# The model (record #18, in plain torch with fp32 master weights)
# =============================================================================================
def norm(x):
    return F.rms_norm(x.float(), (x.size(-1),)).to(x.dtype)


class Rotary(nn.Module):
    """Half-truncated RoPE (upstream): dim/4 frequencies (1/1024)^[0..1], then dim/4 zero frequencies."""

    def __init__(self, dim):
        super().__init__()
        f = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([f, f.new_zeros(dim // 4)]), persistent=False)

    def forward(self, x):  # x: [S, h, d]
        t = torch.arange(x.size(0), dtype=torch.float32, device=x.device)
        theta = torch.outer(t, self.angular_freq)
        cos, sin = theta.cos()[:, None, :], theta.sin()[:, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), -1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x, ve, seg, rt):
        S, H = x.size(0), self.num_heads
        q = self.c_q(x).view(S, H, -1)
        k = self.c_k(x).view(S, H, -1)
        v = self.c_v(x).view(S, H, -1)
        v = self.lambdas[0] * v + (self.lambdas[1] * ve.view_as(v) if ve is not None else 0)
        q, k = norm(q), norm(k)  # QK norm
        q, k = self.rotary(q), self.rotary(k)
        dt = rt["dtype"]
        y = windowed_attention(q.to(dt), k.to(dt), v.to(dt), seg, rt["window_blocks"], rt["block"],
                               q.size(-1) ** -0.5, rt["attn_chunk"])
        return self.c_proj(y.reshape(S, -1))


class MLP(nn.Module):
    prescale = False  # --mlp-prescale

    def __init__(self, dim):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        h = F.relu(self.c_fc(x))
        if self.prescale:  # (h/4)^2 cannot overflow fp16 until h > ~1023; *16 restores the scale in fp32
            return self.c_proj((h * 0.25).square()).float() * 16
        return self.c_proj(h.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads) if use_attn else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, ve, x0, seg, rt):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, seg, rt)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads=6):
        super().__init__()
        self.vocab_size = next_multiple(vocab_size, 128)
        V = self.vocab_size
        self.embed = nn.Embedding(V, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=(i != 7)) for i in range(num_layers)])
        self.value_embeds = nn.ModuleList([nn.Embedding(V, model_dim) for _ in range(3)])
        self.lm_head = nn.Linear(model_dim, V, bias=False)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.canon_mask = None  # optional [V, V/8] uint8 (record #91), built on demand for validation

    def hidden(self, inputs, seg, rt):
        """Final normed hidden state [S, D] for one sequence. inputs [S] int64, seg [S] document ids."""
        x0 = norm(self.embed(inputs))
        ve = [emb(inputs) for emb in self.value_embeds]
        n = len(self.blocks)
        ve = [ve[0], ve[1], ve[2]] + [None] * (n - 6) + [ve[0], ve[1], ve[2]]  # 012 ... 012
        x = x0
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve[i], x0, seg, rt)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            j = self.num_encoder_layers + i
            x = self.blocks[j](x, ve[j], x0, seg, rt)
        return norm(x)


def _softcap_logits(h, w):
    return 15 * torch.tanh(F.linear(h, w).float() / 15)


def ce_per_token(model, h, targets, chunk, inputs=None):
    """Softcapped CE per token, computed in row chunks so [rows, vocab] fp32 logits stay small.
    Training: chunks are checkpointed (logits recomputed in backward). Returns (ce, ce_canonical_or_None)."""
    w = model.lm_head.weight
    out, masked = [], []
    use_mask = model.canon_mask is not None and inputs is not None and not torch.is_grad_enabled()
    shifts = torch.arange(8, dtype=torch.uint8, device=h.device)
    for lo in range(0, h.size(0), chunk):
        hi = min(h.size(0), lo + chunk)
        t = targets[lo:hi]
        if torch.is_grad_enabled():
            ce = torch.utils.checkpoint.checkpoint(
                lambda hh, ww, tt: F.cross_entropy(_softcap_logits(hh, ww), tt, reduction="none"),
                h[lo:hi], w, t, use_reentrant=False)
        else:
            z = _softcap_logits(h[lo:hi], w)
            ce = F.cross_entropy(z, t, reduction="none")
            if use_mask:
                V = z.size(-1)
                dropped = (model.canon_mask[inputs[lo:hi], :, None] >> shifts & 1).view(hi - lo, V).bool()
                masked.append(F.cross_entropy(z.masked_fill(dropped, -60.0), t, reduction="none"))
        out.append(ce)
    return torch.cat(out), (torch.cat(masked) if use_mask else None)


# =============================================================================================
# Schedules and optimizers (record #18: Adam for embeddings/head/scalars, Muon for hidden matrices)
# =============================================================================================
COOLDOWN_FRAC = 0.4
ADAM_BETAS, ADAM_EPS = (0.8, 0.95), 1e-8
LR_EMBED, LR_HEAD, LR_SCALAR, LR_MUON = 0.6, 0.008, 0.04, 0.05
MUON_MOMENTUM_WARMUP = 300
NS_COEFFS = (3.4445, -4.7750, 2.0315)


class Schedule:
    def __init__(self, a):
        self.n = a.num_iters
        scale = {"on": True, "off": False, "auto": a.tiny}[a.scale_step_consts]
        self.momentum_warmup = max(1, round(MUON_MOMENTUM_WARMUP * self.n / UPSTREAM_ITERS)) if scale else MUON_MOMENTUM_WARMUP

    def lr(self, it):
        """Stable then linear decay (upstream get_lr). Upstream's LambdaLR is stepped after every step but the
        last, so the update of step `it` uses lr(it)."""
        t = 1 - it / self.n
        return 1.0 if t >= COOLDOWN_FRAC else t / COOLDOWN_FRAC

    def window_blocks(self, it):
        """Sliding window 128 -> 1792 tokens, linear over training in whole blocks (upstream)."""
        x = it / self.n
        return int(((1 - x) * 128 + x * 1856) // 128)

    def muon_momentum(self, it):
        frac = min(it / self.momentum_warmup, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def same_as_upstream(self, it):
        """True if every schedule value up to step `it` equals upstream's 1390-step schedule (so val at `it`
        compares 1:1 with upstream's log)."""
        up = Schedule(argparse.Namespace(num_iters=UPSTREAM_ITERS, scale_step_consts="off", tiny=False))
        return all(self.lr(i) == up.lr(i) and self.window_blocks(i) == up.window_blocks(i)
                   and self.muon_momentum(i) == up.muon_momentum(i) for i in range(it + 1))


def zeropower_via_newtonschulz5(G, steps=5):
    """Upstream's quintic Newton-Schulz orthogonalization, run in fp32 (upstream: bf16; T4 has no bf16)."""
    a, b, c = NS_COEFFS
    X = G.float()
    tall = G.size(0) > G.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if tall else X


class GradBag(torch.optim.Optimizer):
    """Hands every gradient of a step to torch.amp.GradScaler (unscale + one inf/NaN check for all of them).
    step() does nothing: GradScaler.step returning None means the step overflowed and must be skipped."""

    def __init__(self, params):
        super().__init__(params, {})

    def step(self, closure=None):
        return True


class Optimizers:
    """Adam (embed 0.6, head 0.008, scalars 0.04; betas 0.8/0.95; no weight decay) + Muon (0.05, Nesterov,
    momentum 0.85 -> 0.95 over 300 steps) with all state in fp32. Called with unscaled gradients only."""

    def __init__(self, model):
        self.model = model
        self.hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
        embed = [model.embed.weight, *[e.weight for e in model.value_embeds]]
        scalars = [p for p in model.parameters() if p.ndim < 2]
        self.adam_groups = [(embed, LR_EMBED), ([model.lm_head.weight], LR_HEAD), (scalars, LR_SCALAR)]
        n_all = sum(1 for _ in model.parameters())
        assert len(self.hidden) + len(embed) + 1 + len(scalars) == n_all, "every parameter needs an optimizer"
        self.adam = {p: dict(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                     for g, _ in self.adam_groups for p in g}
        self.muon = {p: torch.zeros_like(p) for p in self.hidden}

    round_embeds_bf16 = False  # --round-embeds-bf16

    @torch.no_grad()
    def step(self, lr_mult, momentum):
        b1, b2 = ADAM_BETAS
        for gi, (params, base_lr) in enumerate(self.adam_groups):  # group 0: embed + value embeds
            lr = base_lr * lr_mult
            for p in params:
                st = self.adam[p]
                st["step"] += 1
                t = st["step"]
                st["exp_avg"].lerp_(p.grad, 1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denom = (st["exp_avg_sq"] / (1 - b2 ** t)).sqrt_().add_(ADAM_EPS)
                p.addcdiv_(st["exp_avg"], denom, value=-lr / (1 - b1 ** t))
                if self.round_embeds_bf16 and gi == 0:
                    p.copy_(p.bfloat16())
        lr = LR_MUON * lr_mult
        for p in self.hidden:
            buf = self.muon[p]
            buf.lerp_(p.grad, 1 - momentum)
            g = p.grad.lerp(buf, momentum)  # Nesterov
            p.add_(zeropower_via_newtonschulz5(g), alpha=-lr * max(1, p.size(0) / p.size(1)) ** 0.5)

    def state_dict(self):
        names = {p: n for n, p in self.model.named_parameters()}
        return dict(adam={names[p]: s for p, s in self.adam.items()}, muon={names[p]: b for p, b in self.muon.items()})

    def load_state_dict(self, s):
        params = dict(self.model.named_parameters())
        for n, st in s["adam"].items():
            mine = self.adam[params[n]]
            mine["step"] = st["step"]
            mine["exp_avg"].copy_(st["exp_avg"])
            mine["exp_avg_sq"].copy_(st["exp_avg_sq"])
        for n, b in s["muon"].items():
            self.muon[params[n]].copy_(b)


# =============================================================================================
# Data: upstream FineWeb uint16 .bin shards read as one contiguous stream (record #18's loader)
# =============================================================================================
SHARD_MAGIC, SHARD_VERSION, HEADER_INTS = 20240520, 1, 256


def read_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == SHARD_MAGIC, f"magic number mismatch in {path}"
    assert header[1] == SHARD_VERSION, "unsupported version"
    n = int(header[2])
    assert os.path.getsize(path) == HEADER_INTS * 4 + 2 * n, "number of tokens does not match header"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(n,))


def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:3] = (SHARD_MAGIC, SHARD_VERSION, len(tokens))
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


class FileSource:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no files match {pattern} (run scripts/get_data.py, or use --synthetic)")

    def __len__(self):
        return len(self.files)

    def shard(self, idx):
        return read_shard(self.files[idx])

    def name(self, idx):
        return os.path.basename(self.files[idx])

    def nbytes(self, idx):
        return os.path.getsize(self.files[idx])

    def sha256(self, idx):
        h = hashlib.sha256()
        with open(self.files[idx], "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        return h.hexdigest()

    def num_tokens(self, idx):
        return int(np.fromfile(self.files[idx], dtype=np.int32, count=HEADER_INTS)[2])


class SyntheticSource:
    """Deterministic documents built from a Zipf-weighted phrase book: structured enough that loss falls."""

    def __init__(self, vocab_size, bos, seed, shard_tokens, stream, num_shards=4):
        self.vocab, self.bos, self.seed, self.n, self.stream, self.num = vocab_size, bos, seed, shard_tokens, stream, num_shards
        self.phrases = np.random.default_rng([seed, 1]).integers(0, bos, size=(512, 8))
        w = 1.0 / np.arange(1, 513)
        self.p = w / w.sum()

    def __len__(self):
        return self.num

    def name(self, idx):
        return f"synthetic-stream{self.stream}-{idx:06d}"

    def nbytes(self, idx):
        return HEADER_INTS * 4 + 2 * self.n

    def num_tokens(self, idx):
        return self.n

    def shard(self, idx):
        rng = np.random.default_rng([self.seed, self.stream, idx])
        out, total = [], 0
        while total < self.n:
            lens = rng.integers(2, 48, size=256)
            ids = rng.choice(512, size=int(lens.sum()), p=self.p)
            body = self.phrases[ids].reshape(-1)
            noise = rng.random(body.size) < 0.05
            body = np.where(noise, rng.integers(0, self.bos, size=body.size), body)
            toks = np.insert(body, np.concatenate([[0], np.cumsum(lens * 8)[:-1]]), self.bos)
            out.append(toks)
            total += toks.size
        return np.concatenate(out)[:self.n].astype(np.uint16)


def steps_in_shard(n_tokens, batch_tokens):
    """Batches StreamLoader takes from one shard: it moves on when pos + batch + 1 >= n_tokens (upstream's rule)."""
    return max(0, (n_tokens - batch_tokens - 2) // batch_tokens + 1)


def steps_capacity(source, batch_tokens):
    return sum(steps_in_shard(source.num_tokens(i), batch_tokens) for i in range(len(source)))


class StreamLoader:
    """Upstream DistributedDataLoader: one contiguous token stream; a step takes the next batch_tokens (+1)
    tokens, moving to the next shard when the current one cannot supply them. Unlike upstream it does not wrap
    to the first shard unless allow_wrap (repeated data would inflate "tokens seen").
    State = (shard, position) plus the shard's file name and byte size, so a resume on a different shard set
    is refused instead of silently reading other tokens."""

    def __init__(self, source, allow_wrap=False, on_wrap=None):
        self.source, self.allow_wrap, self.on_wrap = source, allow_wrap, on_wrap
        self.shard_idx, self.pos, self.wraps = 0, 0, 0
        self._load()

    def _load(self):
        self.tokens = self.source.shard(self.shard_idx)

    def state_dict(self):
        i = self.shard_idx
        return dict(shard_idx=i, pos=self.pos, shard_name=self.source.name(i), shard_bytes=self.source.nbytes(i),
                    num_shards=len(self.source), wraps=self.wraps)

    def load_state_dict(self, s):
        i = s["shard_idx"]
        if "shard_name" in s:
            want = (s["shard_name"], s["shard_bytes"])
            have = (self.source.name(i), self.source.nbytes(i)) if i < len(self.source) else None
            if have != want:
                raise SystemExit(f"resume refused: the checkpoint's data position is in {want[0]} ({want[1]:,} bytes) "
                                 f"but train shard #{i} here is {have}. Download the same shard set (same N).")
        self.shard_idx, self.pos, self.wraps = i, s["pos"], s.get("wraps", 0)
        self._load()

    def next_batch(self, batch_tokens):
        """(tokens [batch+1] of the stream, absolute offset of tokens[0] in the shard)."""
        if self.pos + batch_tokens + 1 >= len(self.tokens):
            nxt = self.shard_idx + 1
            if nxt >= len(self.source):
                if not self.allow_wrap:
                    raise RuntimeError(f"out of train data after {self.source.name(self.shard_idx)} (shard "
                                       f"{self.shard_idx + 1} of {len(self.source)}); pass --allow-wrap to repeat")
                nxt = 0
                self.wraps += 1
                if self.on_wrap:
                    self.on_wrap(self.wraps)
            self.shard_idx = nxt
            self.pos = 0
            self._load()
        start = self.pos
        self.pos += batch_tokens
        return self.tokens, start


def piece_plan(toks, start, rank, world, batch_tokens, seq_len, micro, block, bos, train_ctx=0, doc_ctx=0):
    """How one rank's share of a step is cut into micro-batch pieces: [(s, c)], where piece s scores inputs
    toks[s:s+micro] against targets toks[s+1:s+micro+1] and forwards toks[s-c:s+micro] (c unscored context).

    Upstream gives rank r the tokens [start + r*B/W, +B/W] and forwards each seq_len slice as one sequence. A piece
    never reaches back past its own sequence's start (upstream's attention cannot either), and c is a multiple of
    `block` so the forward keeps upstream's 128-token block grid. train_ctx: fixed c. doc_ctx: c reaches back to
    the block-aligned start of the document containing toks[s], if that is within doc_ctx tokens; then the doc
    mask hides everything before that document and the piece is exact at every layer; otherwise c = doc_ctx."""
    per_rank = batch_tokens // world
    r0 = start + rank * per_rank
    out = []
    for k in range(per_rank // micro):
        s = r0 + k * micro
        seq_start = start + ((s - start) // seq_len) * seq_len
        avail = s - seq_start
        if doc_ctx:
            lo = max(seq_start, s - doc_ctx)
            hits = np.flatnonzero(np.asarray(toks[lo:s + 1]) == bos)
            if len(hits):
                d = lo + int(hits[-1]) - seq_start  # doc start, relative to the sequence
                c = avail - (d - d % block)
            else:
                c = avail - (lo - seq_start)  # doc began before the cap (or before the sequence): truncated
        else:
            c = min(train_ctx, avail)
        out.append((s, c))
    return out


# =============================================================================================
# Optional validation extra: record #91's canonical-pair mask (needs tiktoken)
# =============================================================================================
_CLS = "SNLO?"
_SEAM_OK = np.ones((len(_CLS), len(_CLS)), dtype=bool)
_SEAM_OK[:, _CLS.index("?")] = False
_SEAM_OK[_CLS.index("S"), _CLS.index("S")] = False
_SEAM_OK[_CLS.index("O"), _CLS.index("L")] = False
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _char_cls(ch):
    import unicodedata
    if ch.isspace() and not "\x1c" <= ch <= "\x1f":
        return "S"
    cat = unicodedata.category(ch)[0]
    return cat if cat in "LN" else "O"


def _edge_cls(b, first):
    for n in range(1, 5):
        try:
            s = (b[:n] if first else b[-n:]).decode()
        except UnicodeDecodeError:
            continue
        return _CLS.index(_char_cls(s[0] if first else s[-1]))
    return _CLS.index("?")


def _ends_contraction(text):
    for c in _CONTRACTIONS:
        if text.endswith(c):
            before = text[:-len(c)]
            return not before or _char_cls(before[-1]) in "LN"
    return False


def build_canonical_mask(vocab_size):
    """Bit-packed [V, V/8] mask of (prev, cur) pairs GPT-2's tokenizer never emits (upstream canonical_mask.py)."""
    import tiktoken
    ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks
    tok = {v: k for k, v in ranks.items()}
    never = 1 << 30
    firsts, lasts, rules = {}, {}, {}
    for tid, b in tok.items():
        pieces = [bytes([c]) for c in b]
        first_traj, last_traj = [(0, pieces[0])], [(0, pieces[-1])]
        while len(pieces) > 1:
            best = best_i = None
            for i in range(len(pieces) - 1):
                r = ranks.get(pieces[i] + pieces[i + 1])
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            rules[tid] = (pieces[best_i], pieces[best_i + 1])
            pieces[best_i:best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
            if best_i == 0:
                first_traj.append((best + 1, pieces[0]))
            if best_i == len(pieces) - 1:
                last_traj.append((best, pieces[-1]))
        firsts[tid], lasts[tid] = first_traj, last_traj

    def by_piece(trajs):
        idx = {}
        for tid, traj in trajs.items():
            for i, (start, piece) in enumerate(traj):
                end = traj[i + 1][0] if i + 1 < len(traj) else never
                if start < end:
                    idx.setdefault(piece, []).append((start, end, tid))
        return idx

    by_last, by_first = by_piece(lasts), by_piece(firsts)
    end_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    start_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    closed = np.zeros(vocab_size, dtype=bool)
    for tid, b in tok.items():
        end_cls[tid], start_cls[tid] = _edge_cls(b, first=False), _edge_cls(b, first=True)
        try:
            closed[tid] = _ends_contraction(b.decode())
        except UnicodeDecodeError:
            pass
    mask = np.zeros((vocab_size, vocab_size), dtype=bool)
    for rank, (a, b) in rules.items():
        ps = np.array([p for s, e, p in by_last.get(a, ()) if s <= rank < e], dtype=np.intp)
        ps = ps[~closed[ps]]
        xs = np.array([x for s, e, x in by_first.get(b, ()) if s <= rank < e], dtype=np.intp)
        if len(ps) and len(xs):
            mask[np.ix_(ps, xs)] |= _SEAM_OK[np.ix_(end_cls[ps], start_cls[xs])]
    return torch.from_numpy(np.packbits(mask, axis=1, bitorder="little"))


# =============================================================================================
# Checkpoints: atomic (tmp + fsync + rename); the JSON sidecar, written last, is the commit marker
# =============================================================================================
def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write(path, write_fn):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path) or ".")


def list_checkpoints(ckpt_dir):
    """(next_step, pt, json) of committed checkpoints, newest first. A .pt without a matching sidecar is not
    committed (crash mid-write, or truncated later)."""
    out = []
    for js in glob.glob(os.path.join(ckpt_dir, "ckpt_*.json")):
        pt = js[:-5] + ".pt"
        try:
            with open(js) as f:
                meta = json.load(f)
            if os.path.getsize(pt) != meta["bytes"]:
                continue
            out.append((meta["next_step"], pt, js))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, reverse=True)


def load_latest(ckpt_dir, device):
    for _, pt, js in list_checkpoints(ckpt_dir):
        try:
            ck = torch.load(pt, map_location=device, weights_only=False)
            with open(js) as f:
                return ck, json.load(f), pt
        except Exception as e:  # unreadable despite the size check: fall back to the previous one
            print(f"[resume] skipping unreadable {pt}: {e!r}", flush=True)
    return None, None, None


def save_checkpoint(ckpt_dir, state, meta, keep):
    os.makedirs(ckpt_dir, exist_ok=True)
    base = os.path.join(ckpt_dir, f"ckpt_{state['next_step']:07d}")
    pt, js = base + ".pt", base + ".json"
    atomic_write(pt, lambda f: torch.save(state, f))
    meta = dict(meta, bytes=os.path.getsize(pt), file=os.path.basename(pt))
    atomic_write(js, lambda f: f.write(json.dumps(meta, indent=1).encode()))
    for _, old_pt, old_js in list_checkpoints(ckpt_dir)[keep:]:
        for f in (old_js, old_pt):  # sidecar first: a half-deleted checkpoint is never "committed"
            try:
                os.remove(f)
            except OSError:
                pass
    for tmp in glob.glob(os.path.join(ckpt_dir, "*.tmp.*")):  # leftovers of killed writers
        if not tmp.endswith(f".tmp.{os.getpid()}"):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return pt


def rng_state():
    s = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    torch.set_rng_state(s["torch"].cpu())
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


# =============================================================================================
# Training
# =============================================================================================
class Runner:
    def __init__(self, a):
        self.a = a
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if a.device == "cuda":
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")
        if self.world > 1:
            dist.init_process_group("nccl" if a.device == "cuda" else "gloo")
        self.master = self.rank == 0
        torch.set_num_threads(a.threads)
        if a.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)

        self.model = GPT(a.vocab_size, a.num_layers, a.model_dim).to(self.device)
        for b in self.model.blocks:
            b.mlp.prescale = a.mlp_prescale
        if self.world > 1:
            for t in self.model.parameters():
                dist.broadcast(t.data, 0)
        self.bos = 50256 if a.vocab_size == 50257 else a.vocab_size - 1
        self.sched = Schedule(a)
        self.opt = Optimizers(self.model)
        self.opt.round_embeds_bf16 = a.round_embeds_bf16
        self.bag = GradBag(list(self.model.parameters()))
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=a.scaler == "on", init_scale=a.init_scale)
        self.amp_dtype = torch.float16 if a.amp == "fp16" else torch.float32
        self.fwd = torch.compile(self.model.hidden, dynamic=True) if a.compile else self.model.hidden
        if a.synthetic:
            self.train_src = SyntheticSource(a.vocab_size, self.bos, a.seed, a.synthetic_shard_tokens, stream=2)
            self.val_src = SyntheticSource(a.vocab_size, self.bos, a.seed, max(a.synthetic_shard_tokens, a.val_tokens + 1), stream=3)
        else:
            self.train_src, self.val_src = FileSource(a.train_files), FileSource(a.val_files)
        self.loader = StreamLoader(self.train_src, allow_wrap=a.allow_wrap, on_wrap=self._on_wrap)
        assert a.batch_tokens % (self.world * a.seq_len) == 0 or a.batch_tokens % self.world == 0
        assert a.seq_len % a.micro_tokens == 0 and a.micro_tokens % a.block == 0
        # run bookkeeping (all checkpointed)
        self.next_step = 0
        self.tokens_seen = 0
        self.ctx_tokens_seen = 0  # unscored context tokens re-read by --train-ctx / --doc-ctx (extra compute)
        self.overflow_skips = 0
        self.consec_skips = 0  # current streak of skipped steps; --max-consecutive-skips aborts the run
        self.streak_fwd_bad = 0  # ... of which the forward loss itself was non-finite
        self.scale_alarmed = False
        self.gpu_seconds = 0.0  # compute only: train-step time x world size
        self.wall_seconds = 0.0  # compute only: train-step time
        self.val_seconds = 0.0
        self.ckpt_seconds = 0.0
        self.wall_prev = 0.0  # whole-session wall-clock of earlier sessions, up to their last checkpoint
        self.segments = []
        self.done = False

    def wall_total(self):
        """Wall-clock of all sessions: startup, data checks, train, val and checkpoint time. A killed session's
        time after its last checkpoint is not in it (see its log and the lane's summary.json elapsed_h)."""
        return self.wall_prev + time.time() - _T_PROCESS

    def timing(self):
        w = self.wall_total()
        return dict(wall_total_s=round(w, 2), gpu_seconds_total=round(w * self.world, 2),
                    compute_s=round(self.wall_seconds, 2), gpu_seconds=round(self.gpu_seconds, 2),
                    val_s=round(self.val_seconds, 2), ckpt_s=round(self.ckpt_seconds, 2))

    def val_set_info(self):
        """What the val loss is measured on: file, size, sha256s, token count. Logged at start and with every val."""
        a, src = self.a, self.val_src
        used = np.ascontiguousarray(np.asarray(src.shard(0)[:a.val_tokens + 1]), dtype="<u2")
        info = dict(val_file=src.name(0), val_file_bytes=src.nbytes(0), val_tokens=a.val_tokens,
                    val_tokens_sha256=hashlib.sha256(used.tobytes()).hexdigest(), val_seq_len=a.seq_len,
                    val_sequences=a.val_tokens // a.seq_len, val_chunk=a.val_chunk, synthetic=a.synthetic)
        info["val_file_sha256"] = src.sha256(0) if isinstance(src, FileSource) else None
        info["val_set_is_upstream"] = (not a.synthetic and a.val_tokens == VAL_TOKENS_UPSTREAM
                                       and a.seq_len == 64 * 1024 and info["val_tokens_sha256"] == VAL_SLICE_SHA256
                                       and info["val_file_sha256"] == VAL_FILE_SHA256)
        return info

    def _on_wrap(self, n):
        self.log(event="data_wrap", wraps=n, tokens=self.tokens_seen,
                 warning="train shards exhausted: wrapped to the first shard; later tokens repeat data")
        if self.master:
            print(f"[warning] train data wrapped (#{n}): tokens from here on repeat earlier data", flush=True)

    def log(self, **rec):
        if not self.master:
            return
        rec.setdefault("provider", self.a.provider)
        rec.setdefault("time", round(time.time(), 3))
        os.makedirs(os.path.dirname(self.a.log) or ".", exist_ok=True)
        with open(self.a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp_dtype == torch.float16)

    def rt(self, window_blocks):
        return dict(dtype=self.amp_dtype, window_blocks=window_blocks, block=self.a.block, attn_chunk=self.a.attn_chunk)

    # ---- checkpoint ----
    def state(self):
        return dict(
            version=2, upstream=UPSTREAM_COMMIT, record=UPSTREAM_RECORD, args={k: getattr(self.a, k) for k in ARCH_KEYS},
            next_step=self.next_step, done=self.done, model=self.model.state_dict(), opt=self.opt.state_dict(),
            scaler=self.scaler.state_dict(), loader=self.loader.state_dict(), rng=rng_state(),
            wall_total_s=self.wall_total(), val_seconds=self.val_seconds, ckpt_seconds=self.ckpt_seconds,
            tokens_seen=self.tokens_seen, ctx_tokens_seen=self.ctx_tokens_seen, overflow_skips=self.overflow_skips, consec_skips=self.consec_skips,
            streak_fwd_bad=self.streak_fwd_bad, gpu_seconds=self.gpu_seconds,
            wall_seconds=self.wall_seconds, segments=self.segments)

    def check_ranks_agree(self):
        """Each rank runs Adam and Muon on its own copy; only rank 0's is saved. Abort if the copies drifted."""
        sums = torch.stack([p.detach().double().abs().sum() for p in self.model.parameters()])
        hi, lo = sums.clone(), sums.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        diff = ((hi - lo) / hi.clamp_min(1e-30)).max().item()
        if diff > self.a.rank_check_tol:
            self.log(event="abort", reason="rank_weights_differ", step=self.next_step - 1, max_rel_diff=diff)
            if self.master:
                print(f"ABORT: rank weights differ (max relative |sum| difference {diff:.3g}); not checkpointing",
                      file=sys.stderr, flush=True)
            dist.destroy_process_group()
            raise SystemExit(EXIT_RANK_DRIFT)
        return dict(rank_weights_identical=bool((hi == lo).all()), rank_weights_max_rel_diff=diff)

    def save(self, reason):
        t0 = time.time()
        agree = self.check_ranks_agree() if self.world > 1 else {}
        self.segments[-1].update(end_step=self.next_step - 1, ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 session_wall_s=round(time.time() - _T_PROCESS, 1))
        if self.master:
            meta = dict(next_step=self.next_step, step=self.next_step - 1, tokens_seen=self.tokens_seen,
                        wall_clock_s=round(self.wall_seconds, 3), gpu_seconds=round(self.gpu_seconds, 3),
                        **{k: v for k, v in self.timing().items() if k not in ("gpu_seconds",)},
                        provider=self.a.provider, segments=self.segments, reason=reason, done=self.done,
                        world_size=self.world, data_position=self.loader.state_dict(),
                        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), upstream=UPSTREAM_COMMIT, record=UPSTREAM_RECORD)
            pt = save_checkpoint(self.a.ckpt_dir, self.state(), meta, self.a.keep_ckpts)
            self.log(event="checkpoint", step=self.next_step - 1, next_step=self.next_step, path=pt, reason=reason,
                     seconds=round(time.time() - t0, 3), bytes=os.path.getsize(pt), **agree)
        if self.world > 1:
            dist.barrier()
        self.ckpt_seconds += time.time() - t0

    def try_resume(self):
        ck, meta, pt = load_latest(self.a.ckpt_dir, "cpu")  # not the GPU: ~3 GB transient + allocator residue
        if ck is None:
            return False
        mine = {k: getattr(self.a, k) for k in ARCH_KEYS}
        if ck["args"] != mine:
            raise SystemExit(f"checkpoint {pt} was made with different settings: {ck['args']} vs {mine}")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.scaler.load_state_dict(ck["scaler"])
        self.loader.load_state_dict(ck["loader"])
        set_rng_state(ck["rng"])
        self.next_step, self.done = ck["next_step"], ck["done"]
        self.tokens_seen, self.overflow_skips = ck["tokens_seen"], ck["overflow_skips"]
        self.consec_skips, self.streak_fwd_bad = ck.get("consec_skips", 0), ck.get("streak_fwd_bad", 0)
        self.ctx_tokens_seen = ck.get("ctx_tokens_seen", 0)
        self.gpu_seconds, self.wall_seconds, self.segments = ck["gpu_seconds"], ck["wall_seconds"], ck["segments"]
        self.wall_prev = ck.get("wall_total_s", self.wall_seconds)
        self.val_seconds, self.ckpt_seconds = ck.get("val_seconds", 0.0), ck.get("ckpt_seconds", 0.0)
        self.log(event="resume", from_path=pt, next_step=self.next_step, tokens_seen=self.tokens_seen,
                 gpu_seconds=round(self.gpu_seconds, 3))
        return True

    # ---- validation: first val_tokens of the val stream, as independent seq_len sequences (upstream) ----
    @torch.no_grad()
    def validate(self, step):
        a, m = self.a, self.model
        m.eval()
        if a.canon_mask == "on" and m.canon_mask is None:
            t0 = time.time()
            m.canon_mask = build_canonical_mask(m.vocab_size).to(self.device)
            self.log(event="canon_mask_built", seconds=round(time.time() - t0, 2))
        toks = self.val_src.shard(0)
        assert a.val_tokens % a.seq_len == 0 and a.val_tokens + 1 <= len(toks)
        wb = self.sched.window_blocks(step)
        ctx = next_multiple(len(m.blocks) * wb * a.block, a.block)  # >= the receptive field: chunking is exact
        sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        for sidx in range(self.rank, a.val_tokens // a.seq_len, self.world):
            lo = sidx * a.seq_len
            buf = torch.from_numpy(np.asarray(toks[lo:lo + a.seq_len + 1]).astype(np.int64)).to(self.device)
            x, y = buf[:-1], buf[1:]
            seg = torch.cumsum((x == self.bos).long(), 0)
            for c0 in range(0, a.seq_len, a.val_chunk):
                s0 = max(0, c0 - ctx)
                s0 -= s0 % a.block  # keep window blocks aligned to the sequence start
                c1 = min(a.seq_len, c0 + a.val_chunk)
                with self.autocast():
                    h = self.fwd(x[s0:c1], seg[s0:c1], self.rt(wb))[c0 - s0:]
                    ce, ce_m = ce_per_token(m, h, y[c0:c1], a.loss_chunk, inputs=x[c0:c1])
                sums[0] += ce.double().sum()
                sums[1] += (ce_m if ce_m is not None else ce).double().sum()
                sums[2] += c1 - c0
        if self.world > 1:
            dist.all_reduce(sums)
        m.train()
        return (sums[0] / sums[2]).item(), (sums[1] / sums[2]).item()

    # ---- one optimizer step ----
    def train_step(self, step):
        a, m = self.a, self.model
        B, W, M = a.batch_tokens, self.world, a.micro_tokens
        assert B % W == 0 and (B // W) % M == 0
        per_rank = B // W
        toks, start = self.loader.next_batch(B)
        wb = self.sched.window_blocks(step)
        rt = self.rt(wb)
        loss_sum = torch.zeros(2, dtype=torch.float64, device=self.device)  # [CE sum, re-read context tokens]
        amax, hooks = None, []
        if a.amax_every > 0 and not a.compile and step % a.amax_every == 0:
            amax = torch.zeros(len(m.blocks), device=self.device)

            def track(i):
                def hook(mod, inp, out):
                    amax[i] = torch.maximum(amax[i], out.detach().abs().max().float())
                return hook
            hooks = [b.mlp.c_fc.register_forward_hook(track(i)) for i, b in enumerate(m.blocks)]
        # upstream forwards each rank's seq_len slice as one sequence; we forward M-token pieces of it,
        # optionally with unscored context of the same sequence in front (--train-ctx or --doc-ctx)
        for s, c in piece_plan(toks, start, self.rank, W, B, a.seq_len, M, a.block, self.bos, a.train_ctx, a.doc_ctx):
            loss_sum[1] += c
            buf = torch.from_numpy(np.asarray(toks[s - c:s + M + 1]).astype(np.int64)).to(self.device)
            x, y = buf[:-1], buf[c + 1:]
            seg = torch.cumsum((x == self.bos).long(), 0)
            with self.autocast():
                h = self.fwd(x, seg, rt)[c:]
                ce, _ = ce_per_token(m, h, y, a.loss_chunk)
                if 0 <= a.inject_overflow_from <= step:  # tests: simulate a forward overflow
                    ce = ce * float("inf")
                L = ce.sum() / M  # per-token scale keeps fp16 grads in range; exact rescale to a mean below
            self.scaler.scale(L).backward()
            loss_sum[0] += ce.detach().double().sum()
        for hk in hooks:
            hk.remove()
        params = [p for p in m.parameters() if p.grad is not None]
        if W > 1:  # one flat all-reduce of the (still scaled) grads: every rank then sees the same inf check
            flat = torch.cat([p.grad.reshape(-1) for p in params])
            dist.all_reduce(flat)
            off = 0
            for p in params:
                p.grad.copy_(flat[off:off + p.numel()].view_as(p.grad))
                off += p.numel()
            dist.all_reduce(loss_sum)
            if amax is not None:
                dist.all_reduce(amax, op=dist.ReduceOp.MAX)
        self.scaler.unscale_(self.bag)
        ok = self.scaler.step(self.bag) is not None
        if not self.scaler.is_enabled():  # no GradScaler (fp32): still never apply a non-finite gradient
            ok = bool(torch.stack([p.grad.isfinite().all() for p in params]).all())
        scale = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
        self.scaler.update()
        if ok:
            for p in params:
                p.grad.mul_(M / B)  # == upstream: mean over the step's tokens
            self.opt.step(self.sched.lr(step), self.sched.muon_momentum(step))
        else:
            self.overflow_skips += 1
        m.zero_grad(set_to_none=True)
        loss = (loss_sum[0] / B).item()
        return dict(loss=loss, batch=B, n_micro=per_rank // M, scale=scale, skipped=not ok, window=wb * a.block,
                    ctx_tokens=int(loss_sum[1].item()),  # re-read (unscored) context, all ranks; not in tokens_seen
                    nonfinite_fwd=not math.isfinite(loss),  # forward overflow, vs a backward-only overflow
                    c_fc_amax=None if amax is None else [round(v, 2) for v in amax.tolist()])

    # ---- main loop ----
    def run(self):
        a = self.a
        stop = {"sig": False}

        def on_signal(signum, frame):
            stop["sig"] = True
            print(f"[signal] {signal.Signals(signum).name}: checkpoint after this step", flush=True)

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        resumed = self.try_resume() if a.resume else False
        if self.done:
            self.log(event="already_done", next_step=self.next_step)
            return
        total = a.num_iters
        need = total if a.stop_after_step < 0 else min(total, a.stop_after_step + 1)
        capacity = steps_capacity(self.train_src, a.batch_tokens)
        if need > capacity:
            msg = (f"{need} steps x {a.batch_tokens:,} tokens need more data than the {len(self.train_src)} train "
                   f"shard(s) hold ({capacity} steps)")
            if not a.allow_wrap:
                raise SystemExit(f"refusing to start: {msg}. Download more shards (scripts/get_data.py N) or pass "
                                 "--allow-wrap to repeat data.")
            self.log(event="data_wrap_warning", warning=msg + "; --allow-wrap: data will repeat")
            if self.master:
                print(f"[warning] {msg}; --allow-wrap: data will repeat", flush=True)
        dev_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        self.segments.append(dict(provider=a.provider, start_step=self.next_step, world_size=self.world,
                                  device=dev_name, started=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        self.val_info = self.val_set_info() if self.master else {}
        if self.master:
            v = self.val_info
            print(f"val set: {v['val_file']} {v['val_file_bytes']:,} bytes sha256 {v['val_file_sha256']} | "
                  f"{v['val_tokens']:,} tokens sha256 {v['val_tokens_sha256'][:16]}... | upstream's val set: "
                  f"{'YES' if v['val_set_is_upstream'] else 'NO'}", flush=True)
            if not v["val_set_is_upstream"] and not (a.tiny or a.synthetic):
                print(f"[WARNING] val set is NOT upstream's ({VAL_TOKENS_UPSTREAM:,} tokens of fineweb_val_000000.bin, "
                      f"sha256 {VAL_SLICE_SHA256[:16]}...): val losses are not comparable to 3.28", flush=True)
        self.log(event="start", resumed=resumed, next_step=self.next_step, total_steps=total, world_size=self.world,
                 device=dev_name, amp=a.amp, scaler=a.scaler, upstream=UPSTREAM_COMMIT, record=UPSTREAM_RECORD,
                 params=sum(p.numel() for p in self.model.parameters()), batch_tokens=a.batch_tokens,
                 micro_tokens=a.micro_tokens, train_ctx=a.train_ctx, doc_ctx=a.doc_ctx, torch=torch.__version__,
                 args=vars(a), train_files=[[self.train_src.name(i), self.train_src.nbytes(i)]
                                            for i in range(len(self.train_src))],
                 steps_capacity=capacity, **self.val_info, **self.timing())
        session_t0 = last_ckpt = time.time()
        step = self.next_step
        while True:
            last = step == total
            if last or (a.val_every > 0 and step % a.val_every == 0):
                t0 = time.time()
                val, val_canon = self.validate(step)
                self.val_seconds += time.time() - t0
                rec = dict(event="val", step=step, val_loss=val, tokens=self.tokens_seen,
                           seconds=round(time.time() - t0, 2), **{k: self.val_info.get(k) for k in (
                               "val_file", "val_tokens", "val_tokens_sha256", "val_seq_len", "val_chunk",
                               "val_set_is_upstream")}, **self.timing())
                if a.canon_mask == "on":
                    rec["val_loss_canon"] = val_canon
                if step in UPSTREAM_VAL and self.val_info.get("val_set_is_upstream") and not a.tiny:
                    rec.update(upstream_val_loss=UPSTREAM_VAL[step], val_minus_upstream=round(val - UPSTREAM_VAL[step], 4),
                               schedule_same_as_upstream=self.sched.same_as_upstream(step))
                self.log(**rec)
                if self.master:
                    print(f"step {step}/{total} val_loss {val:.4f}", flush=True)
            if last:
                self.next_step, self.done = total + 1, True
                self.save("final")
                self.log(event="exit", next_step=self.next_step, reason="done", **self.timing())
                break
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = self.train_step(step)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            self.tokens_seen += out["batch"]
            self.ctx_tokens_seen += out["ctx_tokens"]
            self.gpu_seconds += dt * self.world
            self.wall_seconds += dt
            self.next_step = step + 1
            if out["skipped"]:
                self.consec_skips += 1
                self.streak_fwd_bad += out["nonfinite_fwd"]
            else:
                self.consec_skips = self.streak_fwd_bad = 0
            scale_now = float(self.scaler.get_scale()) if self.scaler.is_enabled() else None
            rec = dict(step=step, train_loss=out["loss"], tokens=self.tokens_seen, lr_mult=self.sched.lr(step),
                       muon_momentum=self.sched.muon_momentum(step), window=out["window"], scale=out["scale"],
                       overflow_skips=self.overflow_skips, skipped=out["skipped"], consec_skips=self.consec_skips,
                       nonfinite_fwd=out["nonfinite_fwd"], tokens_per_s=round(out["batch"] / dt, 1),
                       step_s=round(dt, 4), micro=out["n_micro"], ctx_tokens=out["ctx_tokens"],
                       ctx_tokens_seen=self.ctx_tokens_seen, **self.timing())
            if out["c_fc_amax"] is not None:
                rec.update(c_fc_amax=max(out["c_fc_amax"]), c_fc_amax_layers=out["c_fc_amax"])
            self.log(**rec)
            if scale_now is not None and scale_now < a.scale_floor:
                if not self.scale_alarmed:
                    self.log(event="scale_alarm", step=step, scale=scale_now, floor=a.scale_floor,
                             consec_skips=self.consec_skips)
                    if self.master:
                        print(f"[alarm] step {step}: GradScaler scale {scale_now:g} < floor {a.scale_floor:g}",
                              flush=True)
                    self.scale_alarmed = True
            else:
                self.scale_alarmed = False
            if a.max_consecutive_skips > 0 and self.consec_skips >= a.max_consecutive_skips:
                n = self.consec_skips
                msg = (f"ABORT: {n} consecutive overflow-skipped steps ({step - n + 1}..{step}; forward loss "
                       f"non-finite in {self.streak_fwd_bad} of them, so a smaller loss scale cannot help those); "
                       f"scale {scale_now}. Not checkpointing: the newest checkpoint is the last good one.")
                self.log(event="abort", reason="overflow_skips", step=step, consec_skips=n,
                         nonfinite_fwd_in_streak=self.streak_fwd_bad, scale=scale_now)
                self.log(event="exit", next_step=self.next_step, reason="overflow_abort", **self.timing())
                if self.master:
                    print(msg, file=sys.stderr, flush=True)
                if self.world > 1:
                    dist.destroy_process_group()
                raise SystemExit(EXIT_OVERFLOW)
            if self.master and (step % 25 == 0 or step == total - 1):
                print(f"step {step}/{total} loss {out['loss']:.4f} {out['batch'] / dt:.0f} tok/s", flush=True)
            flags = torch.tensor([
                float(stop["sig"]),
                float(time.time() - last_ckpt >= a.ckpt_every_min * 60
                      or (a.ckpt_every_steps > 0 and (step + 1) % a.ckpt_every_steps == 0)),
                float(a.time_budget_min > 0 and time.time() - session_t0 >= a.time_budget_min * 60),
                float(step == a.stop_after_step)], device=self.device)
            if self.world > 1:  # every rank takes the same decision
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            sig, due, budget, stop_here = (bool(f) for f in flags.tolist())
            if sig or budget or stop_here:
                reason = "sigterm" if sig else "time_budget" if budget else "stop_after_step"
                self.save(reason)
                self.log(event="exit", next_step=self.next_step, reason=reason, **self.timing())
                break
            if due and self.consec_skips == 0:  # never make a mid-streak checkpoint the newest one
                self.save("periodic")
                last_ckpt = time.time()
            step += 1
        if self.world > 1:
            dist.destroy_process_group()


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB -> MiB (process peak RSS)


def dry_shape(a):
    """One full-config training step on random tokens: parameter count, step time and peak memory."""
    device = torch.device("cuda" if a.device == "cuda" else "cpu")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    # on cuda: one piece at the largest forward the run will do (micro-batch + its unscored context)
    S = a.dry_shape_tokens or (a.micro_tokens + max(a.train_ctx, a.doc_ctx) if device.type == "cuda" else 1024)
    model = GPT(a.vocab_size, a.num_layers, a.model_dim).to(device)
    opt = Optimizers(model)
    n_all = sum(p.numel() for p in model.parameters())
    n_embed = sum(e.weight.numel() for e in [model.embed, *model.value_embeds])
    x = torch.randint(0, a.vocab_size, (S + 1,), device=device)
    x[::997] = 50256 if a.vocab_size == 50257 else a.vocab_size - 1
    seg = torch.cumsum((x[:-1] == x[0]).long(), 0)
    wb = Schedule(a).window_blocks(a.num_iters)  # widest window (end of training)
    rt = dict(dtype=torch.float16 if a.amp == "fp16" else torch.float32, window_blocks=wb, block=a.block,
              attn_chunk=a.attn_chunk)
    t0 = time.time()
    with torch.autocast(device.type, dtype=torch.float16, enabled=a.amp == "fp16"):
        ce, _ = ce_per_token(model, model.hidden(x[:-1], seg, rt), x[1:], a.loss_chunk)
    ce.mean().backward()
    t1 = time.time()
    opt.step(1.0, 0.95)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    rep = dict(params=n_all, params_non_embedding=n_all - n_embed, tokens=S, window=wb * a.block, amp=a.amp,
               loss=round(ce.mean().item(), 4), fwd_bwd_s=round(t1 - t0, 3), opt_step_s=round(t2 - t1, 3),
               peak_memory_mb=round(peak_memory_mb(device), 1), device=str(device), torch=torch.__version__)
    print(json.dumps(rep))
    return rep


def main(argv=None):
    a = parse_args(argv)
    if a.dry_shape:
        return dry_shape(a)
    Runner(a).run()


if __name__ == "__main__":
    main()
