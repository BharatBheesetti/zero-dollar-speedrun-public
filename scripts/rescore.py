#!/usr/bin/env python3
"""Re-score a saved checkpoint on the val set, in a fresh process, with the trainer's own model and val code.

Closes review M3 finding 4: the 3.2698 was logged by the training loop; this loads the saved `.pt` and scores it
again, outside training. It imports trainer/train_t4.py and calls its `Runner.validate` unchanged (same 160 x
65,536-token sequences, same window, same softcapped CE), so any difference comes from the weights or the
precision, not from a second implementation of the loss.

  # Kaggle 2xT4 (lanes/kaggle_push.py rescore does this): fp16 autocast as in training, then fp32
  torchrun --standalone --nproc_per_node=2 scripts/rescore.py --ckpt ckpt_0001451.pt --data-dir DATA \\
      --amp fp16 --amp fp32 --expect 3.2698315009167063 --out rescore.json
  # CPU, tiny checkpoint from `make smoke`:
  python scripts/rescore.py --ckpt out/smoke/ckpt/ckpt_0000061.pt --tiny --synthetic --device cpu --amp fp32

Unknown flags go to the trainer's parser (e.g. --val-chunk 16384, still exact). The architecture flags
(vocab, width, layers, num_iters, seq_len, block, seed, synthetic) always come from the checkpoint.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "trainer"))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import train_t4 as T  # noqa: E402


class Evaluator:
    """Just the attributes Runner.validate reads; its methods are the trainer's own functions."""
    validate = T.Runner.validate
    autocast = T.Runner.autocast
    rt = T.Runner.rt
    val_set_info = T.Runner.val_set_info

    def __init__(self, a, model, device, rank, world):
        self.a, self.model, self.device, self.rank, self.world = a, model, device, rank, world
        self.master = rank == 0
        self.sched = T.Schedule(a)
        self.bos = 50256 if a.vocab_size == 50257 else a.vocab_size - 1
        self.fwd = model.hidden
        self.amp_dtype = torch.float32
        if a.synthetic:  # same construction as Runner.__init__
            self.val_src = T.SyntheticSource(a.vocab_size, self.bos, a.seed,
                                             max(a.synthetic_shard_tokens, a.val_tokens + 1), stream=3)
        else:
            self.val_src = T.FileSource(a.val_files)

    def log(self, **rec):  # validate() only logs when building the canonical mask
        pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a trainer checkpoint (.pt with its .json sidecar next to it)")
    p.add_argument("--amp", action="append", choices=["fp16", "fp32"],
                   help="precision per pass, repeatable (default: fp16 on cuda, fp32 on cpu)")
    p.add_argument("--expect", type=float, default=None, help="the val loss the training run logged")
    p.add_argument("--out", default=None, help="write the result JSON here (rank 0)")
    p.add_argument("--hash", action="store_true", help="sha256 the checkpoint (~15 s for 3 GB)")
    p.add_argument("--save-model", default=None, help="also save model-only fp32 weights (state_dict) here")
    ra, rest = p.parse_known_args(argv)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    a = T.parse_args(rest)
    if a.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(a.threads)

    t0 = time.time()
    ck = torch.load(ra.ckpt, map_location="cpu", weights_only=False, mmap=True)
    load_s = time.time() - t0
    for k, v in ck["args"].items():  # architecture + schedule from the checkpoint, never from flags
        setattr(a, k, v)
    model = T.GPT(a.vocab_size, a.num_layers, a.model_dim).to(device)
    model.load_state_dict(ck["model"])
    for b in model.blocks:
        b.mlp.prescale = a.mlp_prescale
    # run() validates step `num_iters` and then saves next_step = num_iters + 1; a periodic checkpoint with
    # next_step N holds the state before step N trains, which run() would validate as step N.
    step = min(ck["next_step"], a.num_iters)
    ev = Evaluator(a, model, device, rank, world)
    sidecar = ra.ckpt[:-3] + ".json" if ra.ckpt.endswith(".pt") else None
    meta = json.load(open(sidecar)) if sidecar and os.path.exists(sidecar) else {}

    passes = []
    for amp in ra.amp or (["fp16"] if device.type == "cuda" else ["fp32"]):
        ev.amp_dtype = torch.float16 if amp == "fp16" else torch.float32
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        val, _ = ev.validate(step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rec = dict(amp=amp, val_loss=val, seconds=round(time.time() - t1, 2))
        if ra.expect is not None:
            rec["minus_logged"] = val - ra.expect
        passes.append(rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)

    if rank == 0:
        n_all = sum(t.numel() for t in model.parameters())
        trainer_py = os.path.join(ROOT, "trainer", "train_t4.py")
        out = dict(
            ckpt=os.path.basename(ra.ckpt), ckpt_bytes=os.path.getsize(ra.ckpt),
            ckpt_sha256=sha256(ra.ckpt) if ra.hash else None, ckpt_load_s=round(load_s, 2),
            ckpt_step=ck["next_step"] - 1, ckpt_done=ck["done"], ckpt_tokens_seen=ck["tokens_seen"],
            ckpt_overflow_skips=ck["overflow_skips"], ckpt_args=ck["args"],
            sidecar=dict((k, meta.get(k)) for k in ("step", "next_step", "tokens_seen", "reason", "bytes", "created")),
            val_step=step, window_tokens=ev.sched.window_blocks(step) * a.block, params=n_all,
            passes=passes, expect=ra.expect, **ev.val_set_info(),
            trainer_sha256=sha256(trainer_py), rescore_sha256=sha256(os.path.abspath(__file__)),
            world_size=world, device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
            torch=torch.__version__, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if ra.save_model:
            sd = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
            torch.save(dict(model=sd, args=ck["args"], step=out["ckpt_step"], val_loss=passes[0]["val_loss"],
                            upstream=T.UPSTREAM_COMMIT, record=T.UPSTREAM_RECORD), ra.save_model)
            out.update(model_file=os.path.basename(ra.save_model), model_bytes=os.path.getsize(ra.save_model),
                       model_sha256=sha256(ra.save_model))
        print(json.dumps(out, indent=1), flush=True)
        if ra.out:
            with open(ra.out, "w") as f:
                json.dump(out, f, indent=1)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return passes


if __name__ == "__main__":
    main()
