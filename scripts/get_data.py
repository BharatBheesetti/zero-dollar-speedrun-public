#!/usr/bin/env python3
"""Download GPT-2-tokenized FineWeb10B shards: the val shard + the first N train shards.

Same files as upstream's data/cached_fineweb10B.py (HF dataset kjj0/fineweb10B-gpt2, 100M tokens / ~200 MB
per shard). Record #18 trains on 728.8M tokens, so a full run needs N >= 8 (it streams shards 1..N in order).
Stdlib only: resumable .part downloads, header check, atomic rename. Disk is checked before each file.

  python scripts/get_data.py 1            # laptop: val + 1 train shard (~400 MB)
  python scripts/get_data.py 9 --dir /kaggle/working/data/fineweb10B
"""
import argparse
import os
import shutil
import sys
import urllib.request

import numpy as np

REPO = "kjj0/fineweb10B-gpt2"
URL = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
SHARD_BYTES = 256 * 4 + 2 * 100_000_000  # header + 100M uint16 tokens (the last train shard is shorter)


def check_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, f"{path}: bad magic/version"
    assert os.path.getsize(path) == 256 * 4 + 2 * int(header[2]), f"{path}: size does not match header"
    return int(header[2])


def fetch(name, out_dir, min_free_gb):
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst):
        print(f"have {name} ({check_shard(dst):,} tokens)")
        return
    free = shutil.disk_usage(out_dir).free
    if free - SHARD_BYTES < min_free_gb * 2 ** 30:
        sys.exit(f"refusing to download {name}: {free / 2**30:.1f} GB free, need {SHARD_BYTES / 2**30:.1f} GB "
                 f"+ {min_free_gb} GB headroom")
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(URL.format(repo=REPO, name=name), headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
        if have and r.status != 206:  # server ignored the range: start over
            f.truncate(0)
        total = have + int(r.headers.get("Content-Length", 0))
        done = have
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB", end="", flush=True)
        f.flush()
        os.fsync(f.fileno())
    print()
    check_shard(part)
    os.replace(part, dst)
    print(f"got {name} ({check_shard(dst):,} tokens)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("n", type=int, nargs="?", default=1, help="number of train shards (1..103)")
    p.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fineweb10B"))
    p.add_argument("--min-free-gb", type=float, default=2.0, help="keep at least this much disk free")
    a = p.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    fetch("fineweb_val_000000.bin", a.dir, a.min_free_gb)
    for i in range(1, a.n + 1):
        fetch(f"fineweb_train_{i:06d}.bin", a.dir, a.min_free_gb)


if __name__ == "__main__":
    main()
