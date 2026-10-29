"""Real FineWeb shards (skipped when data/ is absent; `make data` fetches val + 1 train shard): the files are
upstream's format, decode to English with GPT-2's tokenizer, and the val slice is the pinned upstream val set.
This replaces the old "step-0 loss = ln(50304)" check, which holds for any tokens (zero-init lm_head)."""
import glob
import os
import re

import numpy as np
import pytest

import train_t4 as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "fineweb10B")
VAL = os.path.join(DATA, "fineweb_val_000000.bin")
TRAIN = sorted(glob.glob(os.path.join(DATA, "fineweb_train_*.bin")))
pytestmark = pytest.mark.skipif(not (os.path.exists(VAL) and TRAIN), reason="no FineWeb shards in data/ (make data)")

UPSTREAM_MAGIC, UPSTREAM_VERSION = 20240520, 1  # record #18's _load_data_shard asserts exactly these
COMMON = set("the of and to a in is that for it was on with as be are this by at from or have an not but you "
             "we they his her their has all can will one more about which also were been".split())


def english(text):
    """(share of printable ASCII chars, share of words that are very common English words)"""
    ascii_ok = sum(c.isascii() and (c.isprintable() or c in "\n\t") for c in text) / max(1, len(text))
    words = re.findall(r"[a-z]+", text.lower())
    return ascii_ok, sum(w in COMMON for w in words) / max(1, len(words))


@pytest.mark.parametrize("path", [VAL] + TRAIN[:1])
def test_shard_header_is_upstream_format(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == UPSTREAM_MAGIC == T.SHARD_MAGIC and header[1] == UPSTREAM_VERSION == T.SHARD_VERSION
    n = int(header[2])
    assert os.path.getsize(path) == 256 * 4 + 2 * n
    assert len(T.read_shard(path)) == n and (os.path.basename(path) != "fineweb_val_000000.bin" or n == 100_000_000)


@pytest.mark.parametrize("path", [VAL] + TRAIN[:1])
def test_first_200_tokens_decode_to_english(path):
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    toks = np.asarray(T.read_shard(path)[:200]).astype(np.int64)
    assert toks.max() < 50257
    text = enc.decode(toks.tolist())
    ascii_ok, common = english(text)
    print(f"{os.path.basename(path)}: {text[:120]!r} ascii={ascii_ok:.2f} common={common:.2f}")
    assert ascii_ok > 0.95 and common > 0.2, text[:500]
    if path == VAL:
        assert toks[0] == 50256  # the val shard starts at a document boundary (<|endoftext|>)
    noise = enc.decode(np.random.default_rng(0).integers(0, 50257, 200).tolist())  # control: the check can fail
    assert english(noise)[1] < 0.1


def test_val_slice_is_pinned_upstream_val_set(tmp_path):
    r = T.Runner(T.parse_args(["--device", "cpu", "--tiny", "--val-files", VAL, "--train-files", TRAIN[0],
                               "--val-tokens", str(T.VAL_TOKENS_UPSTREAM), "--seq-len", str(64 * 1024),
                               "--out-dir", str(tmp_path)]))
    info = r.val_set_info()
    assert info["val_file"] == "fineweb_val_000000.bin" and info["val_file_bytes"] == 200_001_024
    assert info["val_tokens"] == 10_485_760 and info["val_sequences"] == 160
    assert info["val_file_sha256"] == T.VAL_FILE_SHA256 and info["val_tokens_sha256"] == T.VAL_SLICE_SHA256
    assert info["val_set_is_upstream"] is True
