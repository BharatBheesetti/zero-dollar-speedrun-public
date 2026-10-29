# Zero-Dollar Speedrun. PY defaults to the repo venv; override e.g. PY=/path/to/python make test
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
THREADS ?= 8

.PHONY: test smoke data venv dry-shape kaggle-dryrun

test:
	OMP_NUM_THREADS=$(THREADS) ZDS_THREADS=$(THREADS) $(PY) -m pytest -q -s -o tmp_path_retention_policy=failed tests

smoke:  # one tiny CPU run, prints losses
	OMP_NUM_THREADS=$(THREADS) $(PY) trainer/train_t4.py --tiny --synthetic --device cpu --threads $(THREADS) --out-dir out/smoke

# e.g. make dry-shape DEVICE=cuda ARGS="--micro-tokens 16384 --doc-ctx 2048"
dry-shape:  # full 124M-class config, one fwd/bwd + optimizer step on random tokens: params, peak memory
	OMP_NUM_THREADS=$(THREADS) $(PY) trainer/train_t4.py --dry-shape --device $(or $(DEVICE),cpu) --threads $(THREADS) $(ARGS)

data:  # val shard + N train shards of GPT-2-tokenized FineWeb10B into data/fineweb10B (N=1 on the laptop)
	$(PY) scripts/get_data.py $(or $(N),1)

venv:
	python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install numpy pytest tiktoken kaggle

kaggle-dryrun:  # Kaggle lane offline: pack trainer/ + scripts/, render smoke/full/chain kernels, validate; no Kaggle contact
	$(PY) lanes/kaggle_push.py dryrun
