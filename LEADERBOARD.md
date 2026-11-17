# Leaderboard: your bill is zero

Train a language model to **3.28 FineWeb val loss** on compute that cost you nothing. The modded-nanogpt speedrun
races to that loss on 8×H100. This board races to it on free tiers.

## Rules

1. **Free tiers only.** Use the public free tier of a provider, open to anyone who signs up: Kaggle notebooks,
   Colab free, Lightning AI free credits, Hugging Face free Spaces, and the like. No paid credits, no trials that
   need a card on file, no research or startup grants, no employer or university clusters, no hardware you own.
   Your laptop's CPU is the one exception. Disclose it if you use it.
2. **One account per person per provider.** Several providers at once is fine (that's the interesting part).
   Several accounts at one provider is not.
3. **The standard val set.** Report loss on the first 10,485,760 tokens of `fineweb_val_000000.bin` from
   `kjj0/fineweb10B-gpt2` (the file upstream's `data/cached_fineweb10B.py` downloads), scored as upstream does:
   independent 65,536-token sequences, mean token cross-entropy.
   - file sha256: `5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640`
   - sha256 of tokens `[0, 10,485,761)` as uint16 little-endian:
     `ce6ad9a79700ee93c1e3edd25ecdcb06e279a958b5a3841c1dd538ef8819e375`
4. **Target 3.28.** An entry counts when its final val loss is ≤ 3.28. Train only on FineWeb train shards; val
   data must never be trained on.
5. **Receipts required.** Publish:
   - the training logs of every session, with val loss, step and token count;
   - each session's provider metadata (for Kaggle: `kaggle kernels pull -m` output, which shows the machine shape);
   - the val sha256s above, as computed by your run;
   - the git SHA of the code each session ran;
   - the final weights, or a re-score of them in a separate process.
   If a number in your claim can't be traced to a file, it doesn't count.
6. **Report time two ways.** Give the **session-hours** (wall-clock of the provider sessions, start to end, setup
   included) and the **GPU-hours** (session-hours × GPUs per session). Also give tokens trained, steps, and how many
   attempts you ran (failed ones too).
7. **Any model, any method.** Architecture, optimizer and tokens are up to you, as long as the val set and the
   tokenizer (GPT-2) match. Say which upstream record or recipe you started from.

Ranking: by GPU-hours to ≤ 3.28. Ties go to fewer session-hours.

## Entries

| # | Who | Hardware (provider) | Val loss | Tokens | Session-hours | GPU-hours | Sessions | Receipts |
|---|---|---|---|---|---|---|---|---|
| 1 | this repo, 2026-10-29 | 2×T4 (Kaggle free) | 3.2698 | 760M (1,450 steps) | 15.4 | 30.8 T4-h | 2 (11.7 h + 3.7 h), checkpoint resume | [receipts/](receipts/README.md) |

Entry #1 notes: an fp16 port of modded-nanogpt record #18 (2025-01-04 SoftCap). It was a single run, with no
failed full attempts. One fp16 overflow step was skipped, so 1,450 steps made 1,449 updates. The model has 275.7M
parameters including the embedding tables. The saved final weights re-score to 3.2698 in a separate process (fp16
and fp32). A laptop CPU ran only the watcher script that pulled session 1 and launched session 2. Including the
2.7 h smoke test, the probes and the 0.3 h re-score, the project used about 18.5 h of
Kaggle's ~30 h weekly quota.

## Submitting

Open an issue with a link to your receipts and one sentence in this form: model, hardware and provider, val loss,
tokens, session-hours, GPU-hours, number of sessions.
