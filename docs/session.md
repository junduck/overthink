# Session Notes — 2026-04-13

## What We Built

### Model Architecture (DONE, working)
- **OverthinkBSQ** (`src/overthink/model/overthink_bsq.py`): HierarchicalTokenEmbedding + TemporalEmbedding + ReasoningBlock + two prediction heads (s1, s2)
- **ReasoningBlock** (`src/overthink/block/reasoning.py`): Single shared TransStack, local/global loop, zero init, stateless
- **BSQConfig** (`src/overthink/model/bsq_config.py`): Pydantic config with model/data/train sections
- Small: hidden=256, heads=8, stack=6 → ~5.4M params (~22MB)
- Medium: hidden=512, heads=8, stack=6 → ~16.9M params (~68MB)

### Online Training Pipeline (DONE, working but slow)
- **BSQOnlineDataset** (`src/overthink/data/dataset_online.py`): Streaming IterableDataset, reads per-code .pq files, cross-day context, train/val split
- **OnlineTokenizer** (`src/overthink/data/tokenize.py`): Frozen Kronos + instance-level z-score normalization + decode
- **Training script** (`scripts/train_bsq.py`): Online pipeline, eval metrics, JSON config + CLI overrides
- **Eval metrics** (`src/overthink/eval/metrics.py`): Perplexity, top-k accuracy, directional accuracy, return correlation
- Speed: ~3 batch/s on RTX 3090 (Kronos tokenizer every batch is the bottleneck)
- Training estimate: ~20 hrs/epoch — too slow for experimentation

### Pre-tokenized Pipeline (PARTIALLY DONE, has bugs)
- **BSQPreTokenizedDataset** (`src/overthink/data/dataset_pretokenized.py`): Reads per-code .pt files, streams samples — code is clean and correct
- **Training script** (`scripts/train_bsq.py`): Auto-detects pre-tokenized vs online, separate train/eval functions — working
- **Pre-tokenize script** (`scripts/pre_tokenize.py`): HAS BUGS, see below

### Config System (DONE)
- `BSQRunConfig` with `data`/`train`/`model` sections
- `--config` JSON file + CLI overrides
- Auto-detection: if `tokenized_dir` exists with .pt files → pre-tokenized path

### Repository (DONE)
- Branch: `kronos-tk-v0` on `origin/kronos-tk-v0`
- Squashed clean commit, no internal docs committed
- Workstation: `ssh junda@node-3090.local`, repo at `~/works/overthink`

---

## Data Layout on Workstation

```
~/Data/
  cn_1m_ohlcv/                        # Per-date whole-market files
    2018-01-02-ohlcv-1m.pq            # ~710K rows/day, ~3250 codes
    2018-01-03-ohlcv-1m.pq            # Columns: Code, TimeInterval, Open, High, Low, Close, Volume, Turnover
    ...                               # 1993 files, 2018-01-02 to 2026-04-10
  cn_1m_ohlcv_per_code/               # Per-code files (extracted from tarball)
    000001/
      2018-01-02.pq                   # ~240 bars per file
      ...
    7619 codes total
  tokenized/                          # Output of pre_tokenize.py (incomplete)
```

---

## Known Bugs in pre_tokenize.py

### Bug 1: Normalization corruption from padding

`_tokenize_code` pads variable-length samples with zeros, then calls `tokenizer.tokenize()` which computes z-score over the full padded sequence. The zeros pull mean toward 0 and inflate std.

**Fix options:**
1. Normalize per-sample BEFORE padding, then call Kronos encoder directly on normalized data
2. Pad with NaN and use nanmean/nanstd in OnlineTokenizer
3. Tokenize one sample at a time (slow but correct)

### Bug 2: Not tested end-to-end

The rewrite to load per-date whole-market files was done but timed out during testing on the workstation. Never verified that:
- Year loading works
- partition_by Code works correctly
- Cross-day windows are built correctly
- Output .pt files match what BSQPreTokenizedDataset expects

---

## Design Decisions Needed

### Data pipeline redesign

The current state is messy:
- `data_dir` → per-code layout (used by BSQOnlineDataset)
- `market_dir` → per-date whole-market layout (used by pre_tokenize.py)
- `tokenized_dir` → pre-tokenized output (used by BSQPreTokenizedDataset)
- Three paths for three purposes, confusing

**Options:**
1. **Keep both, fix the bug** — clean up pre_tokenize.py normalization, test end-to-end
2. **Single source of truth** — pick one raw data format, build everything from that
3. **Different approach entirely** — e.g. pre-tokenize into a single large file, or SQLite, or memory-mapped tensors

### Key question: what's the simplest correct pre-tokenize approach?

The whole point is to run Kronos tokenizer once. The simplest correct approach:
1. Load data (any format)
2. For each (code, date): build window, normalize, tokenize → save tokens
3. No batching tricks needed — just loop over samples, normalize each one, tokenize individually or in same-length groups

The Kronos tokenizer is ~31MB VRAM per sample at S=480. We can tokenize thousands per batch IF they're the same length. Most samples are ~480 bars. So:
- Group samples by length (most will be ~480)
- Normalize each sample individually
- Batch same-length normalized samples for GPU tokenization
- Save results

---

## Kronos Tokenizer Facts (profiled on RTX 3090)

| tokenizer_batch_size | Peak VRAM | Status |
|---|---|---|
| 1024 | 5.1 GB | OK |
| 2048 | 10.1 GB | OK (recommended) |
| 4096 | 20.2 GB | OK (max) |
| 5120 | OOM | |

Tokenizer params: ~4M (~16MB fp32). Per-sample VRAM at S=480: ~31MB.

---

## Hardware

- **Local (Mac)**: development, no training
- **Workstation (node-3090.local)**: RTX 3090 24GB VRAM, 256GB RAM, Ubuntu
  - `ssh junda@node-3090.local`
  - Repo: `~/works/overthink`
  - Data: `~/Data/`

---

## File Reference

### Model
- `src/overthink/model/overthink_bsq.py` — OverthinkBSQ
- `src/overthink/model/bsq_config.py` — BSQConfig, BSQDataConfig, BSQTrainConfig, BSQRunConfig
- `src/overthink/block/reasoning.py` — ReasoningBlock
- `src/overthink/block/transformer.py` — TransBlock, TransStack
- `src/overthink/layer/temporal.py` — TemporalEmbedding
- `src/overthink/layer/attention.py` — GQAttention
- `src/overthink/layer/rope.py` — RoPE
- `src/overthink/layer/swiglu.py` — SwiGLU

### Data
- `src/overthink/data/dataset_online.py` — BSQOnlineDataset (working)
- `src/overthink/data/dataset_pretokenized.py` — BSQPreTokenizedDataset (working)
- `src/overthink/data/tokenize.py` — OnlineTokenizer (working)

### Eval
- `src/overthink/eval/metrics.py` — perplexity, top-k, directional, correlation

### Scripts
- `scripts/train_bsq.py` — training (working for online, should work for pre-tokenized)
- `scripts/pre_tokenize.py` — BROKEN, needs redesign

### External
- `extern/kronos/` — Kronos tokenizer (git submodule, `model.kronos.KronosTokenizer`)
