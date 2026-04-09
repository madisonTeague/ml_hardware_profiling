# Attention Profiling Experiments

Hardware-aware comparison of **FP16** vs **INT8 KV** attention,
measuring HBM traffic reduction and roofline shift on A100 GPUs.

## File layout

```
attention_experiments/
├── kernels.py      # Triton kernels (FP16 + INT8 KV flash-attention)
├── quantize.py     # Per-head INT8 quantization + theoretical metrics
├── benchmark.py    # Timing harness, roofline analysis, CSV export
└── modal_app.py    # Modal runner (benchmark + NCU profiling)
```

## Setup

```bash
pip install modal torch triton pandas
modal setup          # authenticate once
```

## Running experiments

### 1 — Correctness check (fast, sanity-test kernels)
```bash
modal run modal_app.py --mode correctness
```
Expected: FP16 max_err < 1e-3, INT8KV max_err < 5e-2 (quantization noise)

### 2 — Full benchmark sweep (seq × batch × dtype)
```bash
modal run modal_app.py                       # uses defaults: seq=[512,2048,4096], batch=[1,4,16]
modal run modal_app.py --out results_exp1.csv
```
Downloads `results.csv` locally with latency, bandwidth, AI, and roofline columns.

### 3 — NCU kernel profiling (one config at a time)
```bash
# FP16 baseline at seq=2048
modal run modal_app.py --mode ncu --seq-len 2048 --dtype fp16

# INT8 KV at seq=2048
modal run modal_app.py --mode ncu --seq-len 2048 --dtype int8kv

# Also try seq=512 and seq=4096 for the three sequence-length points
```
NCU report is saved locally as `ncu_<dtype>_N<seq_len>.txt`.

### 4 — Retrieve saved results from Modal Volume
```bash
modal volume ls attn-results
modal volume get attn-results results.csv ./results_from_modal.csv
```

---

## What to measure / report

### Primary metrics (from `results.csv`)

| Column | Description |
|---|---|
| `latency_ms` | Median kernel latency (CUDA events) |
| `bw_GBps` | Achieved memory bandwidth (theoretical bytes / time) |
| `ai_theory` | Arithmetic intensity (FLOP/byte) |
| `bottleneck` | `memory` or `compute` (roofline classification) |
| `hw_util_pct` | % of roofline ceiling achieved |
| `max_diff_vs_ref` | Max abs error vs PyTorch SDPA |

### Secondary metrics (from NCU report)

Key NCU metrics to highlight in the paper:

```
dram__bytes_read.sum           → actual HBM reads (validates our theory)
dram__bytes_write.sum          → actual HBM writes
gpu__dram_throughput...        → % of peak HBM BW utilized
sm__throughput...              → % of peak SM throughput
sm__warps_active...            → occupancy
```

### Expected findings

- At **long seq_len** (4096): attention is memory-bound (AI < ridge point ~2 FLOP/byte on A100)
- INT8 KV reduces KV read bytes by **~50%** → total bytes ~**33% lower**
- This shifts AI rightward on the roofline plot
- At short seq_len (512): both variants may be closer to compute-bound

---

## Theoretical memory traffic

```
FP16 baseline:
  Reads:  Q[fp16] + K[fp16] + V[fp16]  =  3 * B*H*N*D * 2 bytes
  Writes: O[fp16]                       =      B*H*N*D * 2 bytes
  Total:  8 * B*H*N*D bytes

INT8 KV:
  Reads:  Q[fp16] + K[int8] + V[int8] + scales[fp16]
        = B*H*N*D*2 + B*H*N*D*1 + B*H*N*D*1 + B*H*4 bytes
  Writes: O[fp16]  =  B*H*N*D * 2 bytes
  Total:  ~6 * B*H*N*D bytes  (+tiny scale overhead)
  Reduction: ~25% vs FP16 total  (33% KV-only reduction)
```

---

## Roofline parameters (A100 SXM)

| Parameter | Value |
|---|---|
| Peak HBM bandwidth | 2,000 GB/s |
| Peak FP16 tensor core | 312 TFLOP/s |
| Ridge point | ~156 FLOP/byte |

At seq_len=2048 the arithmetic intensity is **well below** the ridge point,
so both variants are memory-bound — the BW reduction from INT8 KV should
translate almost linearly to latency reduction.

---

## Tuning block sizes

If latency is not as expected, try:
```python
# In benchmark.py BenchConfig
BLOCK_M = 128   # larger M blocks → better compute efficiency
BLOCK_N = 64    # larger N → fewer loop iterations
```

Triton autotune can also be added to `_attn_fp16_kernel` by replacing
the fixed `BLOCK_M/BLOCK_N` with `@triton.autotune(configs=[...], key=['N_CTX', 'HEAD_DIM'])`.
