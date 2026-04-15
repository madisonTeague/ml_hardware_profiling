# ***Hardware-Aware Model Compression Profiling* - Spring 2026 15-442/642 Course Project**

## Team

- **Madison Teague ([mteague@andrew.cmu.edu](mailto:mteague@andrew.cmu.edu))**
- **Jacob Scriffiny ([jscriffi@andrew.cmu.edu](mailto:jscriffi@andrew.cmu.edu))**
- **Gautam Diwan ([gdiwan@andrew.cmu.edu](mailto:gdiwan@andrew.cmu.edu))**

# Execution Plan
---

## Experiment 0 — Environment & Baseline Validation
- Confirm Modal A100 availability and CUDA version (nvidia-smi in interactive shell)
- verify FP16 kernel max error < 1e-3 vs PyTorch SDPA
- verify INT8 round-trip SNR > 30 dB
- Record baseline GPU specs (HBM BW, peak TFLOP/s) to anchor roofline plots
 
## Experiment 1 — KV Cache Quantization (Attention)
- Run full benchmark sweep 
    - Grid: seq_len ∈ {512, 2048, 4096} × batch ∈ {1, 4, 16} × dtype ∈ {fp16, int8kv}
    - Collect: latency (ms), achieved BW (GB/s), arithmetic intensity, roofline bottleneck classification
- Run NCU profiling for each of the 6 seq_len × dtype combos (batch=1 is sufficient for NCU) 
    - Collect from NCU: dram bytes read, dram bytes written, gpu bytes throughput, SM utilization, occupancy
- Key question to answer: does INT8 KV shift the kernel rightward on the roofline, and does latency scale proportionally to the ~33% theoretical memory reduction?
 
## Experiment 2 — Weight Quantization (FFN / Linear Layers)
- Quantize Qwen3-8B to W4A8 using AWQ 
    - Save quantized model checkpoint to Modal Volume so you don't re-quantize every run
- Run layer-wise profiling with torch.profiler on both FP16 and W4A8 models 
    - Grid: batch ∈ {1, 4, 16}, seq_len ∈ {512, 2048, 4096}
    - Collect per-layer: runtime (ms), % of total runtime, memory allocated
    - Separate attention layers from FFN/MLP layers in the profiler output
- Run NCU on isolated FFN forward pass (single nn.Linear call) for FP16 vs W4A8 
    - Collect: DRAM bytes, SM utilization, arithmetic intensity
- Key question to answer: does reducing FFN compute cost cause attention to become a larger share of total runtime (i.e., does the bottleneck shift)?
 
## Experiment 3 — Structural Pruning
- Apply PruneNet to Qwen3-8B at pruning ratios ∈ {10%, 20%, 30%} to get a spread of GEMM shape changes 
- For each pruned model, run inference profiling (batch ∈ {1, 4, 16}, seq_len ∈ {512, 2048, 4096}) 
    - Collect: throughput (tokens/sec), memory footprint (peak VRAM), GEMM shapes (log the weight matrix dims that change)
- Run NCU on the pruned FFN GEMM kernels 
    - Collect: SM utilization, achieved TFLOP/s — key check is whether smaller GEMMs lose efficiency (tile underutilization)
- Key question to answer: does pruning-induced GEMM shape change hurt hardware utilization even when parameter count drops?
 
# Experiment 4 — Combined Configuration
- Compose all three: W4A8 weight quant + INT8 KV attention + structural pruning (pick one pruning ratio, e.g. 20%)
- Run end-to-end inference on WikiText-103 (as specified in the proposal) 
    - Sequence lengths: 512, 2048, 4096 — batch size 1 (realistic serving scenario)
    - Collect: perplexity, total latency, per-layer runtime breakdown
- Run NCU on the combined model for the same 6 seq_len × dtype points as Experiment 1 
    - Collect same metrics as Exp 1 so you can directly compare roofline plots
- Key question to answer: are the gains additive, redundant, or do they conflict (e.g., does INT8 KV BW savings get masked when FFN is already the bottleneck from pruning)?
 
## Analysis & Writeup
- Roofline plots: one per experiment, overlay FP16 vs compressed variant — x-axis = arithmetic intensity, y-axis = TFLOP/s, mark A100 ridge point
- Layer-wise runtime breakdown bar charts: attention vs FFN share across compression configs
- KV memory traffic table: theoretical vs NCU-measured bytes for Exp 1 (validates your kernel)
- Perplexity vs hardware efficiency tradeoff table for the combined config
- One paragraph per experiment in the paper answering the specific research question listed above

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

---

## Experiment 3 — Structural Pruning

Magnitude-based structured pruning of FFN layers at 10%, 20%, 30% ratios.
Removes the least-important intermediate neurons from `gate_proj`,
`up_proj`, and `down_proj` in every transformer layer. No training,
no calibration, no fine-tuning of any kind.

Importance scores are computed with fused **Triton** kernels for
row/column L2 norms. All metrics are logged to **Weights & Biases**.

```
pruning/
├── prune_model.py      # Magnitude-based structured FFN pruning (Triton-accelerated)
├── profile_pruned.py   # Inference profiling (timing, throughput, VRAM, perplexity)
├── ncu_driver.py       # Standalone NCU driver script
├── modal_pruning.py    # Modal runner (prune + profile + NCU on A100)
└── analyze_pruning.py  # W&B analysis and visualization
```

### Setup (in addition to the base setup above)

```bash
pip install wandb datasets transformers accelerate safetensors
wandb login

# W&B secret for Modal (enables logging inside containers)
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

### Quick start — prune + profile in one command
```bash
cd pruning
modal run modal_pruning.py --command all
```

This will:
1. Prune Qwen3-8B at 10%, 20%, and 30% (saved to Modal volume)
2. Profile baseline + all pruned models across batch {1,4,16} x seq_len {512,2048,4096}
3. Download `results_exp3.csv` locally

### Step-by-step

#### 1. Prune
```bash
cd pruning
modal run modal_pruning.py --command prune
```
Produces three pruned checkpoints on the `pruned-models` Modal volume:
- `qwen3-8b-pruned-10pct` (intermediate_size: 12288 -> 11059)
- `qwen3-8b-pruned-20pct` (intermediate_size: 12288 -> 9830)
- `qwen3-8b-pruned-30pct` (intermediate_size: 12288 -> 8602)

#### 2. Profile
```bash
modal run modal_pruning.py --command profile
```
Runs the full sweep (baseline + 3 pruned variants, 9 batch/seq configs each).
Collects layer-wise timing, throughput (tokens/sec), peak VRAM, and
WikiText-103 perplexity. Downloads `results_exp3.csv`.

#### 3. NCU kernel profiling
```bash
# Baseline FFN GEMMs
modal run modal_pruning.py --command ncu --ratio 0.0 --seq-len 2048

# 20% pruned FFN GEMMs
modal run modal_pruning.py --command ncu --ratio 0.2 --seq-len 2048

# 30% pruned at seq_len=4096
modal run modal_pruning.py --command ncu --ratio 0.3 --seq-len 4096
```
NCU reports are saved locally as `ncu_pruning_<label>_N<seq>_B<batch>.txt`.

#### 4. Analyze (uploads charts to W&B)
```bash
cd pruning
python analyze_pruning.py \
    --input results_exp3.csv \
    --gemm-dir . \
    --perplexity perplexity_exp3.csv \
    --wandb-project ml-hw-profiling
```

#### 5. Enable W&B logging during prune/profile
```bash
modal run modal_pruning.py --command all --wandb-project ml-hw-profiling
```

### Running locally (if you have an A100)
```bash
cd pruning
python prune_model.py --model Qwen/Qwen3-8B --ratio 0.1 0.2 0.3 --wandb-project ml-hw-profiling
python profile_pruned.py \
    --baseline Qwen/Qwen3-8B \
    --pruned-10 ./pruned_10pct \
    --pruned-20 ./pruned_20pct \
    --pruned-30 ./pruned_30pct \
    --wandb-project ml-hw-profiling
```

### What gets measured

| Metric | Description |
|---|---|
| `time_ms` (attn / ffn / total) | Median CUDA-event latency per layer type |
| `time_pct` | % of total runtime per layer type |
| `throughput_tok_s` | Prefill throughput in tokens/sec |
| `peak_vram_mb` | Peak GPU memory during forward pass |
| `perplexity` | WikiText-103 test set perplexity |
| GEMM shapes | Original vs pruned weight dimensions per layer |
| NCU: SM utilization | % of peak SM throughput |
| NCU: DRAM throughput | HBM bytes read/written |

### Retrieve pruning results from Modal Volume
```bash
modal volume ls pruning-results
modal volume get pruning-results results_exp3.csv ./results_exp3.csv
modal volume get pruning-results perplexity_exp3.csv ./perplexity_exp3.csv
```

### Key question

> Does pruning-induced GEMM shape change hurt hardware utilization
> even when parameter count drops? (tile underutilization in smaller GEMMs)
