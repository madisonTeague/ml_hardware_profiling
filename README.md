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

