# Experiment 2 — Weight Quantization: Findings

Qwen3-8B quantized to W4A8 using AWQ, checkpoint saved to Modal Volume at `/models/qwen3-8b-w4a8`. Layer-wise profiling sweep covers batch {1, 4, 16} x seq_len {512, 2048, 4096} for both FP16 and W4A8. Isolated FFN kernel profiling done at all three sequence lengths. Perplexity: FP16 = 12.27, W4A8 = 12.71 (+3.6%).

## Key Results

**Memory (the main win):** W4A8 reduces peak GPU memory by ~62% at batch=1 (16.6 GB to 6.3 GB) and ~43% at batch=16/seq=4096 (46.5 GB to 26.6 GB). This is the clearest practical benefit — same model on smaller hardware, or room for larger batches.

**Latency (counterintuitive):** W4A8 does not improve inference speed on A100. At batch=1/seq=512 it is 86% slower (189ms vs 102ms) due to AWQ's dequantization overhead. At batch=16/seq=4096 it roughly breaks even (~2% faster). The isolated FFN profiling confirms this — the W4A8 `gemm_forward_4bit_cuda` kernel runs ~3.4x slower per call than the FP16 CUTLASS GEMM kernel.

**Bottleneck shift (stable):** The attention-vs-FFN runtime split stays at ~30–35% attention / ~50% FFN across all configurations. Quantizing FFN weights does not cause attention to become the dominant bottleneck. The only exception is batch=1/seq=512, where AWQ overhead temporarily inflates FFN share.

## Outputs


| File                                | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| `results_exp2_layers.csv`           | 72-row CSV with per-layer time_ms, time_pct, mem_mb     |
| `figures/exp2_bottleneck_shift.png` | Attention vs FFN runtime % (FP16 vs W4A8)               |
| `figures/exp2_latency.png`          | Total inference latency across the grid                 |
| `figures/exp2_memory.png`           | Peak GPU memory across the grid                         |
| `ncu_exp2_ffn_N{512,2048,4096}.txt` | Isolated FFN kernel profiling with arithmetic intensity |


## Implications for Experiment 4

- The W4A8 checkpoint is ready to load for the combined configuration.
- The profiling hooks in `profile_layers.py` already separate attention / FFN / other, so the same infrastructure can measure the combined model.
- All three experiments use `Qwen/Qwen3-8B`, so model architecture is consistent.
- Since W4A8 primarily saves memory (not latency), the combined config should have headroom to absorb any overhead from INT8 KV or pruning without hitting A100 memory limits.
- Key question for Exp 4: does the ~60% memory reduction from W4A8 compose additively with KV cache and pruning savings, or do they overlap?

