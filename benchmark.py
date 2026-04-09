"""
Benchmarking harness for the attention profiling experiment.

Measures:
  - Wall-clock latency (CUDA events, warm-up + N_TRIALS)
  - Achieved memory bandwidth  (GB/s)
  - Achieved FLOP throughput   (TFLOP/s)
  - Arithmetic intensity       (FLOP/byte)
  - Output numerics (FP16 vs INT8 max-absolute-difference)

All results are returned as a pandas DataFrame for easy CSV export.
"""

import math
import time
from dataclasses import dataclass, asdict
from typing import Literal

import torch
import pandas as pd

from kernels  import fp16_attention, int8kv_attention
from quantize import (
    quantize_headwise,
    kv_memory_bytes,
    attention_flops,
    arithmetic_intensity,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    batch_size:  int   = 1
    num_heads:   int   = 16       # Qwen3-7B: 32 heads; use 16 for A100 profiling
    head_dim:    int   = 128
    seq_len:     int   = 512
    causal:      bool  = True
    n_warmup:    int   = 10
    n_trials:    int   = 50
    dtype:       str   = "fp16"   # "fp16" or "int8kv"
    block_m:     int   = 64
    block_n:     int   = 64


# ---------------------------------------------------------------------------
# Single-run timing
# ---------------------------------------------------------------------------

def _time_kernel(fn, n_warmup: int, n_trials: int) -> float:
    """Returns median latency in milliseconds."""
    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Timed trials
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_trials):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))   # ms

    times.sort()
    # Return median (robust to outliers)
    return times[n_trials // 2]


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def run_one(cfg: BenchConfig) -> dict:
    """
    Run a single (config, dtype) benchmark and return a result dict.
    """
    B, H, N, D = cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim
    device = "cuda"

    # Allocate inputs
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N, D, dtype=torch.float16, device=device)

    if cfg.dtype == "fp16":
        fn = lambda: fp16_attention(q, k, v, causal=cfg.causal,
                                    BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n)
        ref_out = fn()

    elif cfg.dtype == "int8kv":
        k_i8, k_scale = quantize_headwise(k)
        v_i8, v_scale = quantize_headwise(v)
        fn = lambda: int8kv_attention(
            q, k_i8, k_scale, v_i8, v_scale,
            causal=cfg.causal, BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n
        )
        ref_out = fn()

    else:
        raise ValueError(f"Unknown dtype: {cfg.dtype}")

    # Numerics: compare to PyTorch reference using flash attention (avoids
    # materializing the O(N^2) attention matrix which OOMs at large B*N)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        dropout_p=0.0,
        is_causal=cfg.causal,
    )
    max_diff = (ref_out - ref).abs().max().item()
    del ref, ref_out
    torch.cuda.empty_cache()

    # Latency
    latency_ms = _time_kernel(fn, cfg.n_warmup, cfg.n_trials)

    # Theoretical memory traffic
    mem = kv_memory_bytes(B, H, N, D, "fp16" if cfg.dtype == "fp16" else "int8")
    total_bytes = mem["total_bytes"]
    flops       = attention_flops(B, H, N, D, cfg.causal)
    ai_theory   = arithmetic_intensity(B, H, N, D,
                                       "fp16" if cfg.dtype == "fp16" else "int8",
                                       cfg.causal)

    # Achieved throughputs
    latency_s       = latency_ms / 1e3
    bw_achieved     = total_bytes / latency_s / 1e9    # GB/s
    flops_achieved  = flops / latency_s / 1e12         # TFLOP/s

    return {
        # Config
        "dtype":        cfg.dtype,
        "batch":        B,
        "heads":        H,
        "seq_len":      N,
        "head_dim":     D,
        "causal":       cfg.causal,
        # Latency
        "latency_ms":   round(latency_ms, 4),
        # Memory
        "total_MB":     round(total_bytes / 1e6, 2),
        "kv_MB":        round((mem["k_bytes"] + mem["v_bytes"]) / 1e6, 2),
        "bw_GBps":      round(bw_achieved, 2),
        # Compute
        "gflops":       round(flops / 1e9, 2),
        "tflops":       round(flops_achieved, 4),
        # Roofline
        "ai_theory":    round(ai_theory, 4),
        # Numerics
        "max_diff_vs_ref": round(max_diff, 6),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

SEQ_LENS   = [512, 2048, 4096]
BATCH_SIZES = [1, 4, 16]
DTYPES     = ["fp16", "int8kv"]


def run_sweep(
    seq_lens:    list[int] = SEQ_LENS,
    batch_sizes: list[int] = BATCH_SIZES,
    dtypes:      list[str] = DTYPES,
    num_heads:   int = 16,
    head_dim:    int = 128,
    causal:      bool = True,
    n_warmup:    int = 10,
    n_trials:    int = 50,
) -> pd.DataFrame:
    """
    Full sweep over seq_lens × batch_sizes × dtypes.
    Returns a DataFrame with one row per configuration.
    """
    rows = []
    total = len(seq_lens) * len(batch_sizes) * len(dtypes)
    i = 0

    for seq_len in seq_lens:
        for batch in batch_sizes:
            for dtype in dtypes:
                i += 1
                print(f"[{i:3d}/{total}] dtype={dtype:7s}  "
                      f"B={batch}  H={num_heads}  N={seq_len}  D={head_dim} ...",
                      end=" ", flush=True)
                cfg = BenchConfig(
                    batch_size=batch,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    seq_len=seq_len,
                    causal=causal,
                    n_warmup=n_warmup,
                    n_trials=n_trials,
                    dtype=dtype,
                )
                try:
                    torch.cuda.empty_cache()
                    row = run_one(cfg)
                    rows.append(row)
                    print(f"latency={row['latency_ms']:.2f} ms  "
                          f"BW={row['bw_GBps']:.1f} GB/s  "
                          f"AI={row['ai_theory']:.2f}  "
                          f"err={row['max_diff_vs_ref']:.5f}")
                except Exception as e:
                    print(f"FAILED: {e}")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Roofline summary
# ---------------------------------------------------------------------------

def roofline_summary(df: pd.DataFrame,
                     peak_bw_gbps: float = 2000.0,    # A100 HBM2e ~2 TB/s
                     peak_fp16_tflops: float = 312.0  # A100 FP16 tensor core
                     ) -> pd.DataFrame:
    """
    Add roofline columns to a results DataFrame.

    Roofline model:
        perf_ceiling = min(peak_compute, peak_bw * AI)
        hw_utilization = achieved / ceiling
    """
    df = df.copy()
    df["roof_bw_tflops"]   = (peak_bw_gbps * df["ai_theory"]) / 1e3
    df["roof_ceiling"]     = df[["roof_bw_tflops"]].clip(upper=peak_fp16_tflops).min(axis=1)
    df["bottleneck"]       = df["ai_theory"].apply(
        lambda ai: "memory" if ai < peak_fp16_tflops / peak_bw_gbps * 1e3 else "compute"
    )
    df["hw_util_pct"]      = (df["tflops"] / df["roof_ceiling"] * 100).round(2)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens",  nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument("--batches",   nargs="+", type=int, default=[1, 4])
    parser.add_argument("--heads",     type=int,  default=16)
    parser.add_argument("--head-dim",  type=int,  default=128)
    parser.add_argument("--n-warmup",  type=int,  default=10)
    parser.add_argument("--n-trials",  type=int,  default=50)
    parser.add_argument("--out",       type=str,  default="results.csv")
    args = parser.parse_args()

    df = run_sweep(
        seq_lens=args.seq_lens,
        batch_sizes=args.batches,
        num_heads=args.heads,
        head_dim=args.head_dim,
        n_warmup=args.n_warmup,
        n_trials=args.n_trials,
    )
    df = roofline_summary(df)

    df.to_csv(args.out, index=False)
    print(f"\nResults written to {args.out}")

    # Quick pivot for the paper
    pivot = df.pivot_table(
        index=["seq_len", "batch"],
        columns="dtype",
        values=["latency_ms", "bw_GBps", "ai_theory", "bottleneck"],
    )
    print("\n=== Roofline pivot ===")
    print(pivot.to_string())
