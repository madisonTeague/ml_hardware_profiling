"""
Headwise INT8 quantization for KV cache.

"Headwise" = one (scale, zero_point) per attention head, shared across all
token positions in that head.  This matches the proposal's framing and is
the coarsest granularity that still gives decent accuracy (per-token or
per-channel would be finer but requires more metadata).

Quantization scheme: symmetric (zero_point = 0)
    scale  = max(|x|) / 127
    x_q    = round(clip(x / scale, -128, 127))
    x_deq  = x_q * scale
"""

import torch


# ---------------------------------------------------------------------------
# Core quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_headwise(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: [B, H, N, D]  float16 or float32

    Returns:
        x_int8: [B, H, N, D]  int8   (contiguous, same device as x)
        scale:  [B, H]        float16  per-head abs-max scale
    """
    # Per-head max absolute value  →  [B, H]
    abs_max = x.float().abs().amax(dim=(-2, -1))          # [B, H]
    scale   = (abs_max / 127.0).clamp(min=1e-8)           # avoid div-by-zero

    # Quantize
    x_scaled = x.float() / scale[:, :, None, None]        # [B, H, N, D]
    x_int8   = x_scaled.round().clamp(-128, 127).to(torch.int8)

    return x_int8.contiguous(), scale.to(torch.float16)


def dequantize_headwise(
    x_int8: torch.Tensor,   # [B, H, N, D]  int8
    scale:  torch.Tensor,   # [B, H]        float16
) -> torch.Tensor:
    """Reconstruct FP16 tensor from INT8 + per-head scale."""
    return (x_int8.to(torch.float16) * scale[:, :, None, None])


# ---------------------------------------------------------------------------
# Theoretical memory-traffic analysis
# ---------------------------------------------------------------------------

def kv_memory_bytes(B: int, H: int, N: int, D: int, dtype: str) -> dict:
    """
    Compute *theoretical* HBM bytes read for one attention forward pass.

    Counts:
      - Q reads  (always fp16)
      - K reads
      - V reads
      - Scale reads (int8 mode only)
      - Output writes (fp16)

    Ignores softmax intermediate writes (same for both modes).
    """
    bytes_per_elem = {"fp16": 2, "int8": 1}[dtype]
    q_bytes     = B * H * N * D * 2            # fp16
    k_bytes     = B * H * N * D * bytes_per_elem
    v_bytes     = B * H * N * D * bytes_per_elem
    out_bytes   = B * H * N * D * 2            # fp16
    scale_bytes = B * H * 2 * 2 if dtype == "int8" else 0   # K_scale + V_scale, fp16

    total = q_bytes + k_bytes + v_bytes + out_bytes + scale_bytes
    return {
        "q_bytes":     q_bytes,
        "k_bytes":     k_bytes,
        "v_bytes":     v_bytes,
        "out_bytes":   out_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total,
    }


def attention_flops(B: int, H: int, N: int, D: int, causal: bool = True) -> int:
    """
    FLOPs for one attention forward pass (multiply-adds × 2).
      QK^T:  B*H*N*N*D  MACs  (causal: /2)
      PV:    B*H*N*N*D  MACs  (causal: /2)
    """
    factor = 0.5 if causal else 1.0
    return int(4 * B * H * N * N * D * factor)   # 2 for MAC → FLOP


def arithmetic_intensity(B: int, H: int, N: int, D: int,
                         dtype: str, causal: bool = True) -> float:
    """FLOPs / byte  (roofline x-axis)."""
    flops = attention_flops(B, H, N, D, causal)
    mem   = kv_memory_bytes(B, H, N, D, dtype)["total_bytes"]
    return flops / mem


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_quantization_error(x: torch.Tensor, rtol: float = 0.05) -> dict:
    """
    Round-trip quantize → dequantize and report SNR / max relative error.

    Args:
        x: [B, H, N, D]  float16

    Returns:
        dict with snr_db, max_rel_err, mean_rel_err
    """
    x_i8, scale = quantize_headwise(x)
    x_deq       = dequantize_headwise(x_i8, scale).float()
    x_f         = x.float()

    err     = (x_f - x_deq).abs()
    signal  = x_f.pow(2).mean().sqrt()
    noise   = err.pow(2).mean().sqrt()
    snr_db  = 20 * torch.log10(signal / noise.clamp(min=1e-12))

    rel_err = err / (x_f.abs() + 1e-8)
    return {
        "snr_db":       snr_db.item(),
        "max_rel_err":  rel_err.max().item(),
        "mean_rel_err": rel_err.mean().item(),
        "passes":       rel_err.max().item() < rtol,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 512, 128
    x = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")

    result = check_quantization_error(x)
    print(f"Quantization SNR:       {result['snr_db']:.1f} dB")
    print(f"Max relative error:     {result['max_rel_err']:.4f}")
    print(f"Mean relative error:    {result['mean_rel_err']:.6f}")
    print(f"Passes 5% threshold:    {result['passes']}")

    # Memory traffic comparison
    for seq_len in [512, 2048, 4096]:
        fp16_mem = kv_memory_bytes(1, H, seq_len, D, "fp16")["total_bytes"] / 1e6
        i8_mem   = kv_memory_bytes(1, H, seq_len, D, "int8")["total_bytes"] / 1e6
        ai_fp16  = arithmetic_intensity(1, H, seq_len, D, "fp16")
        ai_i8    = arithmetic_intensity(1, H, seq_len, D, "int8")
        print(
            f"\nSeqLen={seq_len:5d} | "
            f"FP16 mem={fp16_mem:.1f} MB  AI={ai_fp16:.2f} | "
            f"INT8 mem={i8_mem:.1f} MB  AI={ai_i8:.2f} | "
            f"Reduction={100*(1-i8_mem/fp16_mem):.1f}%"
        )
