"""
Triton attention kernels for hardware-aware profiling.

Two variants:
  1. fp16_attention  – baseline: Q/K/V all in FP16
  2. int8kv_attention – INT8 KV: Q in FP16, K/V stored as INT8 with per-head scales

Both implement Flash-Attention-style tiled online softmax so that the
difference in HBM traffic comes *only* from the KV dtype, not from a
change in algorithm.  This makes roofline comparisons clean.

Grid layout: (ceil(N/BLOCK_M), H, B) — avoids runtime integer division
of program IDs which triggers MLIR lowering bugs in Triton.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: FP16 baseline
# ---------------------------------------------------------------------------

@triton.jit
def _attn_fp16_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N_CTX,
    HEAD_DIM:  tl.constexpr,
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_base = Q   + pid_b * stride_qb + pid_h * stride_qh
    k_base = K   + pid_b * stride_kb + pid_h * stride_kh
    v_base = V   + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh

    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, HEAD_DIM)
    n_range = tl.arange(0, BLOCK_N)
    q_mask  = m_range < N_CTX

    q = tl.load(
        q_base + m_range[:, None] * stride_qm + d_range[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0,
    ).to(tl.float16)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],               dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],      dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        kv_range = start_n + n_range
        kv_mask  = kv_range < N_CTX

        k = tl.load(
            k_base + kv_range[None, :] * stride_kn + d_range[:, None] * stride_kd,
            mask=kv_mask[None, :], other=0.0,
        ).to(tl.float16)

        qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale

        mask = kv_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (m_range[:, None] >= kv_range[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        m_j      = tl.max(qk, axis=1)
        # Use 0.0 as reference when m_j is -inf (fully masked tile) to avoid
        # -inf - (-inf) = nan. When m_j=-inf all qk are -inf so exp(-inf)=0 regardless.
        m_j_safe = tl.where(m_j == float("-inf"), 0.0, m_j)
        p        = tl.exp(qk - m_j_safe[:, None])
        l_j      = tl.sum(p, axis=1)
        m_new    = tl.maximum(m_i, m_j)
        alpha    = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        beta     = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_j - m_new))
        l_i      = alpha * l_i + beta * l_j

        v = tl.load(
            v_base + kv_range[:, None] * stride_vn + d_range[None, :] * stride_vd,
            mask=kv_mask[:, None], other=0.0,
        ).to(tl.float16)

        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_new

    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0).to(tl.float16)
    tl.store(
        o_base + m_range[:, None] * stride_om + d_range[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


def fp16_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    assert q.dtype == torch.float16
    B, H, N, D = q.shape
    assert D in (64, 128), "HEAD_DIM must be 64 or 128"
    out = torch.empty_like(q)
    sm_scale = D ** -0.5

    grid = (triton.cdiv(N, BLOCK_M), H, B)
    _attn_fp16_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 2: INT8 KV (per-head scale)
# ---------------------------------------------------------------------------

@triton.jit
def _attn_int8kv_kernel(
    Q,
    K_i8, K_scale,
    V_i8, V_scale,
    Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_sb, stride_sh,
    N_CTX,
    HEAD_DIM:  tl.constexpr,
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_base = Q    + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_i8 + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_i8 + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out  + pid_b * stride_ob + pid_h * stride_oh

    scale_ptr = pid_b * stride_sb + pid_h * stride_sh

    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, HEAD_DIM)
    n_range = tl.arange(0, BLOCK_N)
    q_mask  = m_range < N_CTX

    q = tl.load(
        q_base + m_range[:, None] * stride_qm + d_range[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0,
    ).to(tl.float16)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],               dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],      dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        kv_range = start_n + n_range
        kv_mask  = kv_range < N_CTX

        # Load scales inside loop to avoid loop-invariant scalar hoisting
        # which triggers an MLIR SSA lowering bug in Triton 3.x
        k_s = tl.load(K_scale + scale_ptr).to(tl.float16)
        k_raw = tl.load(
            k_base + kv_range[None, :] * stride_kn + d_range[:, None] * stride_kd,
            mask=kv_mask[None, :], other=0,
        )
        k = k_raw.to(tl.float16) * k_s

        qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale

        mask = kv_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (m_range[:, None] >= kv_range[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        m_j      = tl.max(qk, axis=1)
        # Use 0.0 as reference when m_j is -inf (fully masked tile) to avoid
        # -inf - (-inf) = nan. When m_j=-inf all qk are -inf so exp(-inf)=0 regardless.
        m_j_safe = tl.where(m_j == float("-inf"), 0.0, m_j)
        p        = tl.exp(qk - m_j_safe[:, None])
        l_j      = tl.sum(p, axis=1)
        m_new    = tl.maximum(m_i, m_j)
        alpha    = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        beta     = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_j - m_new))
        l_i      = alpha * l_i + beta * l_j

        v_s = tl.load(V_scale + scale_ptr).to(tl.float16)
        v_raw = tl.load(
            v_base + kv_range[:, None] * stride_vn + d_range[None, :] * stride_vd,
            mask=kv_mask[:, None], other=0,
        )
        v = v_raw.to(tl.float16) * v_s

        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_new

    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0).to(tl.float16)
    tl.store(
        o_base + m_range[:, None] * stride_om + d_range[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


def int8kv_attention(
    q:       torch.Tensor,
    k_i8:    torch.Tensor,
    k_scale: torch.Tensor,
    v_i8:    torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool = True,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    B, H, N, D = q.shape
    assert D in (64, 128)
    out = torch.empty_like(q)
    sm_scale = D ** -0.5

    grid = (triton.cdiv(N, BLOCK_M), H, B)
    _attn_int8kv_kernel[grid](
        q,
        k_i8, k_scale,
        v_i8, v_scale,
        out,
        sm_scale,
        q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
        k_i8.stride(0), k_i8.stride(1), k_i8.stride(2), k_i8.stride(3),
        v_i8.stride(0), v_i8.stride(1), v_i8.stride(2), v_i8.stride(3),
        out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
        k_scale.stride(0), k_scale.stride(1),
        N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
    )
    return out
