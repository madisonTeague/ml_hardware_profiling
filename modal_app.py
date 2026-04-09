"""
Modal app for hardware-aware attention profiling.

Usage:
    # Run full benchmark sweep, download results.csv
    modal run modal_app.py

    # Run NCU kernel-level profiling for a specific config
    modal run modal_app.py::profile_ncu --seq-len 2048 --dtype fp16

    # Interactive shell for debugging
    modal shell modal_app.py

Notes on NCU inside Modal:
  Modal containers run as root with CAP_SYS_ADMIN, which NCU needs.
  We install cuda-nsight-compute from the CUDA repo during image build.
  NCU output is returned as stdout and written to /tmp/ncu_report.ncu-rep
  which is then read back and returned.
"""

import io
import os
import subprocess
import tempfile
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

# We build a single image with PyTorch, Triton, and NSight Compute.
# NSight Compute is pulled from the NVIDIA CUDA repository (same as the
# driver version on Modal's A100 hosts – currently CUDA 12.x).

attention_image = (
    modal.Image.from_registry("nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "numpy",
        "pandas",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "nsight-compute 2>/dev/null || true",
    )
    # Copy our local experiment code into the image
    .add_local_dir(".", remote_path="/app", ignore=[".git", "__pycache__", "*.pyc", "*.pyo", "*.md", "*.csv", "*.txt", "*.ncu-rep"])
)


app = modal.App("attn-profiling", image=attention_image)


# ---------------------------------------------------------------------------
# Debug: isolate Triton crash
# ---------------------------------------------------------------------------

@app.function(gpu="A100", timeout=120)
def debug_triton():
    import subprocess, sys
    sys.path.insert(0, "/app")
    import torch, triton
    print(f"torch={torch.__version__}, triton={triton.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    script = r"""
import sys; sys.path.insert(0, "/app")
import torch
import triton
import triton.language as tl

device = "cuda"
B, H, N, D = 1, 4, 64, 64

# Step 1: trivial copy kernel
@triton.jit
def _copy_kernel(X, Y, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(Y + idx, tl.load(X + idx))

try:
    x = torch.randn(64, device=device, dtype=torch.float16)
    y = torch.empty_like(x)
    _copy_kernel[(1,)](x, y, N=64)
    torch.cuda.synchronize()
    print("PASS step 1: trivial copy kernel")
except Exception as e:
    print(f"FAIL step 1: {e}"); sys.exit(1)

# Step 2: tl.dot with out_dtype
@triton.jit
def _dot_kernel(A, B, C, BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
    a = tl.load(A + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]).to(tl.float16)
    b = tl.load(B + tl.arange(0, HEAD_DIM)[:, None] * BLOCK  + tl.arange(0, BLOCK)[None, :]).to(tl.float16)
    c = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], c)

try:
    a = torch.randn(64, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)
    c = torch.empty(64, 64, device=device, dtype=torch.float32)
    _dot_kernel[(1,)](a, b, c, BLOCK=64, HEAD_DIM=64)
    torch.cuda.synchronize()
    print("PASS step 2: tl.dot with out_dtype=float32")
except Exception as e:
    print(f"FAIL step 2: {e}"); sys.exit(1)

# Step 3: tl.dot inside a for loop
@triton.jit
def _dot_loop_kernel(A, B, C, BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    for _ in range(0, 4):
        a = tl.load(A + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]).to(tl.float16)
        b = tl.load(B + tl.arange(0, HEAD_DIM)[:, None] * BLOCK  + tl.arange(0, BLOCK)[None, :]).to(tl.float16)
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], acc)

try:
    a = torch.randn(64, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)
    c = torch.empty(64, 64, device=device, dtype=torch.float32)
    _dot_loop_kernel[(1,)](a, b, c, BLOCK=64, HEAD_DIM=64)
    torch.cuda.synchronize()
    print("PASS step 3: tl.dot inside for loop")
except Exception as e:
    print(f"FAIL step 3: {e}"); sys.exit(1)

# Step 4: fp16_attention causal=False
try:
    from kernels import fp16_attention
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
    k = torch.randn_like(q); v = torch.randn_like(q)
    fp16_attention(q, k, v, causal=False)
    torch.cuda.synchronize()
    print("PASS step 4: fp16_attention causal=False")
except Exception as e:
    print(f"FAIL step 4: {e}"); sys.exit(1)

# Step 5: fp16_attention causal=True
try:
    fp16_attention(q, k, v, causal=True)
    torch.cuda.synchronize()
    print("PASS step 5: fp16_attention causal=True")
except Exception as e:
    print(f"FAIL step 5: {e}"); sys.exit(1)

# Step 6a: load int8 tensor in kernel
@triton.jit
def _load_int8_kernel(X, Y, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(X + idx)
    tl.store(Y + idx, x.to(tl.float16))

try:
    from quantize import quantize_headwise
    k_i8, ks = quantize_headwise(k); v_i8, vs = quantize_headwise(v)
    flat = k_i8.reshape(-1)
    out_f = torch.empty(flat.shape, device=device, dtype=torch.float16)
    _load_int8_kernel[(1,)](flat, out_f, N=flat.shape[0])
    torch.cuda.synchronize()
    print("PASS step 6a: load int8 + cast to fp16")
except Exception as e:
    print(f"FAIL step 6a: {e}"); sys.exit(1)

# Step 6b: load scalar fp16 scale
@triton.jit
def _load_scale_kernel(S, Y, N: tl.constexpr):
    s = tl.load(S).to(tl.float16)
    idx = tl.arange(0, N)
    x = tl.load(Y + idx).to(tl.float16)
    tl.store(Y + idx, x * s)

try:
    flat_f = k_i8.reshape(-1).to(torch.float16)
    scale_val = ks.reshape(-1)[0:1].contiguous()
    _load_scale_kernel[(1,)](scale_val, flat_f, N=flat_f.shape[0])
    torch.cuda.synchronize()
    print("PASS step 6b: load scalar scale + multiply")
except Exception as e:
    print(f"FAIL step 6b: {e}"); sys.exit(1)

# Step 6c: int8 load + scalar scale + dot (no flash, no loop)
@triton.jit
def _int8_dot_kernel(Q, K_i8, K_scale, C, BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
    k_s = tl.load(K_scale).to(tl.float16)
    q = tl.load(Q + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]).to(tl.float16)
    k = tl.load(K_i8 + tl.arange(0, HEAD_DIM)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :])
    k = k.to(tl.float16) * k_s
    c = tl.dot(q, k, out_dtype=tl.float32)
    tl.store(C + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], c)

try:
    q2d = q[0, 0].contiguous()
    k2d = k_i8[0, 0].contiguous()
    scale_scalar = ks[0, 0:1].contiguous()
    c_out = torch.empty(N, N, device=device, dtype=torch.float32)
    _int8_dot_kernel[(1,)](q2d, k2d, scale_scalar, c_out, BLOCK=N, HEAD_DIM=D)
    torch.cuda.synchronize()
    print("PASS step 6c: int8 load + scalar scale + dot (2D)")
except Exception as e:
    print(f"FAIL step 6c: {e}"); sys.exit(1)

# Step 6d: same but with 3D grid (pid_b, pid_h separate)
@triton.jit
def _int8_dot_3d_kernel(Q, K_i8, K_scale, C,
                        stride_qh, stride_kh, stride_sh,
                        BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_h = tl.program_id(1)
    k_s = tl.load(K_scale + pid_h * stride_sh).to(tl.float16)
    q = tl.load(Q + pid_h * stride_qh + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]).to(tl.float16)
    k = tl.load(K_i8 + pid_h * stride_kh + tl.arange(0, HEAD_DIM)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :])
    k = k.to(tl.float16) * k_s
    c = tl.dot(q, k, out_dtype=tl.float32)
    tl.store(C + pid_h * BLOCK * BLOCK + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], c)

try:
    c_out = torch.empty(H, N, N, device=device, dtype=torch.float32)
    _int8_dot_3d_kernel[(1, H, 1)](
        q[0].contiguous(), k_i8[0].contiguous(), ks[0].contiguous(), c_out,
        q.stride(1), k_i8.stride(1), ks.stride(1),
        BLOCK=N, HEAD_DIM=D,
    )
    torch.cuda.synchronize()
    print("PASS step 6d: int8 + scalar scale + dot (3D grid)")
except Exception as e:
    print(f"FAIL step 6d: {e}"); sys.exit(1)

# Step 6e: flash loop with int8 k/v pre-dequantized to fp16 before kernel call
try:
    from quantize import dequantize_headwise
    k_fp16 = dequantize_headwise(k_i8, ks)
    v_fp16 = dequantize_headwise(v_i8, vs)
    from kernels import fp16_attention
    fp16_attention(q, k_fp16, v_fp16, causal=True)
    torch.cuda.synchronize()
    print("PASS step 6e: flash loop with pre-dequantized fp16 k/v")
except Exception as e:
    print(f"FAIL step 6e: {e}"); sys.exit(1)

# Step 6f: flash loop with int8 k/v dequantized inside loop using tl.broadcast_to
@triton.jit
def _attn_int8kv_inscale_kernel(
    Q, K_i8, K_scale, V_i8, V_scale, Out, sm_scale,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    stride_sh,
    N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    q_base  = Q    + pid_h * stride_qh
    k_base  = K_i8 + pid_h * stride_kh
    v_base  = V_i8 + pid_h * stride_vh
    o_base  = Out  + pid_h * stride_oh
    s_ptr   = pid_h * stride_sh   # pointer offset for this head's scales

    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, HEAD_DIM)
    n_range = tl.arange(0, BLOCK_N)
    q_mask  = m_range < N_CTX

    q2 = tl.load(q_base + m_range[:, None] * stride_qm + d_range[None, :] * stride_qd, mask=q_mask[:, None], other=0.0).to(tl.float16)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        kv_range = start_n + n_range
        kv_mask  = kv_range < N_CTX
        # Scales loaded inside loop to avoid loop-invariant scalar hoisting bug
        k_s = tl.load(K_scale + s_ptr).to(tl.float16)
        k_raw = tl.load(k_base + kv_range[None, :] * stride_kn + d_range[:, None] * stride_kd, mask=kv_mask[None, :], other=0)
        k2 = k_raw.to(tl.float16) * k_s
        qk = tl.dot(q2, k2, out_dtype=tl.float32) * sm_scale
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))
        m_j = tl.max(qk, axis=1); p = tl.exp(qk - m_j[:, None]); l_j = tl.sum(p, axis=1)
        m_new = tl.maximum(m_i, m_j); alpha = tl.exp(m_i - m_new); beta = tl.exp(m_j - m_new)
        l_i = alpha * l_i + beta * l_j
        v_s = tl.load(V_scale + s_ptr).to(tl.float16)
        v_raw = tl.load(v_base + kv_range[:, None] * stride_vn + d_range[None, :] * stride_vd, mask=kv_mask[:, None], other=0)
        v2 = v_raw.to(tl.float16) * v_s
        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p.to(tl.float16), v2, out_dtype=tl.float32)
        m_i = m_new

    out2 = (acc / l_i[:, None]).to(tl.float16)
    tl.store(o_base + m_range[:, None] * stride_om + d_range[None, :] * stride_od, out2, mask=q_mask[:, None])

try:
    out_buf = torch.empty_like(q[0])
    _attn_int8kv_inscale_kernel[(triton.cdiv(N, 64), H)](
        q[0].contiguous(), k_i8[0].contiguous(), ks[0].contiguous(),
        v_i8[0].contiguous(), vs[0].contiguous(), out_buf,
        float(D ** -0.5),
        q.stride(1), q.stride(2), q.stride(3),
        k_i8.stride(1), k_i8.stride(2), k_i8.stride(3),
        v_i8.stride(1), v_i8.stride(2), v_i8.stride(3),
        out_buf.stride(0), out_buf.stride(1), out_buf.stride(2),
        ks.stride(1),
        N, HEAD_DIM=D, BLOCK_M=64, BLOCK_N=64,
    )
    torch.cuda.synchronize()
    print("PASS step 6f: flash loop with scales inside loop")
except Exception as e:
    print(f"FAIL step 6f: {e}"); sys.exit(1)

# Step 6g: full int8kv kernel
try:
    from kernels import int8kv_attention
    int8kv_attention(q, k_i8, ks, v_i8, vs, causal=True)
    torch.cuda.synchronize()
    print("PASS step 6g: int8kv_attention causal=True")
except Exception as e:
    print(f"FAIL step 6g: {e}"); sys.exit(1)

print("ALL STEPS PASSED")
"""
    with open("/tmp/debug_kernels.py", "w") as f:
        f.write(script)
    result = subprocess.run(["python", "-u", "/tmp/debug_kernels.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-3000:])


# ---------------------------------------------------------------------------
# Volume for persisting results
# ---------------------------------------------------------------------------

results_vol = modal.Volume.from_name("attn-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helper: compile kernels (Triton JIT warms up on first call)
# ---------------------------------------------------------------------------

def _warmup_kernels():
    import sys
    sys.path.insert(0, "/app")
    import torch
    from kernels import fp16_attention, int8kv_attention
    from quantize import quantize_headwise

    device = "cuda"
    B, H, N, D = 1, 4, 64, 128
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    fp16_attention(q, k, v)
    k_i8, ks = quantize_headwise(k)
    v_i8, vs = quantize_headwise(v)
    int8kv_attention(q, k_i8, ks, v_i8, vs)
    torch.cuda.synchronize()
    print("Kernel warm-up complete.")


# ---------------------------------------------------------------------------
# Function 1: Full benchmark sweep
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={"/results": results_vol},
)
def run_benchmark(
    seq_lens:   list[int] = [512, 2048, 4096],
    batch_sizes: list[int] = [1, 4, 16],
    num_heads:  int = 16,
    head_dim:   int = 128,
    n_warmup:   int = 10,
    n_trials:   int = 50,
    out_name:   str = "results.csv",
) -> bytes:
    """
    Run the full seq_len × batch × dtype sweep.
    Returns the CSV bytes and also saves to the Modal Volume.
    """
    import sys
    sys.path.insert(0, "/app")
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    _warmup_kernels()

    from benchmark import run_sweep, roofline_summary
    df = run_sweep(
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        num_heads=num_heads,
        head_dim=head_dim,
        n_warmup=n_warmup,
        n_trials=n_trials,
    )
    df = roofline_summary(df)

    # Save to volume
    out_path = f"/results/{out_name}"
    df.to_csv(out_path, index=False)
    results_vol.commit()
    print(f"Saved to {out_path}")

    # Also return as bytes so the caller can download directly
    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Function 2: NCU kernel profiling
# ---------------------------------------------------------------------------

# NCU metric groups to capture:
#   - Memory: HBM throughput, L2 traffic
#   - Compute: SM utilization, FP16 throughput
#   - Roofline: achieved occupancy

NCU_METRICS = ",".join([
    # Memory
    "l1tex__t_bytes.sum",                           # L1 read bytes
    "lts__t_bytes.sum",                             # L2 read bytes
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",  # HBM util %
    "dram__bytes_read.sum",                         # HBM bytes read
    "dram__bytes_write.sum",                        # HBM bytes write
    # Compute
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",  # FP32 FMAs
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",  # FP16 FMAs
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
])


@app.function(
    gpu="A100",
    timeout=600,
    volumes={"/results": results_vol},
)
def profile_ncu(
    seq_len: int = 2048,
    batch:   int = 1,
    heads:   int = 16,
    head_dim: int = 128,
    dtype:   str = "fp16",    # "fp16" or "int8kv"
    causal:  bool = True,
    out_name: str = None,
) -> str:
    """
    Run a single attention forward pass under NCU and return the report.
    
    NCU captures kernel-level metrics for the Triton attention kernel,
    giving us the raw HBM bytes read/written and SM utilization needed
    to plot the roofline.
    """
    import sys
    sys.path.insert(0, "/app")

    if out_name is None:
        out_name = f"ncu_{dtype}_N{seq_len}_B{batch}.txt"

    # Write a tiny driver script that NCU will wrap
    driver_code = f"""
import sys; sys.path.insert(0, '/app')
import torch
from kernels import fp16_attention, int8kv_attention
from quantize import quantize_headwise

torch.cuda.cudart().cudaProfilerStart()

B, H, N, D = {batch}, {heads}, {seq_len}, {head_dim}
device = 'cuda'
q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)

if '{dtype}' == 'fp16':
    out = fp16_attention(q, k, v, causal={causal})
else:
    k_i8, ks = quantize_headwise(k)
    v_i8, vs = quantize_headwise(v)
    out = int8kv_attention(q, k_i8, ks, v_i8, vs, causal={causal})

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
"""
    driver_path = "/tmp/ncu_driver.py"
    with open(driver_path, "w") as f:
        f.write(driver_code)

    report_path = f"/tmp/{out_name.replace('.txt', '.ncu-rep')}"
    txt_path    = f"/tmp/{out_name}"

    # ── Locate ncu binary ───────────────────────────────────────────────────
    import shutil
    ncu_bin = shutil.which("ncu") or shutil.which("ncu", path="/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2024.1/target/linux-desktop-glibc_2_11_3-x64")
    if ncu_bin is None:
        for candidate in ["/usr/local/cuda/bin/ncu", "/usr/bin/ncu", "/opt/nvidia/nsight-compute/2024.1/ncu"]:
            if os.path.exists(candidate):
                ncu_bin = candidate
                break
    if ncu_bin is None:
        print("ncu not found, falling back to torch profiler")
        return _torch_profiler_fallback(batch, heads, seq_len, head_dim, dtype, causal)
    print(f"Using ncu: {ncu_bin}")

    # ── NCU invocation ──────────────────────────────────────────────────────
    # Grant perf counter access (requires root, which Modal containers have)
    try:
        with open("/proc/sys/kernel/perf_event_paranoid", "w") as f:
            f.write("-1")
        print("Set perf_event_paranoid=-1")
    except Exception as e:
        print(f"Could not set perf_event_paranoid: {e}")

    ncu_cmd = [
        ncu_bin,
        "--target-processes", "all",
        "--clock-control", "none",
        "--replay-mode", "application",  # replays whole app, avoids perf_event_paranoid requirement
        "--metrics", NCU_METRICS,
        "--csv",
        "--log-file", txt_path,
        "--export", report_path,
        "--print-kernel-base", "function",
        "python", driver_path,
    ]

    print(f"Running: {' '.join(ncu_cmd)}")
    result = subprocess.run(
        ncu_cmd,
        capture_output=True,
        text=True,
        cwd="/app",
    )

    stdout = result.stdout
    stderr = result.stderr
    print(f"NCU returncode: {result.returncode}")
    if stderr:
        print(f"NCU stderr: {stderr[:1000]}")

    if result.returncode != 0 and not stdout.strip():
        print("Falling back to torch.profiler...")
        return _torch_profiler_fallback(batch, heads, seq_len, head_dim, dtype, causal)

    # Save to volume
    vol_path = f"/results/{out_name}"
    with open(vol_path, "w") as f:
        f.write(stdout)
    results_vol.commit()
    print(f"NCU report saved to {vol_path}")

    return stdout


def _torch_profiler_fallback(B, H, N, D, dtype, causal):
    """torch.profiler-based fallback when NCU is unavailable."""
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    from kernels import fp16_attention, int8kv_attention
    from quantize import quantize_headwise

    device = "cuda"
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if dtype == "int8kv":
        k_i8, ks = quantize_headwise(k)
        v_i8, vs = quantize_headwise(v)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with record_function("attention_forward"):
            if dtype == "fp16":
                fp16_attention(q, k, v, causal=causal)
            else:
                int8kv_attention(q, k_i8, ks, v_i8, vs, causal=causal)

    report = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    )
    return report


# ---------------------------------------------------------------------------
# Function 3: Quick correctness + quantization-error check
# ---------------------------------------------------------------------------

@app.function(gpu="A100")
def check_correctness(
    seq_lens: list[int] = [512, 2048, 4096],
    heads:    int = 16,
    head_dim: int = 128,
) -> str:
    import sys
    sys.path.insert(0, "/app")
    import torch
    from quantize import check_quantization_error
    from kernels import fp16_attention, int8kv_attention
    from quantize import quantize_headwise

    lines = []
    device = "cuda"
    for N in seq_lens:
        q = torch.randn(1, heads, N, head_dim, dtype=torch.float16, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # FP16 kernel vs torch reference
        out_fp16 = fp16_attention(q, k, v, causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        fp16_err = (out_fp16 - ref).abs().max().item()

        # INT8 KV kernel vs torch reference
        k_i8, ks = quantize_headwise(k)
        v_i8, vs = quantize_headwise(v)
        out_i8 = int8kv_attention(q, k_i8, ks, v_i8, vs, causal=True)
        i8_err = (out_i8 - ref).abs().max().item()

        # Quantization round-trip SNR
        quant = check_quantization_error(k)

        lines.append(
            f"N={N:5d} | FP16 max_err={fp16_err:.5f} | "
            f"INT8KV max_err={i8_err:.5f} | "
            f"K quant SNR={quant['snr_db']:.1f} dB"
        )

    result = "\n".join(lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode:      str = "benchmark",   # "benchmark" | "ncu" | "correctness"
    seq_len:   int = 2048,
    dtype:     str = "fp16",
    out:       str = "results.csv",
):
    """
    Examples:
        modal run modal_app.py                          # full benchmark sweep
        modal run modal_app.py --mode ncu --seq-len 2048 --dtype fp16
        modal run modal_app.py --mode ncu --dtype int8kv
        modal run modal_app.py --mode correctness
    """
    if mode == "benchmark":
        csv_bytes = run_benchmark.remote(out_name=out)
        Path(out).write_bytes(csv_bytes)
        print(f"Downloaded results → {out}")

    elif mode == "ncu":
        report = profile_ncu.remote(seq_len=seq_len, dtype=dtype)
        fname = f"ncu_{dtype}_N{seq_len}.txt"
        Path(fname).write_text(report)
        print(f"NCU report → {fname}")
        print(report[:3000])

    elif mode == "correctness":
        result = check_correctness.remote()
        print(result)

    else:
        raise ValueError(f"Unknown mode: {mode}")
