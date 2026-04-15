"""
Modal app for weight quantization experiments (Experiment 2).

Usage:
    modal run modal_weight_quant.py                          # quantize model
    modal run modal_weight_quant.py --command profile        # layer profiling
    modal run modal_weight_quant.py --command ncu            # NCU on isolated FFN
"""
import modal
from pathlib import Path

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
weight_quant_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.5.1",
        "transformers==4.51.3",
        "accelerate==1.2.1",
        "autoawq==0.2.9",
        "autoawq-kernels==0.0.9",
        "datasets==2.21.0",
        "pandas==2.2.0",
        "numpy==1.26.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "nsight-compute 2>/dev/null || true",
    )
    .add_local_file("weight_quantization.py", remote_path="/app/weight_quantization.py")
    .add_local_file("profile_layers.py", remote_path="/app/profile_layers.py")
)

app = modal.App("weight-quant-exp2", image=weight_quant_image)
models_vol = modal.Volume.from_name("quantized-models", create_if_missing=True)
results_vol = modal.Volume.from_name("weight-quant-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------
@app.function(gpu="A100", timeout=3600,
              volumes={"/models": models_vol, "/results": results_vol})
def quantize_model(
    model_name: str = "Qwen/Qwen3-8B",
    output_name: str = "qwen3-8b-w4a8",
) -> dict:
    import sys; sys.path.insert(0, "/app")
    import torch
    from weight_quantization import quantize_model as _quantize

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    output_dir = f"/models/{output_name}"

    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        print(f"Model already exists at {output_dir}, skipping.")
        return {"status": "already_exists", "path": output_dir}

    model, tokenizer = _quantize(model_name, output_dir)
    models_vol.commit()
    return {"status": "success", "path": output_dir}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@app.function(gpu="A100-80GB", timeout=3600,
              volumes={"/models": models_vol, "/results": results_vol})
def profile_layers_sweep(
    model_fp16: str = "Qwen/Qwen3-8B",
    model_w4a8: str = "/models/qwen3-8b-w4a8",
    output_name: str = "results_exp2_layers.csv",
) -> bytes:
    import sys; sys.path.insert(0, "/app")
    import torch
    from profile_layers import profile_sweep
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_path = f"/results/{output_name}"
    df, ppl = profile_sweep(model_fp16, model_w4a8, out_path)
    results_vol.commit()

    # Append perplexity as a separate small CSV
    import pandas as pd
    ppl_df = pd.DataFrame([{"dtype": k, "perplexity": v} for k, v in ppl.items()])
    ppl_path = f"/results/perplexity.csv"
    ppl_df.to_csv(ppl_path, index=False)
    results_vol.commit()

    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# NCU profiling on isolated FFN forward pass
# ---------------------------------------------------------------------------

NCU_METRICS = ",".join([
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
])


@app.function(gpu="A100-80GB", timeout=1200,
              volumes={"/models": models_vol, "/results": results_vol})
def profile_ncu_ffn(
    model_fp16: str = "Qwen/Qwen3-8B",
    model_w4a8: str = "/models/qwen3-8b-w4a8",
    seq_len: int = 2048,
    batch: int = 1,
) -> str:
    """
    Run NCU on an isolated FFN (single MLP block) forward pass for
    both FP16 and W4A8, collecting DRAM bytes, SM utilization, and
    arithmetic intensity as specified in the Experiment 2 plan.

    Memory-efficient: loads one model at a time, extracts the MLP,
    deletes the full model, then profiles only the MLP forward pass.
    """
    import os
    import shutil
    import subprocess

    # Qwen3-8B architecture: hidden=4096, intermediate=12288, SwiGLU MLP
    driver_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 4096
INTER  = 12288
SEQ    = {seq_len}
BATCH  = {batch}
DEVICE = "cuda"

class SyntheticMLP(nn.Module):
    """Qwen3-8B SwiGLU MLP with FP16 weights (no model loading needed)."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.up_proj   = nn.Linear(HIDDEN, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def profile_fp16():
    print("=== NCU FFN: fp16 ===")
    mlp = SyntheticMLP().half().to(DEVICE).eval()
    x = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.float16, device=DEVICE)

    with torch.no_grad():
        for _ in range(3):
            _ = mlp(x)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        _ = mlp(x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    del mlp, x
    torch.cuda.empty_cache()


def profile_w4a8():
    print("\\n=== NCU FFN: w4a8 ===")
    from awq.modules.linear import WQLinear_GEMM
    import awq_ext

    w_bit, group_size = 4, 128

    def make_wqlinear(in_f, out_f):
        return WQLinear_GEMM(w_bit, group_size, in_f, out_f, bias=False, dev=DEVICE)

    gate = make_wqlinear(HIDDEN, INTER)
    up   = make_wqlinear(HIDDEN, INTER)
    down = make_wqlinear(INTER, HIDDEN)

    x = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.float16, device=DEVICE)

    with torch.no_grad():
        for _ in range(3):
            _ = down(F.silu(gate(x)) * up(x))
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        _ = down(F.silu(gate(x)) * up(x))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    del gate, up, down, x
    torch.cuda.empty_cache()


profile_fp16()
torch.cuda.synchronize()
profile_w4a8()
torch.cuda.synchronize()
'''

    driver_path = "/tmp/ncu_ffn_driver.py"
    with open(driver_path, "w") as f:
        f.write(driver_code)

    ncu_bin = shutil.which("ncu") or shutil.which(
        "ncu", path="/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2024.1/target/linux-desktop-glibc_2_11_3-x64"
    )
    if ncu_bin is None:
        for candidate in ["/usr/local/cuda/bin/ncu", "/usr/bin/ncu",
                          "/opt/nvidia/nsight-compute/2024.1/ncu"]:
            if os.path.exists(candidate):
                ncu_bin = candidate
                break
    if ncu_bin is None:
        return "ERROR: ncu binary not found in container"

    print(f"Using ncu: {ncu_bin}")

    try:
        with open("/proc/sys/kernel/perf_event_paranoid", "w") as f:
            f.write("-1")
        print("Set perf_event_paranoid=-1")
    except Exception as e:
        print(f"Could not set perf_event_paranoid: {e}")

    out_name = f"ncu_exp2_ffn_N{seq_len}_B{batch}.txt"
    txt_path = f"/tmp/{out_name}"
    rep_path = txt_path.replace(".txt", ".ncu-rep")

    ncu_cmd = [
        ncu_bin,
        "--target-processes", "all",
        "--clock-control", "none",
        "--replay-mode", "application",
        "--metrics", NCU_METRICS,
        "--csv",
        "--log-file", txt_path,
        "--export", rep_path,
        "--print-kernel-base", "function",
        "python", driver_path,
    ]

    print(f"Running: {' '.join(ncu_cmd)}")
    result = subprocess.run(ncu_cmd, capture_output=True, text=True, cwd="/tmp")

    stdout = result.stdout
    stderr = result.stderr
    print(f"NCU returncode: {result.returncode}")
    if stderr:
        print(f"NCU stderr (first 2000 chars): {stderr[:2000]}")

    if result.returncode != 0 and not stdout.strip():
        print(f"NCU failed (code {result.returncode}), using torch.profiler fallback")
        return _ffn_profiler_fallback(seq_len, batch)

    # Compute arithmetic intensity from raw NCU CSV output
    summary = _compute_ai_from_ncu(stdout)
    combined = stdout + "\n" + summary

    vol_path = f"/results/{out_name}"
    with open(vol_path, "w") as f:
        f.write(combined)
    results_vol.commit()
    print(f"NCU report saved to {vol_path}")

    return combined


def _ffn_profiler_fallback(seq_len: int, batch: int) -> str:
    """torch.profiler fallback when NCU is unavailable (same approach as Exp 1)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity

    HIDDEN, INTER = 4096, 12288
    device = "cuda"

    class SwiGLUMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
            self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)
            self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    results = []
    for dtype_label in ["fp16", "w4a8"]:
        if dtype_label == "fp16":
            mlp = SwiGLUMLP().half().to(device).eval()
        else:
            from awq.modules.linear import WQLinear_GEMM
            w_bit, group_size = 4, 128

            class AWQMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gate = WQLinear_GEMM(w_bit, group_size, HIDDEN, INTER, bias=False, dev=device)
                    self.up = WQLinear_GEMM(w_bit, group_size, HIDDEN, INTER, bias=False, dev=device)
                    self.down = WQLinear_GEMM(w_bit, group_size, INTER, HIDDEN, bias=False, dev=device)

                def forward(self, x):
                    return self.down(F.silu(self.gate(x)) * self.up(x))

            mlp = AWQMLP().eval()

        x = torch.randn(batch, seq_len, HIDDEN, dtype=torch.float16, device=device)

        with torch.no_grad():
            for _ in range(5):
                _ = mlp(x)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                      record_shapes=True, with_stack=False,
                      profile_memory=True) as prof:
            with torch.no_grad():
                _ = mlp(x)
            torch.cuda.synchronize()

        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=60)
        results.append(f"\n{'='*70}\nFFN PROFILE: {dtype_label.upper()} "
                       f"(batch={batch}, seq_len={seq_len}, dims={HIDDEN}->{INTER})\n{'='*70}\n{table}")

        # Theoretical memory traffic
        if dtype_label == "fp16":
            weight_bytes = (HIDDEN * INTER * 2 + HIDDEN * INTER * 2 + INTER * HIDDEN * 2)
            act_bytes = batch * seq_len * HIDDEN * 2
        else:
            weight_bytes = (HIDDEN * INTER // 2 + HIDDEN * INTER // 2 + INTER * HIDDEN // 2)
            act_bytes = batch * seq_len * HIDDEN * 2

        flops = batch * seq_len * (2 * HIDDEN * INTER + 2 * HIDDEN * INTER + 2 * INTER * HIDDEN)
        total_bytes = weight_bytes + act_bytes
        ai = flops / total_bytes if total_bytes > 0 else 0

        results.append(f"\nTheoretical estimates ({dtype_label}):")
        results.append(f"  Weight bytes:     {weight_bytes / 1e6:.1f} MB")
        results.append(f"  Activation bytes: {act_bytes / 1e6:.1f} MB")
        results.append(f"  FLOPs:            {flops / 1e9:.2f} GFLOP")
        results.append(f"  Arithmetic intensity: {ai:.2f} FLOP/byte")

        del mlp, x
        torch.cuda.empty_cache()

    output = "\n".join(results)
    vol_path = f"/results/ncu_exp2_ffn_N{seq_len}_B{batch}.txt"
    with open(vol_path, "w") as f:
        f.write(output)
    results_vol.commit()
    print(f"Profiler fallback saved to {vol_path}")
    return output


def _compute_ai_from_ncu(ncu_csv: str) -> str:
    """Parse NCU CSV output and compute arithmetic intensity per kernel."""
    import csv
    import io

    lines = [l for l in ncu_csv.strip().splitlines() if l and not l.startswith("==")]
    if not lines:
        return "\n# No NCU data to compute arithmetic intensity from.\n"

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        return "\n# No NCU kernel rows found.\n"

    parts = ["\n# --- Arithmetic Intensity Summary ---"]
    parts.append(f"{'Kernel':<60} {'DRAM Read':>12} {'DRAM Write':>12} {'HFMAs':>14} {'FLOPs':>14} {'AI (FLOP/B)':>12}")
    parts.append("-" * 130)

    for row in rows:
        name = row.get("Kernel Name", row.get("kernel_name", "unknown"))[:60]
        try:
            dr = float(row.get("dram__bytes_read.sum", 0))
            dw = float(row.get("dram__bytes_write.sum", 0))
            hfma = float(row.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0))
            ffma = float(row.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0))
        except (ValueError, TypeError):
            continue

        total_bytes = dr + dw
        flops = (hfma + ffma) * 2
        ai = flops / total_bytes if total_bytes > 0 else 0.0
        parts.append(f"{name:<60} {dr:>12.0f} {dw:>12.0f} {hfma:>14.0f} {flops:>14.0f} {ai:>12.4f}")

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(command: str = "quantize", seq_len: int = 2048):
    if command == "quantize":
        result = quantize_model.remote()
        print(f"Result: {result}")
    elif command == "profile":
        csv_bytes = profile_layers_sweep.remote()
        Path("results_exp2_layers.csv").write_bytes(csv_bytes)
        print("Downloaded results_exp2_layers.csv")
    elif command == "ncu":
        report = profile_ncu_ffn.remote(seq_len=seq_len)
        fname = f"ncu_exp2_ffn_N{seq_len}.txt"
        Path(fname).write_text(report)
        print(f"Downloaded {fname}")
    else:
        raise ValueError(f"Unknown command: {command}. Use: quantize, profile, ncu")
