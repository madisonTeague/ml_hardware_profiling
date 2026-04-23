"""
Modal app for structural pruning experiments (Experiment 3).

Usage:
    modal run modal_pruning.py                                 # prune at 10/20/30%
    modal run modal_pruning.py --command profile               # full profiling sweep
    modal run modal_pruning.py --command ncu --ratio 0.2       # NCU on 20% pruned FFN
    modal run modal_pruning.py --command all                   # prune + profile + NCU
"""

import modal
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Model directory helpers
# ---------------------------------------------------------------------------

def _method_slug(method: str) -> str:
    """Filesystem-safe method identifier.  'spectral+wanda' → 'spectral-wanda'."""
    return method.replace("+", "-")


def _tag_for_ratio(ratio: float, method: str = "") -> str:
    """Base tag without timestamp.

    With method:  'qwen3-8b-wanda-pruned-10pct'
    Without:      'qwen3-8b-pruned-10pct'  (legacy / baseline lookup)
    """
    pct = int(ratio * 100)
    if method:
        return f"qwen3-8b-{_method_slug(method)}-pruned-{pct}pct"
    return f"qwen3-8b-pruned-{pct}pct"


def _make_timestamped_dir(base_dir: str, ratio: float, method: str = "") -> str:
    """Create a new timestamped model directory path (does NOT mkdir)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{base_dir}/{_tag_for_ratio(ratio, method)}_{ts}"


def _find_latest_model(base_dir: str, ratio: float,
                       method: str = "") -> str | None:
    """Find the most recent timestamped model dir for a given ratio/method.

    Searches for dirs named like:
        qwen3-8b-wanda-pruned-10pct_20260420T...   (method-aware, preferred)
        qwen3-8b-pruned-10pct_20260415T...          (legacy, fallback)

    Returns the newest matching directory path, or None.
    """
    root = Path(base_dir)
    if not root.exists():
        return None

    # Build candidate prefixes: method-specific first, then legacy
    prefixes: list[str] = []
    if method:
        prefixes.append(_tag_for_ratio(ratio, method))
    prefixes.append(_tag_for_ratio(ratio))  # legacy / no-method dirs

    candidates: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        for pfx in prefixes:
            if name == pfx or name.startswith(pfx + "_"):
                if any(d.iterdir()):
                    candidates.append(d)
                break

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.name, reverse=True)
    return str(candidates[0])


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
pruning_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "transformers==4.51.3",
        "accelerate==1.2.1",
        "datasets==2.21.0",
        "pandas==2.2.0",
        "numpy==1.26.0",
        "safetensors",
        "wandb",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        # nsight-compute provides ncu; nsight-systems provides nsys (no perf-counter
        # privilege needed — uses CUDA runtime interception instead).
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "nsight-compute nsight-systems 2>/dev/null || true",
    )
    .add_local_file("prune_model.py", remote_path="/app/prune_model.py")
    .add_local_file("profile_pruned.py", remote_path="/app/profile_pruned.py")
    .add_local_file("ncu_driver.py", remote_path="/app/ncu_driver.py")
)

app = modal.App("pruning-exp3", image=pruning_image)
models_vol = modal.Volume.from_name("pruned-models", create_if_missing=True)
results_vol = modal.Volume.from_name("pruning-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/models": models_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
)
def prune_all_ratios(
    model_name: str = "Qwen/Qwen3-8B",
    ratios: list[float] = [0.05, 0.1, 0.2, 0.3],
    method: str = "wanda",
    n_calib_samples: int = 64,
    n_sv: int = 256,
    wandb_project: str | None = None,
    regenerate: bool = False,
) -> dict:
    import sys; sys.path.insert(0, "/app")
    import gc, os, torch
    from prune_model import prune_model

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = {}

    wb_project = wandb_project or os.environ.get("WANDB_PROJECT")

    for ratio in ratios:
        tag = _tag_for_ratio(ratio, method)

        if not regenerate:
            existing = _find_latest_model("/models", ratio, method)
            if existing:
                print(f"  {Path(existing).name}: already exists, reusing.")
                results[tag] = {"status": "already_exists", "path": existing}
                continue

        out_dir = _make_timestamped_dir("/models", ratio, method)

        model, tokenizer, gemm_rows = prune_model(
            model_name, ratio, out_dir,
            torch_dtype=torch.float16,
            method=method,
            n_calib_samples=n_calib_samples,
            n_sv=n_sv,
            wandb_project=wb_project,
        )

        import shutil
        slug = _method_slug(method)
        gemm_src = Path(out_dir) / "gemm_shapes.csv"
        if gemm_src.exists():
            shutil.copy(gemm_src,
                        f"/results/gemm_shapes_{slug}_{int(ratio*100)}pct.csv")

        models_vol.commit()
        results_vol.commit()
        results[tag] = {"status": "success", "path": out_dir}

        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/models": models_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
)
def profile_pruned_sweep(
    model_name: str = "Qwen/Qwen3-8B",
    ratios: list[float] = [0.1, 0.2, 0.3],
    method: str = "",
    output_name: str = "",
    wandb_project: str | None = None,
) -> bytes:
    import sys; sys.path.insert(0, "/app")
    import os, torch, pandas as pd
    from profile_pruned import profile_sweep

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    slug = _method_slug(method) if method else "mixed"
    if not output_name:
        output_name = f"results_{slug}.csv"

    pruned_paths = {}
    for r in ratios:
        latest = _find_latest_model("/models", r, method)
        if latest is None:
            print(f"WARNING: no {method or 'any'} model found for ratio {r}, skipping")
            continue
        print(f"  Using {Path(latest).name} for ratio {int(r*100)}%")
        pruned_paths[str(int(r * 100))] = latest

    wb_project = wandb_project or os.environ.get("WANDB_PROJECT")

    out_path = f"/results/{output_name}"
    df, ppl = profile_sweep(model_name, pruned_paths, out_path,
                            wandb_project=wb_project)
    results_vol.commit()

    ppl_df = pd.DataFrame([{"prune_ratio": k, "perplexity": v}
                           for k, v in ppl.items()])
    ppl_path = f"/results/perplexity_{slug}.csv"
    ppl_df.to_csv(ppl_path, index=False)
    results_vol.commit()

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# NCU profiling of FFN GEMM kernels
# ---------------------------------------------------------------------------

NCU_METRICS = ",".join([
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
])


def _parse_time_str(s: str) -> float:
    """Convert a profiler time string like '137.472ms' or '68.7us' to milliseconds."""
    s = s.strip()
    if s.endswith("ms"):
        return float(s[:-2])
    if s.endswith("us"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s") and not s.endswith("us") and not s.endswith("ms"):
        return float(s[:-1]) * 1000.0
    return 0.0


def _parse_pct(s: str) -> float:
    return float(s.strip().rstrip("%"))


def _log_torch_profiler_to_wandb(report: str, run_config: dict) -> None:
    """Parse the PyTorch profiler text table and log structured metrics to W&B."""
    import re, wandb

    # Fixed-width table: split on 2+ consecutive spaces
    # Columns: Name | Self CPU % | Self CPU | CPU total % | CPU total |
    #          CPU time avg | Self CUDA | Self CUDA % | CUDA total |
    #          CUDA time avg | CPU Mem | Self CPU Mem | CUDA Mem |
    #          Self CUDA Mem | # of Calls | Total KFLOPs
    COL_NAME, COL_SELF_CUDA, COL_SELF_CUDA_PCT, COL_CUDA_TOTAL, COL_CALLS, COL_KFLOPS = (
        0, 6, 7, 8, 14, 15,
    )

    # Ops to extract in priority order (first match wins per key)
    targets = {
        "total_forward":    re.compile(r"forward_pass"),
        "gemm":             re.compile(r"aten::mm\b"),
        "flash_attn":       re.compile(r"aten::_flash_attention_forward"),
        "elementwise":      re.compile(r"aten::mul\b"),
        "layernorm":        re.compile(r"aten::rms_norm|aten::layer_norm"),
        "gemm_kernel_ampere": re.compile(r"ampere_fp16_s16816gemm"),
    }

    extracted: dict[str, dict] = {}
    for line in report.splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        cols = re.split(r"\s{2,}", line)
        if len(cols) < 9:
            continue
        name = cols[COL_NAME].strip()
        for key, pat in targets.items():
            if key in extracted:
                continue
            if pat.search(name):
                try:
                    extracted[key] = {
                        "op": name[:60],
                        "self_cuda_ms":  _parse_time_str(cols[COL_SELF_CUDA]),
                        "self_cuda_pct": _parse_pct(cols[COL_SELF_CUDA_PCT]),
                        "cuda_total_ms": _parse_time_str(cols[COL_CUDA_TOTAL]),
                        "n_calls":       int(cols[COL_CALLS]) if cols[COL_CALLS].strip().isdigit() else 0,
                        "kflops":        float(cols[COL_KFLOPS]) if len(cols) > COL_KFLOPS and cols[COL_KFLOPS] not in ("--", "") else 0.0,
                    }
                except (ValueError, IndexError):
                    pass

    # ---- Scalar summary metrics ----
    log: dict = {}
    for key, data in extracted.items():
        log[f"ncu/{key}/cuda_total_ms"]  = data["cuda_total_ms"]
        log[f"ncu/{key}/self_cuda_ms"]   = data["self_cuda_ms"]
        log[f"ncu/{key}/self_cuda_pct"]  = data["self_cuda_pct"]
        if data["kflops"] > 0:
            log[f"ncu/{key}/kflops"] = data["kflops"]
        if data["n_calls"] > 0:
            log[f"ncu/{key}/n_calls"] = data["n_calls"]

    # Derived: GEMM share of total forward-pass time
    if "total_forward" in extracted and "gemm" in extracted:
        total_ms = extracted["total_forward"]["cuda_total_ms"]
        gemm_ms  = extracted["gemm"]["self_cuda_ms"]
        if total_ms > 0:
            log["ncu/gemm_share_of_total_pct"] = 100.0 * gemm_ms / total_ms

    # Derived: attention share
    if "total_forward" in extracted and "flash_attn" in extracted:
        total_ms  = extracted["total_forward"]["cuda_total_ms"]
        attn_ms   = extracted["flash_attn"]["cuda_total_ms"]
        if total_ms > 0:
            log["ncu/attn_share_of_total_pct"] = 100.0 * attn_ms / total_ms

    # Derived: TFLOPS from aten::mm (kflops × 1e3 → FLOP, divide by time in s)
    if "gemm" in extracted:
        g = extracted["gemm"]
        if g["kflops"] > 0 and g["self_cuda_ms"] > 0:
            tflops = (g["kflops"] * 1e3) / (g["self_cuda_ms"] * 1e-3) / 1e12
            log["ncu/gemm_tflops"] = tflops

    # ---- W&B Table: one row per op ----
    table = wandb.Table(
        columns=["op_key", "op_name", "self_cuda_ms", "self_cuda_pct",
                 "cuda_total_ms", "n_calls", "kflops"],
    )
    for key, data in extracted.items():
        table.add_data(
            key, data["op"], data["self_cuda_ms"], data["self_cuda_pct"],
            data["cuda_total_ms"], data["n_calls"], data["kflops"],
        )
    log["ncu/kernel_breakdown"] = table

    wandb.log(log)


def _log_ncu_csv_to_wandb(report: str, run_config: dict) -> None:
    """Parse NCU --csv output (long format) and log per-kernel metrics to W&B."""
    import io, csv, collections, wandb

    reader = csv.DictReader(io.StringIO(report))
    # Pivot: kernel_name -> {metric_name -> value}
    kernels: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in reader:
        kname  = row.get("Kernel Name", row.get("\"Kernel Name\"", "unknown")).strip('"')
        metric = row.get("Metric Name",  row.get("\"Metric Name\"", "")).strip('"')
        val    = row.get("Metric Value", row.get("\"Metric Value\"", "0")).strip('"').replace(",", "")
        try:
            kernels[kname][metric] = float(val)
        except ValueError:
            pass

    log: dict = {}
    table = wandb.Table(
        columns=["kernel", "dram_read_gb", "dram_write_gb",
                 "dram_tp_pct", "sm_tp_pct", "hfma_ops_b",
                 "warp_occ_pct", "l1_gb", "l2_gb"],
    )

    for kname, metrics in kernels.items():
        dram_read  = metrics.get("dram__bytes_read.sum", 0) / 1e9
        dram_write = metrics.get("dram__bytes_write.sum", 0) / 1e9
        dram_tp    = metrics.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 0)
        sm_tp      = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        hfma       = metrics.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0) / 1e9
        occ        = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
        l1         = metrics.get("l1tex__t_bytes.sum", 0) / 1e9
        l2         = metrics.get("lts__t_bytes.sum", 0) / 1e9
        table.add_data(kname[:60], dram_read, dram_write, dram_tp, sm_tp, hfma, occ, l1, l2)

    # Aggregate across all kernels
    all_metrics_list = list(kernels.values())
    if all_metrics_list:
        def avg(key): return sum(m.get(key, 0) for m in all_metrics_list) / len(all_metrics_list)
        log["ncu/avg_dram_throughput_pct"]    = avg("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
        log["ncu/avg_sm_throughput_pct"]      = avg("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        log["ncu/avg_warp_occupancy_pct"]     = avg("sm__warps_active.avg.pct_of_peak_sustained_active")
        log["ncu/total_hfma_ops_billions"]    = sum(m.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0) for m in all_metrics_list) / 1e9
        log["ncu/total_dram_read_gb"]         = sum(m.get("dram__bytes_read.sum", 0) for m in all_metrics_list) / 1e9
        log["ncu/n_kernels"]                  = len(kernels)

    log["ncu/kernel_breakdown"] = table
    wandb.log(log)


def _find_binary(*candidates: str) -> str | None:
    """Return the first binary that exists, checking PATH then explicit paths."""
    import shutil, os
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
        if os.path.exists(name):
            return name
    return None


def _run_ncu(ncu_bin: str, model_path: str, seq_len: int,
             batch_size: int, out_name: str) -> tuple[str, bool]:
    """Try NCU with kernel-replay mode (no --target-processes, no application replay).

    kernel-replay replays each kernel individually without re-running the full
    application, so it does NOT need SYS_PTRACE or SYS_ADMIN — just GPU perf-
    counter access.  Returns (output, success).
    """
    import os, subprocess

    txt_path = f"/tmp/{out_name}"
    env = os.environ.copy()
    env["MODEL_PATH"]  = model_path
    env["BATCH_SIZE"]  = str(batch_size)
    env["SEQ_LEN"]     = str(seq_len)

    # Lower-privilege NCU invocation:
    #   --replay-mode kernel   → replay individual kernels (no full app re-run)
    #   NO --target-processes  → trace only the directly launched process
    #   --clock-control none   → no clock manipulation (avoids some driver hooks)
    ncu_cmd = [
        ncu_bin,
        "--clock-control", "none",
        "--replay-mode", "kernel",
        "--metrics", NCU_METRICS,
        "--csv",
        "--log-file", txt_path,
        "--print-kernel-base", "function",
        "python", "/app/ncu_driver.py",
    ]
    print(f"[ncu] Running: {' '.join(ncu_cmd)}")
    res = subprocess.run(ncu_cmd, capture_output=True, text=True,
                         cwd="/app", env=env, timeout=900)
    print(f"[ncu] returncode={res.returncode}  "
          f"stderr={res.stderr[-300:] if res.stderr else '(none)'}")

    if res.returncode == 0 and res.stdout.strip():
        return res.stdout, True
    return res.stdout or "", False


def _run_nsys(nsys_bin: str, model_path: str, seq_len: int,
              batch_size: int) -> tuple[str, bool]:
    """Profile with Nsight Systems (no hardware perf counters needed).

    nsys intercepts the CUDA runtime API, so it works in unprivileged
    containers.  Returns the combined stdout+stderr containing the
    CUDA Kernel Statistics table printed by --stats=true.
    """
    import os, subprocess

    env = os.environ.copy()
    env["MODEL_PATH"]  = model_path
    env["BATCH_SIZE"]  = str(batch_size)
    env["SEQ_LEN"]     = str(seq_len)

    nsys_cmd = [
        nsys_bin, "profile",
        "--trace=cuda",
        "--stats=true",          # print CUDA kernel summary to stdout
        "--force-overwrite", "true",
        "--output", "/tmp/nsys_prof",
        "python", "/app/ncu_driver.py",
    ]
    print(f"[nsys] Running: {' '.join(nsys_cmd)}")
    res = subprocess.run(nsys_cmd, capture_output=True, text=True,
                         cwd="/app", env=env, timeout=900)
    print(f"[nsys] returncode={res.returncode}  "
          f"stderr_tail={res.stderr[-200:] if res.stderr else '(none)'}")

    # nsys prints the stats table to stderr; combine both streams
    combined = (res.stdout or "") + "\n" + (res.stderr or "")
    ok = res.returncode == 0 or "CUDA Kernel Statistics" in combined
    return combined, ok


def _log_nsys_to_wandb(report: str, run_config: dict) -> None:
    """Parse Nsight Systems --stats=true output and log to W&B.

    The relevant section looks like:
    CUDA Kernel Statistics:
     Time (%)  Total Time (ns)  Instances   Avg (ns)  ...  Name
       67.3     140116000         253  ...              ampere_fp16_...
    """
    import re, wandb

    # ---- Parse the CUDA Kernel Statistics table ----
    # Section starts after "CUDA Kernel Statistics:" header
    section_match = re.search(
        r"CUDA Kernel Statistics:.*?\n(.*?)(?:\n\n|\Z)", report,
        re.S | re.I,
    )
    if not section_match:
        print("[nsys] Could not find CUDA Kernel Statistics section in output.")
        return

    rows_text = section_match.group(1)
    kernel_rows = []
    for line in rows_text.splitlines():
        line = line.strip()
        if not line or line.startswith("Time") or line.startswith("-"):
            continue
        # Columns: Time(%) TotalTime(ns) Instances Avg Med Min Max StdDev Name
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 5:
            continue
        try:
            kernel_rows.append({
                "time_pct":    float(parts[0]),
                "total_ns":    float(parts[1]),
                "instances":   int(parts[2]),
                "avg_ns":      float(parts[3]),
                "name":        parts[-1][:60],
            })
        except (ValueError, IndexError):
            continue

    if not kernel_rows:
        return

    total_ns = sum(r["total_ns"] for r in kernel_rows)

    # ---- Classify and aggregate key groups ----
    def _match(name, *pats):
        return any(p in name for p in pats)

    gemm_rows  = [r for r in kernel_rows if _match(r["name"], "gemm", "Kernel2")]
    attn_rows  = [r for r in kernel_rows if _match(r["name"], "flash", "fmha")]
    elem_rows  = [r for r in kernel_rows if _match(r["name"], "elementwise",
                                                    "vectorized_elementwise")]

    def _sum_ns(rows): return sum(r["total_ns"] for r in rows)

    log: dict = {
        "ncu/total_cuda_time_ms":   total_ns / 1e6,
        "ncu/gemm_time_ms":         _sum_ns(gemm_rows) / 1e6,
        "ncu/attn_time_ms":         _sum_ns(attn_rows) / 1e6,
        "ncu/elementwise_time_ms":  _sum_ns(elem_rows) / 1e6,
        "ncu/n_kernels":            len(kernel_rows),
    }
    if total_ns > 0:
        log["ncu/gemm_share_of_total_pct"]  = 100 * _sum_ns(gemm_rows) / total_ns
        log["ncu/attn_share_of_total_pct"]  = 100 * _sum_ns(attn_rows) / total_ns

    # ---- W&B Table ----
    table = wandb.Table(
        columns=["kernel", "time_pct", "total_ms", "instances", "avg_us"])
    for r in sorted(kernel_rows, key=lambda x: -x["total_ns"])[:20]:
        table.add_data(r["name"], r["time_pct"],
                       r["total_ns"] / 1e6,
                       r["instances"],
                       r["avg_ns"] / 1e3)
    log["ncu/kernel_breakdown"] = table

    wandb.log(log)


@app.function(
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/models": models_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
)
def profile_ncu(
    model_path: str = "Qwen/Qwen3-8B",
    seq_len: int = 2048,
    batch_size: int = 1,
    label: str = "baseline",
    ratio: float = 0.0,
    method: str = "",
    wandb_project: str | None = None,
) -> str:
    """GPU profiling of a forward pass via three-tier fallback.

    Tier 1 — NCU (kernel-replay, no --target-processes):
        Requires GPU perf-counter access. Gives hardware metrics (DRAM
        throughput, SM utilisation, FLOP counts).
        Fails with exit-9 (SIGKILL) when the container seccomp profile
        blocks perf_event_open — the most common Modal failure mode.

    Tier 2 — nsys (Nsight Systems, --stats=true):
        Uses CUDA runtime API interception — NO perf counters needed.
        Works in unprivileged containers. Gives per-kernel CUDA timing.

    Tier 3 — PyTorch profiler:
        Pure Python, always works. Gives op-level CUDA timing + KFLOPs.
    """
    import os

    if ratio > 0:
        resolved = _find_latest_model("/models", ratio, method)
        if resolved:
            model_path = resolved
            print(f"  Profiling resolved to {Path(resolved).name}")
        elif not model_path:
            raise RuntimeError(
                f"No pruned model found for ratio={ratio} method={method!r}. "
                "Run --command prune first."
            )

    # Lower perf_event_paranoid — helps if the kernel allows it at all
    try:
        with open("/proc/sys/kernel/perf_event_paranoid", "w") as f:
            f.write("-1")
        print("  perf_event_paranoid set to -1")
    except Exception as e:
        print(f"  perf_event_paranoid: could not set ({e})")

    slug = _method_slug(method) if method else ""
    out_name = (f"ncu_{slug}_{label}_N{seq_len}_B{batch_size}.txt"
                if slug else f"ncu_{label}_N{seq_len}_B{batch_size}.txt")
    report   = ""
    profiler = "unknown"

    # ---- Tier 1: NCU ----
    ncu_bin = _find_binary("ncu", "/usr/local/cuda/bin/ncu", "/usr/bin/ncu")
    if ncu_bin:
        report, ok = _run_ncu(ncu_bin, model_path, seq_len, batch_size, out_name)
        if ok:
            profiler = "ncu"
            print("[ncu] Success.")
        else:
            print("[ncu] Failed (likely seccomp/privilege). Trying nsys ...")
    else:
        print("[ncu] Binary not found. Trying nsys ...")

    # ---- Tier 2: nsys ----
    if profiler == "unknown":
        nsys_bin = _find_binary(
            "nsys", "/usr/local/cuda/bin/nsys",
            "/opt/nvidia/nsight-systems/2024.3.1/bin/nsys",
        )
        if nsys_bin:
            report, ok = _run_nsys(nsys_bin, model_path, seq_len, batch_size)
            if ok:
                profiler = "nsys"
                print("[nsys] Success.")
            else:
                print("[nsys] Failed. Falling back to torch profiler ...")
        else:
            print("[nsys] Binary not found. Falling back to torch profiler ...")

    # ---- Tier 3: torch profiler ----
    if profiler == "unknown":
        report   = _torch_profiler_fallback(model_path, seq_len, batch_size)
        profiler = "torch_profiler"
        print("[torch_profiler] Done.")

    # ---- Persist report ----
    vol_path = f"/results/{out_name}"
    with open(vol_path, "w") as f:
        f.write(report)
    results_vol.commit()
    print(f"Profiling report ({profiler}) saved to {vol_path}")

    # ---- W&B ----
    wb_project = wandb_project or os.environ.get("WANDB_PROJECT")
    if wb_project:
        try:
            import wandb
            run_config = {
                "label":            label,
                "seq_len":          seq_len,
                "batch_size":       batch_size,
                "prune_ratio":      ratio,
                "method":           method or "baseline",
                "tokens_per_forward": seq_len * batch_size,
                "profiler":         profiler,
                "model_path":       model_path,
            }
            m_slug = _method_slug(method) if method else "baseline"
            wandb.init(
                project=wb_project,
                name=f"ncu-{m_slug}-{label}-N{seq_len}-B{batch_size}",
                config=run_config,
                tags=["ncu", label, profiler, m_slug,
                      f"seq{seq_len}", f"bs{batch_size}"],
            )
            if profiler == "ncu":
                _log_ncu_csv_to_wandb(report, run_config)
            elif profiler == "nsys":
                _log_nsys_to_wandb(report, run_config)
            else:
                _log_torch_profiler_to_wandb(report, run_config)
            wandb.finish()
            print(f"W&B run logged to project '{wb_project}' (profiler={profiler}).")
        except Exception as e:
            print(f"W&B NCU logging failed (non-fatal): {e}")

    return report


def _torch_profiler_fallback(model_path, seq_len, batch_size):
    """Fallback when NCU is unavailable: run torch.profiler and return text table."""
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    input_ids = torch.randint(0, tokenizer.vocab_size,
                              (batch_size, seq_len), device="cuda")

    with torch.no_grad():
        model(input_ids)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_flops=True,
    ) as prof:
        with record_function("forward_pass"):
            with torch.no_grad():
                model(input_ids)

    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_DEFAULT_RATIOS = [0.01, 0.05, 0.1, 0.2, 0.3]
_ALL_METHODS    = ["wanda", "spectral", "spectral+wanda"]


def _parse_ratios(ratios_str: str) -> list[float]:
    """Parse a comma-separated ratio string → list of floats.

    Accepts fractions (0.01,0.05,0.1) or integer percentages (1,5,10,20,30).
    Values > 1 are divided by 100 automatically.
    """
    parts = [r.strip() for r in ratios_str.split(",") if r.strip()]
    result = []
    for p in parts:
        v = float(p)
        if v > 1.0:
            v /= 100.0
        result.append(v)
    return result


def _parse_methods(methods_str: str) -> list[str]:
    """Parse a comma-separated method string.  'all' expands to all three methods."""
    if methods_str.strip().lower() == "all":
        return list(_ALL_METHODS)
    return [m.strip() for m in methods_str.split(",") if m.strip()]


@app.local_entrypoint()
def main(
    command: str = "prune",
    model: str = "Qwen/Qwen3-8B",
    ratio: float = 0.0,
    ratios: str = "",
    methods: str = "wanda",
    seq_len: int = 2048,
    batch_size: int = 1,
    calib_samples: int = 64,
    n_sv: int = 256,
    wandb_project: str = "CMU-16542-MLSys-Compression",
    regenerate: bool = False,
):
    """
    Commands:
        prune    -- Prune at specified ratios and methods
        profile  -- Perplexity + latency sweep for specified ratios and methods
        ncu      -- GPU kernel profiling for specified ratios and methods
        all      -- prune + profile + ncu  (full pipeline)

    Key flags:
        --ratios     Comma-separated pruning ratios.
                     Fractions (0.01,0.05,0.1,0.2,0.3) or percentages (1,5,10,20,30).
                     Default: 1,5,10,20,30%.
        --methods    Comma-separated methods: wanda | spectral | spectral+wanda | all
                     Default: wanda.
                     'all' runs wanda, spectral, and spectral+wanda sequentially.
        --regenerate Force re-pruning even if models already exist.

    Full experiment (all methods × all ratios + NCU):
        modal run modal_pruning.py --command all --methods all --regenerate

    Single method:
        modal run modal_pruning.py --command all --methods wanda --regenerate
        modal run modal_pruning.py --command all --methods spectral --regenerate
        modal run modal_pruning.py --command all --methods spectral+wanda --regenerate

    Custom ratios:
        modal run modal_pruning.py --command all --methods all --ratios "1,5,10,20,30"

    NCU only (for an already-pruned model):
        modal run modal_pruning.py --command ncu --methods wanda --ratio 0.1
    """
    parsed_ratios  = _parse_ratios(ratios) if ratios else _DEFAULT_RATIOS
    parsed_methods = _parse_methods(methods)
    wb = wandb_project or None

    pct_list = [int(r * 100) for r in parsed_ratios]
    print(f"  ratios={pct_list}%   methods={parsed_methods}")

    # ------------------------------------------------------------------
    if command == "prune":
        for meth in parsed_methods:
            print(f"\n--- Pruning  method={meth} ---")
            result = prune_all_ratios.remote(
                model_name=model, ratios=parsed_ratios,
                method=meth, n_calib_samples=calib_samples, n_sv=n_sv,
                wandb_project=wb, regenerate=regenerate)
            for tag, info in result.items():
                print(f"  {tag}: {info['status']}  ({info['path']})")

    # ------------------------------------------------------------------
    elif command == "profile":
        for meth in parsed_methods:
            print(f"\n--- Profiling  method={meth} ---")
            slug = _method_slug(meth)
            csv_bytes = profile_pruned_sweep.remote(
                model_name=model, ratios=parsed_ratios,
                method=meth, wandb_project=wb)
            out = Path(f"results_{slug}.csv")
            out.write_bytes(csv_bytes)
            print(f"  Downloaded {out}")

    # ------------------------------------------------------------------
    elif command == "ncu":
        ncu_jobs = []
        for meth in parsed_methods:
            slug = _method_slug(meth)
            if ratio > 0:
                # single ratio requested via --ratio
                label = f"{slug}-pruned-{int(ratio * 100)}pct"
                ncu_jobs.append((meth, ratio, label))
            else:
                # profile baseline + all requested ratios
                ncu_jobs.append((meth, 0.0, "baseline"))
                for r in parsed_ratios:
                    ncu_jobs.append((meth, r, f"{slug}-pruned-{int(r*100)}pct"))

        # Fan out NCU runs in parallel (each is a separate Modal container)
        futures = [
            profile_ncu.spawn(
                model_path=model if r == 0.0 else "",
                seq_len=seq_len, batch_size=batch_size,
                label=lbl, ratio=r, method=meth,
                wandb_project=wb,
            )
            for meth, r, lbl in ncu_jobs
        ]
        for (meth, r, lbl), fut in zip(ncu_jobs, futures):
            report = fut.get()
            slug   = _method_slug(meth) if meth else "baseline"
            fname  = f"ncu_{slug}_{lbl}_N{seq_len}_B{batch_size}.txt"
            Path(fname).write_text(report)
            print(f"  NCU report -> {fname}")

    # ------------------------------------------------------------------
    elif command == "all":
        for meth in parsed_methods:
            slug = _method_slug(meth)
            print(f"\n{'='*60}")
            print(f"METHOD: {meth}  ratios={pct_list}%")
            print(f"{'='*60}")

            # Step 1: Prune
            print(f"\nStep 1/3: Pruning ...")
            result = prune_all_ratios.remote(
                model_name=model, ratios=parsed_ratios,
                method=meth, n_calib_samples=calib_samples, n_sv=n_sv,
                wandb_project=wb, regenerate=regenerate)
            for tag, info in result.items():
                print(f"  {tag}: {info['status']}  ({info['path']})")

            # Step 2: Perplexity + latency profiling
            print(f"\nStep 2/3: Profiling (perplexity + latency) ...")
            csv_bytes = profile_pruned_sweep.remote(
                model_name=model, ratios=parsed_ratios,
                method=meth, wandb_project=wb)
            out = Path(f"results_{slug}.csv")
            out.write_bytes(csv_bytes)
            print(f"  Downloaded {out}")

            # Step 3: NCU — baseline + all pruned ratios, launched in parallel
            print(f"\nStep 3/3: NCU profiling ...")
            ncu_jobs = [("baseline", 0.0)] + [
                (f"{slug}-pruned-{int(r*100)}pct", r) for r in parsed_ratios
            ]
            futures = [
                profile_ncu.spawn(
                    model_path=model if r == 0.0 else "",
                    seq_len=seq_len, batch_size=batch_size,
                    label=lbl, ratio=r, method=meth,
                    wandb_project=wb,
                )
                for lbl, r in ncu_jobs
            ]
            for (lbl, _r), fut in zip(ncu_jobs, futures):
                report = fut.get()
                fname  = f"ncu_{slug}_{lbl}_N{seq_len}_B{batch_size}.txt"
                Path(fname).write_text(report)
                print(f"  NCU report -> {fname}")

        print(f"\nAll done. Methods processed: {parsed_methods}")

    else:
        raise ValueError(f"Unknown command: {command!r}. "
                         "Choose prune | profile | ncu | all.")
