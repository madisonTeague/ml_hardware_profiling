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

def _tag_for_ratio(ratio: float) -> str:
    """Base tag without timestamp, e.g. 'qwen3-8b-pruned-10pct'."""
    return f"qwen3-8b-pruned-{int(ratio * 100)}pct"


def _make_timestamped_dir(base_dir: str, ratio: float) -> str:
    """Create a new timestamped model directory path (does NOT mkdir)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{base_dir}/{_tag_for_ratio(ratio)}_{ts}"


def _find_latest_model(base_dir: str, ratio: float) -> str | None:
    """Find the most recent timestamped model dir for a given ratio.

    Directories are named like  qwen3-8b-pruned-10pct_20260414T183012.
    Also recognises the legacy name without a timestamp suffix.
    Returns None if nothing is found.
    """
    root = Path(base_dir)
    prefix = _tag_for_ratio(ratio)

    candidates: list[Path] = []
    if not root.exists():
        return None

    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name == prefix or name.startswith(prefix + "_"):
            if any(d.iterdir()):
                candidates.append(d)

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
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "nsight-compute 2>/dev/null || true",
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
        tag = _tag_for_ratio(ratio)

        if not regenerate:
            existing = _find_latest_model("/models", ratio)
            if existing:
                print(f"  {Path(existing).name}: already exists, reusing.")
                results[tag] = {"status": "already_exists", "path": existing}
                continue

        out_dir = _make_timestamped_dir("/models", ratio)

        model, tokenizer, gemm_rows = prune_model(
            model_name, ratio, out_dir,
            torch_dtype=torch.float16,
            method=method,
            n_calib_samples=n_calib_samples,
            n_sv=n_sv,
            wandb_project=wb_project,
        )

        import shutil
        gemm_src = Path(out_dir) / "gemm_shapes.csv"
        if gemm_src.exists():
            shutil.copy(gemm_src, f"/results/gemm_shapes_{int(ratio*100)}pct.csv")

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
    output_name: str = "results_exp3.csv",
    wandb_project: str | None = None,
) -> bytes:
    import sys; sys.path.insert(0, "/app")
    import os, torch, pandas as pd
    from profile_pruned import profile_sweep

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    pruned_paths = {}
    for r in ratios:
        latest = _find_latest_model("/models", r)
        if latest is None:
            print(f"WARNING: no model found for ratio {r}, skipping")
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
    ppl_path = "/results/perplexity_exp3.csv"
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


@app.function(
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/models": models_vol, "/results": results_vol},
)
def profile_ncu(
    model_path: str = "Qwen/Qwen3-8B",
    seq_len: int = 2048,
    batch_size: int = 1,
    label: str = "baseline",
    ratio: float = 0.0,
) -> str:
    """NCU profiling of a forward pass, focusing on FFN GEMM kernels."""
    import os, subprocess, shutil

    if ratio > 0:
        resolved = _find_latest_model("/models", ratio)
        if resolved:
            model_path = resolved
            print(f"  NCU: resolved to {Path(resolved).name}")

    out_name = f"ncu_pruning_{label}_N{seq_len}_B{batch_size}.txt"
    txt_path = f"/tmp/{out_name}"
    rep_path = txt_path.replace(".txt", ".ncu-rep")

    ncu_bin = shutil.which("ncu")
    if ncu_bin is None:
        for candidate in ["/usr/local/cuda/bin/ncu", "/usr/bin/ncu"]:
            if os.path.exists(candidate):
                ncu_bin = candidate
                break

    if ncu_bin is None:
        return _torch_profiler_fallback(model_path, seq_len, batch_size)

    try:
        with open("/proc/sys/kernel/perf_event_paranoid", "w") as f:
            f.write("-1")
    except Exception:
        pass

    env = os.environ.copy()
    env["MODEL_PATH"] = model_path
    env["BATCH_SIZE"] = str(batch_size)
    env["SEQ_LEN"] = str(seq_len)

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
        "python", "/app/ncu_driver.py",
    ]

    print(f"Running: {' '.join(ncu_cmd)}")
    result = subprocess.run(ncu_cmd, capture_output=True, text=True,
                            cwd="/app", env=env)
    print(f"NCU returncode: {result.returncode}")
    if result.stderr:
        print(f"NCU stderr (last 1000): {result.stderr[-1000:]}")

    stdout = result.stdout
    if result.returncode != 0 and not stdout.strip():
        return _torch_profiler_fallback(model_path, seq_len, batch_size)

    vol_path = f"/results/{out_name}"
    with open(vol_path, "w") as f:
        f.write(stdout)
    results_vol.commit()
    print(f"NCU report saved to {vol_path}")

    return stdout


def _torch_profiler_fallback(model_path, seq_len, batch_size):
    """Fallback when NCU is unavailable."""
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
_DEFAULT_RATIOS = [0.05, 0.1, 0.2, 0.3]


def _parse_ratios(ratios_str: str) -> list[float]:
    """Parse a comma-separated ratio string into a list of floats.

    Accepts fractions (0.05,0.1,0.2) or integer percentages (5,10,20,30).
    Integer values > 1 are automatically divided by 100.
    """
    parts = [r.strip() for r in ratios_str.split(",") if r.strip()]
    result = []
    for p in parts:
        v = float(p)
        if v > 1.0:          # treat as percentage, e.g. "5" -> 0.05
            v = v / 100.0
        result.append(v)
    return result


@app.local_entrypoint()
def main(
    command: str = "prune",
    model: str = "Qwen/Qwen3-8B",
    ratio: float = 0.0,
    ratios: str = "",
    seq_len: int = 2048,
    batch_size: int = 1,
    method: str = "wanda",
    calib_samples: int = 64,
    n_sv: int = 256,
    wandb_project: str = "",
    regenerate: bool = False,
):
    """
    Commands:
        prune    -- Prune model at default ratios (5/10/20/30%)
        profile  -- Full profiling sweep (baseline + all pruned)
        ncu      -- NCU profiling (specify --ratio, --seq-len, --batch-size)
        all      -- prune + profile

    Flags:
        --ratios         Comma-separated list of ratios to prune/profile.
                         Accepts fractions (0.05,0.1,0.2,0.3) or integer
                         percentages (5,10,20,30).  Defaults to 5/10/20/30%.
        --method         Importance scoring: wanda | spectral | spectral+wanda
                         (default: wanda).
        --calib-samples  WikiText-103 chunks for activation calibration
                         (default 64; set 0 for weight-only; ignored for spectral).
        --n-sv           Singular values for truncated SVD in spectral paths
                         (default 256).
        --regenerate     Force re-pruning even if models already exist.
                         New models are saved with a timestamp suffix;
                         profiling always picks the latest version.

    Examples:
        modal run modal_pruning.py --command all --regenerate
        modal run modal_pruning.py --command all --ratios "5,10,20,30" --method spectral
        modal run modal_pruning.py --command prune --ratios "0.05,0.15,0.25" --method spectral+wanda
        modal run modal_pruning.py --command ncu --ratio 0.1
    """
    parsed_ratios = _parse_ratios(ratios) if ratios else _DEFAULT_RATIOS
    wb = wandb_project or None

    if command == "prune":
        result = prune_all_ratios.remote(
            model_name=model, ratios=parsed_ratios,
            method=method, n_calib_samples=calib_samples, n_sv=n_sv,
            wandb_project=wb, regenerate=regenerate)
        for tag, info in result.items():
            print(f"  {tag}: {info['status']}  ({info['path']})")

    elif command == "profile":
        csv_bytes = profile_pruned_sweep.remote(
            model_name=model, ratios=parsed_ratios, wandb_project=wb)
        out = Path("results_exp3.csv")
        out.write_bytes(csv_bytes)
        print(f"Downloaded {out}")

    elif command == "ncu":
        if ratio > 0:
            label = f"pruned_{int(ratio * 100)}pct"
            model_path = f"/models/{_tag_for_ratio(ratio)}"
        else:
            model_path = model
            label = "baseline"

        report = profile_ncu.remote(
            model_path=model_path, seq_len=seq_len,
            batch_size=batch_size, label=label, ratio=ratio,
        )
        fname = f"ncu_pruning_{label}_N{seq_len}_B{batch_size}.txt"
        Path(fname).write_text(report)
        print(f"NCU report -> {fname}")
        print(report[:3000])

    elif command == "all":
        print(f"Step 1/2: Pruning  ratios={[int(r*100) for r in parsed_ratios]}%  "
              f"method={method} ...")
        result = prune_all_ratios.remote(
            model_name=model, ratios=parsed_ratios,
            method=method, n_calib_samples=calib_samples, n_sv=n_sv,
            wandb_project=wb, regenerate=regenerate)
        for tag, info in result.items():
            print(f"  {tag}: {info['status']}  ({info['path']})")

        print("\nStep 2/2: Profiling ...")
        csv_bytes = profile_pruned_sweep.remote(
            model_name=model, ratios=parsed_ratios, wandb_project=wb)
        out = Path("results_exp3.csv")
        out.write_bytes(csv_bytes)
        print(f"Downloaded {out}")

    else:
        raise ValueError(f"Unknown command: {command!r}. "
                         "Choose prune | profile | ncu | all.")
