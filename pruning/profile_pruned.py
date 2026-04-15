"""
Inference profiling for Experiment 3: Structural Pruning.

Measures layer-wise timing (attention vs FFN), throughput, peak VRAM,
and perplexity for the FP16 baseline and each pruned variant.

All metrics are logged to Weights & Biases when a project is provided,
in addition to CSV output.

Uses CUDA-event hooks identical to weight_quant/profile_layers.py.

Usage (standalone, requires GPU):
    python profile_pruned.py --baseline Qwen/Qwen3-8B \
        --pruned-10 /models/pruned_10pct \
        --pruned-20 /models/pruned_20pct \
        --pruned-30 /models/pruned_30pct \
        --output results_exp3.csv \
        --wandb-project ml-hw-profiling
"""

import argparse
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class ProfileConfig:
    batch_size: int = 1
    seq_len: int = 512
    prune_ratio: str = "baseline"   # "baseline", "10", "20", "30"
    n_warmup: int = 3
    n_trials: int = 5


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(model_path: str, torch_dtype=torch.float16):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch_dtype, device_map="auto",
        rope_scaling={"type": "dynamic", "factor": 3.0}
    )
    model.eval()
    return model, tokenizer


# ------------------------------------------------------------------
# Hook-based layer timing
# ------------------------------------------------------------------

class _TimingHook:
    """Pre/post forward hooks that record elapsed CUDA time."""
    def __init__(self):
        self.times: list[float] = []
        self._start = None

    def pre(self, module, inp):
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def post(self, module, inp, out):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        self.times.append(self._start.elapsed_time(end))


def profile_model(model, tokenizer, cfg: ProfileConfig) -> list[dict]:
    """Return per-layer-type timing rows for a single (batch, seq_len)."""
    device = next(model.parameters()).device
    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (cfg.batch_size, cfg.seq_len), device=device,
    )

    print(f"  Warming up ({cfg.n_warmup} iters)...", end="", flush=True)
    with torch.no_grad():
        for _ in range(cfg.n_warmup):
            model(input_ids)
    torch.cuda.synchronize()
    print(" done")

    handles = []
    attn_hooks: list[_TimingHook] = []
    ffn_hooks: list[_TimingHook] = []

    for name, mod in model.named_modules():
        if name.endswith(".self_attn"):
            th = _TimingHook()
            attn_hooks.append(th)
            handles.append(mod.register_forward_pre_hook(th.pre))
            handles.append(mod.register_forward_hook(th.post))
        elif name.endswith(".mlp"):
            th = _TimingHook()
            ffn_hooks.append(th)
            handles.append(mod.register_forward_pre_hook(th.pre))
            handles.append(mod.register_forward_hook(th.post))

    total_times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    print(f"  Profiling ({cfg.n_trials} iters)...", end="", flush=True)
    with torch.no_grad():
        for _ in range(cfg.n_trials):
            start_ev.record()
            model(input_ids)
            end_ev.record()
            torch.cuda.synchronize()
            total_times.append(start_ev.elapsed_time(end_ev))
    print(" done")

    for h in handles:
        h.remove()

    def _per_trial_sum(hooks_list):
        per_trial = [0.0] * cfg.n_trials
        for th in hooks_list:
            for i, t in enumerate(th.times):
                per_trial[i] += t
        return per_trial

    attn_ms = float(np.median(_per_trial_sum(attn_hooks)))
    ffn_ms = float(np.median(_per_trial_sum(ffn_hooks)))
    total_ms = float(np.median(total_times))
    other_ms = max(0.0, total_ms - attn_ms - ffn_ms)

    def _pct(v):
        return round(v / total_ms * 100, 2) if total_ms > 0 else 0.0

    print("  Timing breakdown (median over trials, per forward pass):")
    print(f"    Total forward latency: {total_ms:.2f} ms")
    print(f"    Attention modules:     {attn_ms:.2f} ms ({_pct(attn_ms)}% of total)")
    print(f"    FFN modules:           {ffn_ms:.2f} ms ({_pct(ffn_ms)}% of total)")
    print(f"    Other runtime:         {other_ms:.2f} ms ({_pct(other_ms)}% of total)")

    base = dict(prune_ratio=cfg.prune_ratio, batch=cfg.batch_size,
                seq_len=cfg.seq_len)
    return [
        {**base, "layer_type": "attention", "time_ms": attn_ms,
         "time_pct": _pct(attn_ms)},
        {**base, "layer_type": "ffn", "time_ms": ffn_ms,
         "time_pct": _pct(ffn_ms)},
        {**base, "layer_type": "other", "time_ms": other_ms,
         "time_pct": _pct(other_ms)},
        {**base, "layer_type": "total", "time_ms": total_ms,
         "time_pct": 100.0},
    ]


# ------------------------------------------------------------------
# Throughput
# ------------------------------------------------------------------

def measure_throughput(model, tokenizer, cfg: ProfileConfig) -> float:
    """Measure prefill throughput in tokens/sec."""
    device = next(model.parameters()).device
    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (cfg.batch_size, cfg.seq_len), device=device,
    )
    total_tokens = cfg.batch_size * cfg.seq_len

    with torch.no_grad():
        for _ in range(cfg.n_warmup):
            model(input_ids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.no_grad():
        for _ in range(cfg.n_trials):
            start.record()
            model(input_ids)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

    median_ms = float(np.median(latencies))
    tok_per_sec = total_tokens / (median_ms / 1000.0)
    return tok_per_sec


# ------------------------------------------------------------------
# Peak VRAM
# ------------------------------------------------------------------

def measure_peak_vram(model, tokenizer, cfg: ProfileConfig) -> float:
    """Run a forward pass and return peak VRAM in MB."""
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)

    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (cfg.batch_size, cfg.seq_len), device=device,
    )
    with torch.no_grad():
        model(input_ids)
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 2)


# ------------------------------------------------------------------
# Perplexity
# ------------------------------------------------------------------

def evaluate_perplexity(model, tokenizer, max_samples=50, seq_len=512) -> float:
    print("  Evaluating perplexity on WikiText-103 test set ...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")

    device = next(model.parameters()).device
    input_ids = enc.input_ids.to(device)
    nlls = []

    with torch.no_grad():
        for i in range(0, min(len(input_ids[0]), max_samples * seq_len), seq_len):
            chunk = input_ids[:, i:i + seq_len]
            if chunk.size(1) < seq_len:
                continue
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item())
            if len(nlls) >= max_samples:
                break

    ppl = float(np.exp(np.mean(nlls)))
    print(f"  Perplexity: {ppl:.2f} (n={len(nlls)} chunks)")
    return ppl


# ------------------------------------------------------------------
# W&B logging
# ------------------------------------------------------------------

def _wandb_log_config_metrics(rows: list[dict], throughput: float,
                              peak_vram: float, cfg: ProfileConfig):
    """Log metrics for a single (prune_ratio, batch, seq_len) config."""
    if not HAS_WANDB or wandb.run is None:
        return

    prefix = f"profile/{cfg.prune_ratio}"
    total_row = next((r for r in rows if r["layer_type"] == "total"), None)
    attn_row = next((r for r in rows if r["layer_type"] == "attention"), None)
    ffn_row = next((r for r in rows if r["layer_type"] == "ffn"), None)

    log_dict = {
        f"{prefix}/batch": cfg.batch_size,
        f"{prefix}/seq_len": cfg.seq_len,
        f"{prefix}/throughput_tok_s": throughput,
        f"{prefix}/peak_vram_mb": peak_vram,
    }
    if total_row:
        log_dict[f"{prefix}/total_ms"] = total_row["time_ms"]
    if attn_row:
        log_dict[f"{prefix}/attn_ms"] = attn_row["time_ms"]
        log_dict[f"{prefix}/attn_pct"] = attn_row["time_pct"]
    if ffn_row:
        log_dict[f"{prefix}/ffn_ms"] = ffn_row["time_ms"]
        log_dict[f"{prefix}/ffn_pct"] = ffn_row["time_pct"]
    wandb.log(log_dict)


def _wandb_log_sweep_summary(df: pd.DataFrame, ppl_results: dict):
    """Log final summary tables to W&B."""
    if not HAS_WANDB or wandb.run is None:
        return

    wandb.log({"results/full_table": wandb.Table(dataframe=df)})

    ppl_table = wandb.Table(columns=["prune_ratio", "perplexity"])
    for label, ppl in ppl_results.items():
        ppl_table.add_data(label, ppl)
    wandb.log({"results/perplexity": ppl_table})

    # Latency comparison bar chart
    df_total = df[df["layer_type"] == "total"].copy()
    if not df_total.empty:
        wandb.log({
            "results/latency_table": wandb.Table(
                dataframe=df_total[["prune_ratio", "batch", "seq_len",
                                    "time_ms", "throughput_tok_s",
                                    "peak_vram_mb"]]
            )
        })

    # Bottleneck shift table
    df_layers = df[df["layer_type"].isin(["attention", "ffn"])].copy()
    if not df_layers.empty:
        wandb.log({
            "results/bottleneck_shift": wandb.Table(
                dataframe=df_layers[["prune_ratio", "batch", "seq_len",
                                     "layer_type", "time_ms", "time_pct"]]
            )
        })


# ------------------------------------------------------------------
# Full sweep
# ------------------------------------------------------------------

def profile_sweep(
    baseline_path: str,
    pruned_paths: dict[str, str],
    output_csv: str,
    wandb_project: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Profile baseline + each pruned model across the full grid.

    Args:
        baseline_path:  HuggingFace name or local path for FP16 baseline.
        pruned_paths:   {"10": "/path/pruned_10pct", "20": ..., "30": ...}
        output_csv:     Where to write combined results.
        wandb_project:  If set, initialises a W&B run for logging.

    Returns:
        (DataFrame, perplexity_dict)
    """
    import os
    wb_project = wandb_project or os.environ.get("WANDB_PROJECT")
    if HAS_WANDB and wb_project and wandb.run is None:
        wandb.init(
            project=wb_project,
            name="profile-pruning-sweep",
            config={
                "baseline": baseline_path,
                "pruned_ratios": list(pruned_paths.keys()),
                "batch_sizes": [1, 4, 16],
                "seq_lens": [512, 2048, 4096],
            },
        )

    batch_sizes = [1, 4, 16]
    seq_lens = [512, 2048, 4096]
    all_rows: list[dict] = []
    ppl_results: dict[str, float] = {}

    configs = [("baseline", baseline_path)] + [
        (k, v) for k, v in sorted(pruned_paths.items())
    ]

    for label, path in configs:
        print(f"\n{'='*60}")
        print(f"PROFILING  prune_ratio={label}  path={path}")
        print(f"{'='*60}")

        model, tok = load_model(path)
        ppl = evaluate_perplexity(model, tok)
        ppl_results[label] = ppl

        if HAS_WANDB and wandb.run is not None:
            wandb.log({f"perplexity/{label}": ppl})

        for batch in batch_sizes:
            for seq_len in seq_lens:
                total_tokens = batch * seq_len
                print(f"\n  Config: batch={batch}, seq_len={seq_len} "
                      f"(tokens/forward={total_tokens})")
                cfg = ProfileConfig(batch_size=batch, seq_len=seq_len,
                                    prune_ratio=label)
                try:
                    rows = profile_model(model, tok, cfg)

                    throughput = measure_throughput(model, tok, cfg)
                    peak_vram = measure_peak_vram(model, tok, cfg)

                    for r in rows:
                        r["throughput_tok_s"] = throughput
                        r["peak_vram_mb"] = round(peak_vram, 1)

                    all_rows.extend(rows)
                    _wandb_log_config_metrics(rows, throughput, peak_vram, cfg)
                except Exception as e:
                    print(f"  FAILED: {e}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    print(f"\n{'='*60}")
    print("PERPLEXITY SUMMARY")
    print(f"{'='*60}")
    for label, ppl in ppl_results.items():
        print(f"  {label}: {ppl:.2f}")

    _wandb_log_sweep_summary(df, ppl_results)

    if HAS_WANDB and wandb.run is not None:
        wandb.save(output_csv)

    return df, ppl_results


# ------------------------------------------------------------------
# CLI (for standalone / debug use)
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile pruned models (Experiment 3)")
    parser.add_argument("--baseline", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--pruned-10", type=str, default=None)
    parser.add_argument("--pruned-20", type=str, default=None)
    parser.add_argument("--pruned-30", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_exp3.csv")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (enables logging)")
    args = parser.parse_args()

    pruned = {}
    if args.pruned_10:
        pruned["10"] = args.pruned_10
    if args.pruned_20:
        pruned["20"] = args.pruned_20
    if args.pruned_30:
        pruned["30"] = args.pruned_30

    df, ppl = profile_sweep(args.baseline, pruned, args.output,
                            wandb_project=args.wandb_project)

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
