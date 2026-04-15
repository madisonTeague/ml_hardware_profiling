"""
Analysis and visualization for Experiment 3: Structural Pruning.

Reads results_exp3.csv (and optionally gemm_shapes_*.csv / perplexity CSV)
and uploads all charts + tables to Weights & Biases.

Usage:
    python analyze_pruning.py --input results_exp3.csv --wandb-project ml-hw-profiling
    python analyze_pruning.py --input results_exp3.csv --gemm-dir . --perplexity perplexity_exp3.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import wandb


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Prune ratios: {sorted(df['prune_ratio'].unique())}")
    print(f"  Batch sizes:  {sorted(df['batch'].unique())}")
    print(f"  Seq lens:     {sorted(df['seq_len'].unique())}")
    return df


def load_gemm_shapes(gemm_dir: str) -> pd.DataFrame | None:
    """Concatenate all gemm_shapes_*pct.csv files in a directory."""
    dfs = []
    for p in sorted(Path(gemm_dir).glob("gemm_shapes_*pct.csv")):
        dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded GEMM shapes: {len(df)} rows")
    return df


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ratio_label(r) -> str:
    return "Baseline" if str(r) == "baseline" else f"Pruned {r}%"


def _sorted_ratios(df):
    return sorted(df["prune_ratio"].unique(),
                  key=lambda x: (str(x) != "baseline", str(x)))


# ------------------------------------------------------------------
# Chart 1: Latency breakdown (attention vs FFN share)
# ------------------------------------------------------------------

def log_latency_breakdown(df: pd.DataFrame):
    df_layers = df[df["layer_type"].isin(["attention", "ffn"])].copy()
    df_layers["label"] = df_layers.apply(
        lambda r: f"{_ratio_label(r['prune_ratio'])} {r['layer_type']}", axis=1)
    df_layers["config"] = df_layers.apply(
        lambda r: f"B={r['batch']},N={r['seq_len']}", axis=1)

    table = wandb.Table(dataframe=df_layers[[
        "prune_ratio", "batch", "seq_len", "layer_type",
        "time_ms", "time_pct", "config", "label",
    ]])
    wandb.log({
        "charts/latency_breakdown_table": table,
        "charts/latency_breakdown": wandb.plot.bar(
            table, "config", "time_pct",
            title="Attention vs FFN Runtime Share (%)",
        ),
    })


# ------------------------------------------------------------------
# Chart 2: Total latency comparison
# ------------------------------------------------------------------

def log_latency(df: pd.DataFrame):
    df_total = df[df["layer_type"] == "total"].copy()
    ratios = _sorted_ratios(df_total)

    df_total["config"] = df_total.apply(
        lambda r: f"B={r['batch']},N={r['seq_len']}", axis=1)
    df_total["ratio_label"] = df_total["prune_ratio"].apply(_ratio_label)

    table = wandb.Table(dataframe=df_total[[
        "config", "ratio_label", "time_ms", "prune_ratio",
        "batch", "seq_len",
    ]])
    wandb.log({"charts/latency_table": table})

    if "baseline" in df_total["prune_ratio"].values:
        baseline = df_total[df_total["prune_ratio"] == "baseline"].set_index(
            ["batch", "seq_len"])["time_ms"]
        speedup_rows = []
        for _, row in df_total.iterrows():
            if row["prune_ratio"] == "baseline":
                continue
            key = (row["batch"], row["seq_len"])
            if key in baseline.index:
                base_ms = baseline.loc[key]
                speedup_rows.append({
                    "config": row["config"],
                    "prune_ratio": _ratio_label(row["prune_ratio"]),
                    "speedup": base_ms / row["time_ms"],
                    "latency_change_pct": (row["time_ms"] / base_ms - 1) * 100,
                })
        if speedup_rows:
            sp_table = wandb.Table(dataframe=pd.DataFrame(speedup_rows))
            wandb.log({"charts/speedup_table": sp_table})


# ------------------------------------------------------------------
# Chart 3: Throughput (tokens/sec)
# ------------------------------------------------------------------

def log_throughput(df: pd.DataFrame):
    df_total = df[df["layer_type"] == "total"].copy()
    if "throughput_tok_s" not in df_total.columns:
        return

    df_total["config"] = df_total.apply(
        lambda r: f"B={r['batch']},N={r['seq_len']}", axis=1)
    df_total["ratio_label"] = df_total["prune_ratio"].apply(_ratio_label)

    table = wandb.Table(dataframe=df_total[[
        "config", "ratio_label", "throughput_tok_s",
    ]])
    wandb.log({
        "charts/throughput_table": table,
        "charts/throughput": wandb.plot.bar(
            table, "config", "throughput_tok_s",
            title="Prefill Throughput (tokens/sec)",
        ),
    })


# ------------------------------------------------------------------
# Chart 4: Peak VRAM
# ------------------------------------------------------------------

def log_memory(df: pd.DataFrame):
    df_total = df[df["layer_type"] == "total"].copy()
    if "peak_vram_mb" not in df_total.columns:
        return

    df_total["config"] = df_total.apply(
        lambda r: f"B={r['batch']},N={r['seq_len']}", axis=1)
    df_total["ratio_label"] = df_total["prune_ratio"].apply(_ratio_label)
    df_total["peak_vram_gb"] = df_total["peak_vram_mb"] / 1024

    table = wandb.Table(dataframe=df_total[[
        "config", "ratio_label", "peak_vram_gb",
    ]])
    wandb.log({
        "charts/memory_table": table,
        "charts/peak_vram": wandb.plot.bar(
            table, "config", "peak_vram_gb",
            title="Peak GPU Memory (GB)",
        ),
    })


# ------------------------------------------------------------------
# Chart 5: GEMM shape changes
# ------------------------------------------------------------------

def log_gemm_shapes(gemm_df: pd.DataFrame):
    gemm_df = gemm_df.copy()
    gemm_df["compression_pct"] = (
        (1 - gemm_df["pruned_elements"] / gemm_df["original_elements"]) * 100
    ).round(2)

    table = wandb.Table(dataframe=gemm_df[[
        "layer", "projection", "prune_ratio",
        "original_shape", "pruned_shape",
        "original_elements", "pruned_elements", "compression_pct",
    ]])
    wandb.log({"charts/gemm_shapes_table": table})

    gate = gemm_df[gemm_df["projection"] == "gate_proj"]
    for ratio in sorted(gate["prune_ratio"].unique()):
        sub = gate[gate["prune_ratio"] == ratio]
        data = [[l, e / 1e6] for l, e in
                zip(sub["layer"], sub["pruned_elements"])]
        chart_table = wandb.Table(data=data,
                                  columns=["layer", "params_M"])
        wandb.log({
            f"charts/gemm_gate_pruned_{int(ratio*100)}pct": wandb.plot.line(
                chart_table, "layer", "params_M",
                title=f"gate_proj Params/Layer ({int(ratio*100)}% pruned)",
            )
        })


# ------------------------------------------------------------------
# Chart 6: Perplexity vs pruning ratio
# ------------------------------------------------------------------

def log_perplexity(ppl_csv: str):
    df = pd.read_csv(ppl_csv)
    table = wandb.Table(dataframe=df)
    wandb.log({
        "charts/perplexity_table": table,
        "charts/perplexity": wandb.plot.bar(
            table, "prune_ratio", "perplexity",
            title="Perplexity vs Pruning Ratio (WikiText-103)",
        ),
    })


# ------------------------------------------------------------------
# Summary tables (console)
# ------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("LAYER-WISE RUNTIME BREAKDOWN")
    print(f"{'='*70}")

    df_layers = df[df["layer_type"].isin(["attention", "ffn"])].copy()
    pivot = df_layers.pivot_table(
        index=["batch", "seq_len", "layer_type"],
        columns="prune_ratio",
        values=["time_ms", "time_pct"],
    )
    print(pivot.to_string())

    print(f"\n{'='*70}")
    print("TOTAL LATENCY & SPEEDUP vs BASELINE")
    print(f"{'='*70}")
    df_total = df[df["layer_type"] == "total"].copy()
    lat = df_total.pivot_table(index=["batch", "seq_len"],
                               columns="prune_ratio", values="time_ms")
    if "baseline" in lat.columns:
        for r in lat.columns:
            if r != "baseline":
                lat[f"speedup_{r}"] = lat["baseline"] / lat[r]
    print(lat.to_string())

    if "throughput_tok_s" in df_total.columns:
        print(f"\n{'='*70}")
        print("THROUGHPUT (tokens/sec)")
        print(f"{'='*70}")
        thr = df_total.pivot_table(index=["batch", "seq_len"],
                                   columns="prune_ratio",
                                   values="throughput_tok_s")
        print(thr.to_string())

    if "peak_vram_mb" in df_total.columns:
        print(f"\n{'='*70}")
        print("PEAK VRAM (MB)")
        print(f"{'='*70}")
        mem = df_total.pivot_table(index=["batch", "seq_len"],
                                   columns="prune_ratio",
                                   values="peak_vram_mb")
        print(mem.to_string())

    print(f"\n{'='*70}")
    print("FFN SHARE SHIFT (attention % point change vs baseline)")
    print(f"{'='*70}")
    for (b, n), g in df_layers.groupby(["batch", "seq_len"]):
        base_a = g[(g["prune_ratio"] == "baseline") &
                   (g["layer_type"] == "attention")]["time_pct"].values
        if len(base_a) == 0:
            continue
        for r in sorted(g["prune_ratio"].unique()):
            if str(r) == "baseline":
                continue
            pruned_a = g[(g["prune_ratio"] == r) &
                         (g["layer_type"] == "attention")]["time_pct"].values
            if len(pruned_a):
                shift = pruned_a[0] - base_a[0]
                print(f"  B={b}, N={n:4d}: baseline attn {base_a[0]:.1f}% "
                      f"-> {r}% pruned attn {pruned_a[0]:.1f}% "
                      f"({shift:+.1f} pp)")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Experiment 3 results (logs to W&B)")
    parser.add_argument("--input", default="results_exp3.csv",
                        help="Profiling results CSV")
    parser.add_argument("--gemm-dir", default=None,
                        help="Directory containing gemm_shapes_*pct.csv files")
    parser.add_argument("--perplexity", default=None,
                        help="Perplexity CSV (perplexity_exp3.csv)")
    parser.add_argument("--wandb-project", default="ml-hw-profiling",
                        help="W&B project name")
    parser.add_argument("--wandb-name", default="analyze-pruning-exp3",
                        help="W&B run name")
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="analysis",
    )

    df = load_results(args.input)

    wandb.log({"raw/results_csv": wandb.Table(dataframe=df)})
    wandb.save(args.input)

    log_latency_breakdown(df)
    log_latency(df)
    log_throughput(df)
    log_memory(df)
    print_summary(df)

    if args.gemm_dir:
        gemm_df = load_gemm_shapes(args.gemm_dir)
        if gemm_df is not None:
            log_gemm_shapes(gemm_df)

    if args.perplexity:
        log_perplexity(args.perplexity)
        wandb.save(args.perplexity)

    wandb.finish()
    print("\nAll charts uploaded to W&B.")


if __name__ == "__main__":
    main()
