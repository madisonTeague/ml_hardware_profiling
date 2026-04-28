"""
Analysis and visualization for Experiment 2: Weight Quantization.

Reads results_exp2_layers.csv and generates figures + tables.

Usage:
    python analyze_weight_quant.py --input results_exp2_layers.csv
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Dtypes present: {sorted(df['dtype'].unique())}")
    print(f"Batch sizes: {sorted(df['batch'].unique())}")
    print(f"Seq lens: {sorted(df['seq_len'].unique())}")
    return df


def _add_batch_dividers(ax, pivot_index, x):
    """Add vertical dividers between batch-size groups and batch labels."""
    batches = [b for (b, _) in pivot_index]
    prev_batch = batches[0]
    group_starts = [0]
    for idx, b in enumerate(batches):
        if b != prev_batch:
            midpoint = (x[idx - 1] + x[idx]) / 2
            ax.axvline(midpoint, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            group_starts.append(idx)
            prev_batch = b
    group_starts.append(len(batches))

    for g in range(len(group_starts) - 1):
        start, end = group_starts[g], group_starts[g + 1] - 1
        center = (x[start] + x[end]) / 2
        batch_val = batches[group_starts[g]]
        ax.text(center, -0.12, f"Batch = {batch_val}",
                ha="center", va="top", fontsize=10, fontweight="bold",
                transform=ax.get_xaxis_transform())


def plot_bottleneck_shift(df, output_dir):
    """Grouped bar chart: attention vs FFN share for FP16 vs W4A8."""
    Path(output_dir).mkdir(exist_ok=True)
    df_layers = df[df["layer_type"].isin(["attention", "ffn"])].copy()

    configs = df_layers.groupby(["batch", "seq_len"]).ngroups
    fig, ax = plt.subplots(figsize=(max(12, configs * 1.4), 6.5))

    pivot = df_layers.pivot_table(
        index=["batch", "seq_len"],
        columns=["dtype", "layer_type"],
        values="time_pct",
    )

    labels = [f"N={n}" for (_, n) in pivot.index]
    x = np.arange(len(labels))
    width = 0.18

    dtypes = sorted(df["dtype"].unique())
    colors = {"fp16": {"attention": "#4C72B0", "ffn": "#DD8452"},
              "w4a8": {"attention": "#7BAFD4", "ffn": "#F0A875"}}

    i = 0
    for dt in dtypes:
        for lt in ["attention", "ffn"]:
            if (dt, lt) in pivot.columns:
                vals = pivot[(dt, lt)].values
                bars = ax.bar(x + (i - 1.5) * width, vals, width,
                              label=f"{dt.upper()} {lt}",
                              color=colors.get(dt, {}).get(lt, "gray"))
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.5,
                                f"{v:.0f}%", ha="center", va="bottom",
                                fontsize=10, color="black")
                i += 1

    _add_batch_dividers(ax, pivot.index, x)

    ax.set_ylabel("% of Total Runtime")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_title("Attention vs FFN Runtime Share: FP16 vs W4A8 (Qwen3-8B, A100-80GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.18)

    out = Path(output_dir) / "exp2_bottleneck_shift.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_latency(df, output_dir):
    """Total latency comparison FP16 vs W4A8 with speedup annotations."""
    Path(output_dir).mkdir(exist_ok=True)
    df_total = df[df["layer_type"] == "total"].copy()

    pivot = df_total.pivot_table(index=["batch", "seq_len"],
                                 columns="dtype", values="time_ms")

    labels = [f"N={n}" for (_, n) in pivot.index]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.4), 6.5))
    bar_colors = {"fp16": "#4C72B0", "w4a8": "#DD8452"}
    dtypes = sorted(pivot.columns)
    for i, dt in enumerate(dtypes):
        ax.bar(x + (i - 0.5) * width, pivot[dt].values, width,
               label=dt.upper(), color=bar_colors.get(dt, "gray"), alpha=0.85)

    if "fp16" in pivot.columns and "w4a8" in pivot.columns:
        for idx in range(len(x)):
            fp16_val = pivot["fp16"].iloc[idx]
            w4a8_val = pivot["w4a8"].iloc[idx]
            if pd.notna(fp16_val) and pd.notna(w4a8_val) and fp16_val > 0:
                change = (w4a8_val / fp16_val - 1) * 100
                taller = max(fp16_val, w4a8_val)
                color = "#2a7f2a" if change < -0.5 else ("#b22222" if change > 0.5 else "gray")
                sign = "+" if change > 0 else ""
                ax.text(x[idx], taller + taller * 0.02,
                        f"{sign}{change:.1f}%", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=color)

    _add_batch_dividers(ax, pivot.index, x)

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_title("End-to-End Latency: FP16 vs W4A8 (Qwen3-8B, A100-80GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.18)

    out = Path(output_dir) / "exp2_latency.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_memory(df, output_dir):
    """Peak memory (total row) comparison FP16 vs W4A8."""
    if "mem_mb" not in df.columns:
        print("No mem_mb column in CSV — skipping memory plot")
        return
    Path(output_dir).mkdir(exist_ok=True)
    df_total = df[df["layer_type"] == "total"].copy()

    pivot = df_total.pivot_table(index=["batch", "seq_len"],
                                 columns="dtype", values="mem_mb")

    labels = [f"N={n}" for (_, n) in pivot.index]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.4), 6.5))
    bar_colors = {"fp16": "#4C72B0", "w4a8": "#DD8452"}
    dtypes = sorted(pivot.columns)
    for i, dt in enumerate(dtypes):
        vals = pivot[dt].values / 1e3
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=dt.upper(), color=bar_colors.get(dt, "gray"), alpha=0.85)

    if "fp16" in pivot.columns and "w4a8" in pivot.columns:
        for idx in range(len(x)):
            fp16_val = pivot["fp16"].iloc[idx]
            w4a8_val = pivot["w4a8"].iloc[idx]
            if pd.notna(fp16_val) and pd.notna(w4a8_val) and fp16_val > 0:
                change = (w4a8_val / fp16_val - 1) * 100
                taller = max(fp16_val, w4a8_val) / 1e3
                color = "#2a7f2a" if change < -0.5 else ("#b22222" if change > 0.5 else "gray")
                sign = "+" if change > 0 else ""
                ax.text(x[idx], taller + taller * 0.02,
                        f"{sign}{change:.1f}%", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=color)

    _add_batch_dividers(ax, pivot.index, x)

    ax.set_ylabel("Peak Memory (GB)")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_title("Peak GPU Memory: FP16 vs W4A8 (Qwen3-8B, A100-80GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.18)

    out = Path(output_dir) / "exp2_memory.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_summary(df):
    print("\n" + "=" * 70)
    print("LAYER-WISE RUNTIME BREAKDOWN")
    print("=" * 70)

    df_layers = df[df["layer_type"].isin(["attention", "ffn"])].copy()
    pivot = df_layers.pivot_table(
        index=["batch", "seq_len", "layer_type"],
        columns="dtype", values=["time_ms", "time_pct"],
    )
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("TOTAL LATENCY & SPEEDUP")
    print("=" * 70)
    df_total = df[df["layer_type"] == "total"].copy()
    lat = df_total.pivot_table(index=["batch", "seq_len"],
                               columns="dtype", values="time_ms")
    if "fp16" in lat.columns and "w4a8" in lat.columns:
        lat["speedup"] = lat["fp16"] / lat["w4a8"]
        lat["change_pct"] = (lat["w4a8"] / lat["fp16"] - 1) * 100
    print(lat.to_string())

    if "mem_mb" in df.columns:
        print("\n" + "=" * 70)
        print("MEMORY ALLOCATION (MB)")
        print("=" * 70)
        mem_pivot = df[df["layer_type"].isin(["attention", "ffn", "total"])].pivot_table(
            index=["batch", "seq_len", "layer_type"],
            columns="dtype", values="mem_mb",
        )
        print(mem_pivot.to_string())

    print("\n" + "=" * 70)
    print("BOTTLENECK SHIFT (attention % change)")
    print("=" * 70)
    for (b, n), g in df_layers.groupby(["batch", "seq_len"]):
        fp16_a = g[(g["dtype"] == "fp16") & (g["layer_type"] == "attention")]["time_pct"].values
        w4a8_a = g[(g["dtype"] == "w4a8") & (g["layer_type"] == "attention")]["time_pct"].values
        if len(fp16_a) and len(w4a8_a):
            shift = w4a8_a[0] - fp16_a[0]
            print(f"  B={b}, N={n:4d}: FP16 attn {fp16_a[0]:.1f}% -> W4A8 attn {w4a8_a[0]:.1f}% ({shift:+.1f} pp)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_exp2_layers.csv")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    df = load_results(args.input)
    plot_bottleneck_shift(df, args.output_dir)
    plot_latency(df, args.output_dir)
    plot_memory(df, args.output_dir)
    print_summary(df)


if __name__ == "__main__":
    main()
