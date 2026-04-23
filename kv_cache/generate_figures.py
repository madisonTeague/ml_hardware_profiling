"""
Figure generation for the hardware profiling paper.

Produces four figures that bridge Exp 1 (KV cache) and Exp 2 (weight quant):
  figures/fig_kv_vram.png         — KV cache VRAM vs model weights
  figures/fig_amdahl.png          — Amdahl's law projection of INT8KV full-model benefit
  figures/fig_roofline_attn.png   — Roofline plot for attention kernels (Exp 1)
  figures/fig_memory_comparison.png — Side-by-side: W4A8 VRAM vs INT8KV KV savings

Run locally (requires matplotlib, pandas, numpy):
    python generate_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE      = Path(__file__).parent
KV_CSV    = HERE / "results" / "results_from_modal.csv"
EXP2_CSV  = HERE.parent / "weight_quant" / "results_exp2_layers.csv"
COMBINED_MEM_CSV  = HERE.parent / "combined" / "results" / "combined_memory.csv"
COMBINED_PPL_JSON = HERE.parent / "combined" / "results" / "combined_perplexity.json"
OUT_DIR   = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Shared style
BLUE   = "#4878CF"
ORANGE = "#E87722"
GREEN  = "#59A14F"
RED    = "#E15759"
GRAY   = "#9E9E9E"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

kv_df   = pd.read_csv(KV_CSV)
exp2_df = pd.read_csv(EXP2_CSV)

# Qwen3-8B constants
MODEL_GB   = 8e9 * 2 / 1e9   # FP16 weights ~16 GB
LAYERS     = 32
HEADS      = 32
HEAD_DIM   = 128
SEQ_LENS   = [512, 2048, 4096]
BATCHES    = [1, 4, 16]

# ---------------------------------------------------------------------------
# Fig 1: KV cache VRAM vs model weights
# ---------------------------------------------------------------------------

def fig_kv_vram():
    configs, fp16_vals, int8_vals, model_vals = [], [], [], []
    for N in SEQ_LENS:
        for B in BATCHES:
            kv_fp16 = 2 * LAYERS * B * HEADS * N * HEAD_DIM * 2 / 1e9
            kv_int8 = 2 * LAYERS * B * HEADS * N * HEAD_DIM * 1 / 1e9
            configs.append(f"B={B}\nN={N}")
            fp16_vals.append(kv_fp16)
            int8_vals.append(kv_int8)
            model_vals.append(MODEL_GB)

    x = np.arange(len(configs))
    w = 0.28
    fig, ax = plt.subplots(figsize=(14, 5))

    b1 = ax.bar(x - w, model_vals, w, label="Model weights (FP16)", color=GRAY,   zorder=2)
    b2 = ax.bar(x,     fp16_vals,  w, label="KV cache FP16",        color=BLUE,   zorder=2)
    b3 = ax.bar(x + w, int8_vals,  w, label="KV cache INT8",        color=ORANGE, zorder=2)

    # Annotate savings > 1 GB
    for i, (fp, i8) in enumerate(zip(fp16_vals, int8_vals)):
        saving = fp - i8
        if saving >= 1.0:
            ax.annotate(f"−{saving:.1f}GB", xy=(x[i] + w, i8 + saving/2),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.axhline(MODEL_GB, color=GRAY, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Memory (GB)")
    ax.set_title("KV Cache Memory vs Model Weights — Qwen3-8B on A100\n"
                 "INT8KV saves 50% of KV cache; critical when KV > model weights (large B×N)")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)

    # Shade region where KV > model weights
    for i, fp in enumerate(fp16_vals):
        if fp > MODEL_GB:
            ax.axvspan(x[i] - 1.5*w, x[i] + 1.5*w, alpha=0.08, color=RED, zorder=1)

    plt.tight_layout()
    out = OUT_DIR / "fig_kv_vram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2: Amdahl's law projection
# ---------------------------------------------------------------------------

def fig_amdahl():
    """
    For each (B, N): use attention share from Exp 2 (FP16) and kernel
    speedup from Exp 1 to project full-model benefit.
    """
    # Kernel speedups from Exp 1 (median across B within each N)
    speedup_by_N = {}
    for N in SEQ_LENS:
        fp16_lat = kv_df[(kv_df.dtype == "fp16")   & (kv_df.seq_len == N)]["latency_ms"]
        i8_lat   = kv_df[(kv_df.dtype == "int8kv") & (kv_df.seq_len == N)]["latency_ms"]
        s = (fp16_lat.values / i8_lat.values).mean()   # mean speedup ratio across batch
        speedup_by_N[N] = s

    rows = []
    for N in SEQ_LENS:
        for B in BATCHES:
            attn_row = exp2_df[
                (exp2_df.dtype == "fp16") &
                (exp2_df.seq_len == N) &
                (exp2_df.batch == B) &
                (exp2_df.layer_type == "attention")
            ]
            if attn_row.empty:
                continue
            f = attn_row["time_pct"].values[0] / 100.0
            s = speedup_by_N[N]
            # Amdahl: S_total = 1 / ((1 - f) + f/s)
            S_total = 1.0 / ((1 - f) + f / s)
            proj_pct = (S_total - 1) * 100
            rows.append(dict(B=B, N=N, attn_share=f*100, kernel_speedup=(s-1)*100,
                             projected_fullmodel=proj_pct))

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: projected full-model speedup grouped by batch
    ax = axes[0]
    batch_colors = {1: BLUE, 4: ORANGE, 16: GREEN}
    for B, grp in df.groupby("B"):
        ax.plot(grp["N"], grp["projected_fullmodel"], "o-",
                color=batch_colors[B], label=f"B={B}", linewidth=2, markersize=7)
    # Also plot the raw kernel speedup for reference
    for N, s in speedup_by_N.items():
        ax.scatter(N, (s-1)*100, marker="x", color=GRAY, s=80, zorder=5)
    ax.scatter([], [], marker="x", color=GRAY, label="Kernel speedup (isolated)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Speedup (%)")
    ax.set_title("Amdahl's Law: INT8KV Full-Model Projection",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Sequence length\n"
                  r"$\it{Kernel\ gains\ (✕)\ diluted\ —\ attention\ ≈\ 30–35\%\ of\ runtime}$",
                  fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xticks(SEQ_LENS)

    # Right: scatter attention share vs projected speedup
    ax = axes[1]
    sc = ax.scatter(df["attn_share"], df["projected_fullmodel"],
                    c=df["N"], cmap="viridis", s=120, zorder=3)

    # N is encoded by color — label only batch size.
    # Offsets derived from actual point coordinates:
    #   N=512:  B=1 at (34.72, 4.90) — isolated; B=4/16 at (29.48/29.63, 4.13/4.15) — nearly identical
    #   N=2048: B=1/4/16 at (31.29–31.85, 5.68–5.79) — tight diagonal
    #   N=4096: B=1/4/16 at (33.85–34.28, 6.78–6.87) — tight diagonal
    label_offsets = {
        (1,  512):  ( 8,   2),   # isolated — go right
        (4,  512):  (-46,  14),  # stacked pair — go upper-left
        (16, 512):  (-46, -16),  # stacked pair — go lower-left
        (1,  2048): (-46,   2),  # leftmost of cluster — go left
        (4,  2048): (  2,  16),  # middle — go up
        (16, 2048): (  8,   2),  # rightmost — go right
        (1,  4096): (-46, -16),  # leftmost — go lower-left
        (4,  4096): (  2,  16),  # middle — go up
        (16, 4096): (  8,   2),  # rightmost — go right
    }
    label_style = dict(
        fontsize=9, color="#111111",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
    )
    for _, row in df.iterrows():
        key = (int(row.B), int(row.N))
        dx, dy = label_offsets.get(key, (8, 4))
        ax.annotate(
            f"B={int(row.B)}",
            (row.attn_share, row.projected_fullmodel),
            textcoords="offset points", xytext=(dx, dy),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
            **label_style,
        )
    plt.colorbar(sc, ax=ax, label="Sequence length (N)")
    ax.set_xlabel("Attention share of total runtime (%)")
    ax.set_ylabel("Projected full-model speedup (%)")
    ax.set_title("Attention Share vs. Projected INT8KV Benefit",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Attention share of total runtime (%)\n"
                  r"$\it{Share\ stable\ at\ 30–35\%,\ capping\ projected\ gain\ at\ ≈6\%}$",
                  fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle("", fontsize=1)   # prevent suptitle from stealing space
    plt.tight_layout(rect=[0, 0, 1, 1])
    out = OUT_DIR / "fig_amdahl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Roofline plot for attention kernels (Exp 1)
# ---------------------------------------------------------------------------

def fig_roofline():
    PEAK_BW_GBPS   = 2000.0   # A100 HBM2e
    PEAK_FP16_TFLOPS = 312.0  # A100 tensor core FP16
    RIDGE = PEAK_FP16_TFLOPS * 1e3 / PEAK_BW_GBPS   # FLOP/byte

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw roofline
    ai_range = np.logspace(-1, 4, 400)
    roof = np.minimum(PEAK_BW_GBPS * ai_range / 1e3, PEAK_FP16_TFLOPS)
    ax.loglog(ai_range, roof, "k-", linewidth=2.5, label="A100 roofline", zorder=2)
    ax.axvline(RIDGE, color="k", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.text(RIDGE * 1.15, 0.15, f"Ridge\n{RIDGE:.0f} FLOP/B", fontsize=9, va="bottom")

    # Plot kernel points
    markers = {1: "o", 4: "s", 16: "^"}
    colors  = {"fp16": BLUE, "int8kv": ORANGE}
    labels_added = set()

    for _, row in kv_df.iterrows():
        ai   = row["ai_theory"]
        perf = row["tflops"]
        B, N = row["batch"], row["seq_len"]
        d    = row["dtype"]
        m    = markers[B]
        c    = colors[d]
        label = f"{d.upper()}" if d not in labels_added else "_"
        ax.scatter(ai, perf, marker=m, color=c, s=80, zorder=4,
                   label=label, alpha=0.85, edgecolors="white", linewidths=0.5)
        labels_added.add(d)

    # Legend for batch size markers
    for B, m in markers.items():
        ax.scatter([], [], marker=m, color=GRAY, label=f"B={B}")

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Achieved Performance (TFLOP/s)")
    ax.set_title("Roofline: FP16 vs INT8KV Attention Kernels on A100\n"
                 "Both variants run at 3–11% of compute ceiling — Triton occupancy is the bottleneck")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.grid(True, which="both", alpha=0.2)
    ax.xaxis.grid(True, which="both", alpha=0.2)

    # Annotate the hardware utilization gap.
    # Arrow spans from the A100 FP16 ceiling (312 TFLOPS) down to the
    # best-case INT8KV point at AI~1365 (~35 TFLOPS): gap = 312/35 ≈ 9×.
    ax.annotate("", xy=(1500, 280), xytext=(1500, 35),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(1600, 120, "~9×\nbelow\nceiling", color=RED, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "fig_roofline_attn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4: Memory savings comparison — W4A8 VRAM vs INT8KV KV savings
# ---------------------------------------------------------------------------

def fig_memory_comparison():
    """
    Side-by-side bar chart showing GB saved by each technique at B=1 and B=16.
    Makes the 'orthogonal resources' argument visually concrete.
    Uses measured values from combined_memory.csv.
    """
    cm = pd.read_csv(COMBINED_MEM_CSV)
    cm_idx = cm.set_index(["dtype", "batch", "seq_len"])["peak_mem_gb"]

    configs, kv_save, w4a8_save = [], [], []
    for N in SEQ_LENS:
        for B in [1, 16]:
            fp16_mem = cm_idx[("fp16",   B, N)]
            int8_mem = cm_idx[("int8kv", B, N)]
            w4a8_mem = cm_idx[("w4a8",   B, N)]
            configs.append(f"B={B}, N={N}")
            kv_save.append(fp16_mem - int8_mem)
            w4a8_save.append(fp16_mem - w4a8_mem)

    x = np.arange(len(configs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w/2, w4a8_save, w, label="W4A8: weight memory saved",  color=BLUE,   zorder=2)
    ax.bar(x + w/2, kv_save,   w, label="INT8KV: KV cache saved",     color=ORANGE, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("GPU Memory Saved (GB)")
    ax.set_title("Memory Savings: W4A8 (static weights) vs INT8KV (dynamic KV cache)\n"
                 "W4A8 dominates at small B×N; INT8KV savings grow with batch and sequence length")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    out = OUT_DIR / "fig_memory_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 5: Combined experiment — measured peak GPU memory (all 4 configs)
# ---------------------------------------------------------------------------

def fig_combined_memory():
    """
    Line plots of measured peak GPU memory for all 4 configs across N,
    one subplot per batch size. Directly shows additive composition.
    """
    df = pd.read_csv(COMBINED_MEM_CSV)

    dtype_colors = {"fp16": GRAY, "int8kv": BLUE, "w4a8": ORANGE, "combined": GREEN}
    dtype_labels = {
        "fp16":     "FP16 baseline",
        "int8kv":   "INT8 KV cache",
        "w4a8":     "W4A8 weights",
        "combined": "W4A8 + INT8KV",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, B in zip(axes, [1, 4, 16]):
        sub = df[df.batch == B]
        for dtype in ["fp16", "int8kv", "w4a8", "combined"]:
            grp = sub[sub.dtype == dtype].sort_values("seq_len")
            ax.plot(grp["seq_len"], grp["peak_mem_gb"], "o-",
                    color=dtype_colors[dtype], label=dtype_labels[dtype],
                    linewidth=2, markersize=7)
        ax.set_title(f"Batch size B={B}", fontsize=11)
        ax.set_xlabel("Sequence length")
        ax.set_xticks(SEQ_LENS)
        ax.yaxis.grid(True, alpha=0.3)
        if B == 1:
            ax.set_ylabel("Peak GPU Memory (GB)")
        if B == 16:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Exp 4: Measured Peak GPU Memory — Qwen3-8B on A100\n"
        "W4A8 saves static weights (~10.3 GB); INT8KV saves scale with B×N; savings compose additively",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "fig_combined_memory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 6: Perplexity bar chart with 5% degradation budget
# ---------------------------------------------------------------------------

def fig_perplexity():
    import json

    with open(COMBINED_PPL_JSON) as f:
        ppl = json.load(f)

    order  = ["fp16", "int8kv", "w4a8", "combined"]
    labels = ["FP16\nbaseline", "INT8\nKV", "W4A8", "W4A8 +\nINT8KV"]
    values = [ppl[k] for k in order]
    colors = [GRAY, BLUE, ORANGE, GREEN]

    fp16_ppl    = ppl["fp16"]
    budget_ppl  = fp16_ppl * 1.05

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, zorder=2, width=0.5)

    ax.axhline(budget_ppl, color=RED, linestyle="--", linewidth=1.5,
               label=f"5% degradation budget ({budget_ppl:.2f})")

    for bar, val, key in zip(bars, values, order):
        deg_str = "" if key == "fp16" else f"\n({(val/fp16_ppl - 1)*100:+.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.3f}{deg_str}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Perplexity — WikiText-103 (lower is better)")
    ax.set_title(
        "Exp 4: Perplexity — W4A8 + INT8KV vs FP16 Baseline\n"
        "Combined quantization stays within 5% quality budget",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(11.5, 13.8)

    plt.tight_layout()
    out = OUT_DIR / "fig_perplexity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 7: Savings waterfall at B=16 — decomposing W4A8 vs INT8KV contributions
# ---------------------------------------------------------------------------

def fig_savings_waterfall():
    """
    Stacked bar showing how fp16 memory breaks down into:
      remaining (combined) + INT8KV saved + W4A8 saved
    at B=16 across all three sequence lengths.
    """
    df  = pd.read_csv(COMBINED_MEM_CSV)
    idx = df.set_index(["dtype", "batch", "seq_len"])["peak_mem_gb"]

    w4a8_save = [idx[("fp16", 16, N)] - idx[("w4a8",   16, N)] for N in SEQ_LENS]
    kv_save   = [idx[("w4a8", 16, N)] - idx[("combined", 16, N)] for N in SEQ_LENS]
    remaining = [idx[("combined", 16, N)] for N in SEQ_LENS]

    x = np.arange(len(SEQ_LENS))
    w = 0.5

    fig, ax = plt.subplots(figsize=(9, 6))

    b_rem  = ax.bar(x, remaining, w, label="Combined remaining",          color=GREEN,  zorder=2)
    b_kv   = ax.bar(x, kv_save,   w, bottom=remaining,                    label="INT8KV saves (KV cache)", color=ORANGE, zorder=2)
    b_w4a8 = ax.bar(x, w4a8_save, w,
                    bottom=[r + k for r, k in zip(remaining, kv_save)],
                    label="W4A8 saves (weights)", color=BLUE, zorder=2)

    for i, (w4, kv, rem) in enumerate(zip(w4a8_save, kv_save, remaining)):
        ax.text(i, rem / 2,         f"{rem:.1f} GB",  ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(i, rem + kv / 2,    f"−{kv:.1f} GB",  ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(i, rem + kv + w4/2, f"−{w4:.1f} GB",  ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={N}" for N in SEQ_LENS])
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title(
        "Memory Savings Decomposition at B=16 — Qwen3-8B on A100\n"
        "W4A8 saves a fixed ~10.3 GB (static weights); INT8KV savings grow with sequence length",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    out = OUT_DIR / "fig_savings_waterfall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Writing figures to {OUT_DIR}/")
    fig_kv_vram()
    fig_amdahl()
    fig_roofline()
    fig_memory_comparison()
    fig_combined_memory()
    fig_perplexity()
    fig_savings_waterfall()
    print("Done.")
