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
KV_CSV    = HERE / "results_from_modal.csv"
EXP2_CSV  = HERE.parent / "weight_quant" / "results_exp2_layers.csv"
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
    """
    # W4A8 VRAM savings from findings.md
    w4a8_savings = {
        (1,  512):  16.6 - 6.3,
        (1,  2048): 16.6 - 6.3,   # model weights dominate at B=1
        (1,  4096): 16.6 - 6.3,
        (4,  512):  20.0 - 8.0,   # approximate — model + small KV
        (4,  2048): 25.0 - 10.0,
        (4,  4096): 30.0 - 12.0,
        (16, 512):  30.0 - 14.0,
        (16, 2048): 46.5 - 26.6,
        (16, 4096): 46.5 - 26.6,
    }

    configs, kv_save, w4a8_save = [], [], []
    for N in SEQ_LENS:
        for B in [1, 16]:
            kv_fp16 = 2 * LAYERS * B * HEADS * N * HEAD_DIM * 2 / 1e9
            kv_int8 = 2 * LAYERS * B * HEADS * N * HEAD_DIM * 1 / 1e9
            configs.append(f"B={B}, N={N}")
            kv_save.append(kv_fp16 - kv_int8)
            w4a8_save.append(w4a8_savings.get((B, N), 0))

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
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Writing figures to {OUT_DIR}/")
    fig_kv_vram()
    fig_amdahl()
    fig_roofline()
    fig_memory_comparison()
    print("Done.")
