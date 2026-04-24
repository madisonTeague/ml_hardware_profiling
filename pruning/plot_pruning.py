"""
Poster-quality figures for Experiment 3: Structured FFN Pruning.

NARRATIVE: Structured FFN pruning produces hardware efficiency gains (throughput,
attention/FFN balance shift) but ALL scoring methods violate the 5% perplexity
budget even at 1% pruning on Qwen3-8B.  The throughput/VRAM gains are therefore
only meaningful as upper-bound hardware characterisation, not practical results.

Run from the pruning/ directory:
    python plot_pruning.py

Perplexity data (fill in PPL_DATA below once downloaded from Modal):
    modal volume get pruning-results perplexity_wanda.csv
    modal volume get pruning-results perplexity_spectral.csv
    modal volume get pruning-results perplexity_spectral-wanda.csv

Outputs (saved to pruning/figures/):
    fig1_ppl_regression.png/pdf          — Perplexity vs ratio (primary quality chart)
    fig2_quality_efficiency.png/pdf      — PPL cost vs throughput gain (Pareto frontier)
    fig3_compute_balance.png/pdf         — Stacked bar: attn/FFN share per ratio
    fig4_attn_ffn_convergence.png/pdf    — Attn vs FFN latency convergence at B=1, N=4096
    fig5_speedup_heatmap.png/pdf         — Speedup heatmap across all B×N at 30%
    fig6_poster_panel.png/pdf            — 2-panel: PPL regression + throughput (poster drop-in)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Perplexity data
# Fill these in from the perplexity CSVs on Modal once downloaded.
# Keys are pruning ratios as integers (0 = baseline).
# ─────────────────────────────────────────────────────────────────────────────
PPL_DATA: dict[str, dict[int, float]] = {
    # From perplexity_wanda.csv
    "Wanda": {
        0:  12.27,
        1:  17.92,
        5:  29.94,
        10: 40.96,
        20: 116.36,
        30: 231.61,
    },
    # From perplexity_spectral.csv  (pure SVD, no activation calibration)
    "Spectral": {
        0:  12.27,
        1:  20.11,
        5:  51.17,
        10: 152.35,
        20: 464.85,
        30: 1536.41,
    },
    # From perplexity_spectral-wanda.csv  (best of the three)
    "Spectral+Wanda": {
        0:  12.27,
        1:  16.32,
        5:  33.08,
        10: 44.83,
        20: 76.33,
        30: 177.41,
    },
}

# 5% perplexity budget: baseline PPL × 1.05
BASELINE_PPL   = 12.27
PPL_BUDGET     = BASELINE_PPL * 1.05   # 12.88
PPL_BUDGET_PCT = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Palette
M_COLORS  = {"Wanda": "#1f77b4", "Spectral": "#d62728", "Spectral+Wanda": "#2ca02c"}
M_MARKERS = {"Wanda": "o",       "Spectral": "s",        "Spectral+Wanda": "^"}
M_LINES   = {"Wanda": "-",       "Spectral": "--",       "Spectral+Wanda": "-."}

ATTN_CLR  = "#e15759"   # warm red
FFN_CLR   = "#4e79a7"   # steel blue
OTH_CLR   = "#bab0ac"   # grey

RATIOS      = [0, 1, 5, 10, 20, 30]
RATIO_LBLS  = ["Base", "1%", "5%", "10%", "20%", "30%"]

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ratio_pct"] = df["prune_ratio"].apply(
        lambda x: 0 if str(x).strip().lower() == "baseline" else int(x)
    )
    return df


DFS: dict[str, pd.DataFrame] = {
    "Wanda":          _load("results_wanda.csv"),
    "Spectral":       _load("results_spectral.csv"),
    "Spectral+Wanda": _load("results_spectral-wanda.csv"),
}


def sel(df: pd.DataFrame, batch: int, seq_len: int,
        layer_type: str = "total") -> pd.DataFrame:
    return (
        df[(df.batch == batch) & (df.seq_len == seq_len) &
           (df.layer_type == layer_type)]
        .sort_values("ratio_pct")
        .reset_index(drop=True)
    )


def throughput(df: pd.DataFrame, batch: int, seq_len: int,
               ratio_pct: int) -> float:
    row = df[(df.batch == batch) & (df.seq_len == seq_len) &
             (df.layer_type == "total") & (df.ratio_pct == ratio_pct)]
    return float(row["throughput_tok_s"].iloc[0])


def layer_pct(df: pd.DataFrame, batch: int, seq_len: int,
              ratio_pct: int, layer: str) -> float:
    row = df[(df.batch == batch) & (df.seq_len == seq_len) &
             (df.layer_type == layer) & (df.ratio_pct == ratio_pct)]
    return float(row["time_pct"].iloc[0])


def layer_ms(df: pd.DataFrame, batch: int, seq_len: int,
             ratio_pct: int, layer: str) -> float:
    row = df[(df.batch == batch) & (df.seq_len == seq_len) &
             (df.layer_type == layer) & (df.ratio_pct == ratio_pct)]
    return float(row["time_ms"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Perplexity regression vs pruning ratio (PRIMARY quality chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_ppl_regression() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw the 5% budget line first so it sits behind the data
    ax.axhline(PPL_BUDGET, color="#d62728", linestyle="--", linewidth=1.8,
               label=f"+5% budget  (PPL = {PPL_BUDGET:.2f})", zorder=1)
    ax.fill_between([-1, 35], PPL_BUDGET, ax.get_ylim()[1] if ax.get_ylim()[1] > PPL_BUDGET else 200,
                    color="#d62728", alpha=0.06, zorder=0)

    has_data = False
    for method, ppl_dict in PPL_DATA.items():
        xs = sorted(k for k, v in ppl_dict.items() if v is not None)
        ys = [ppl_dict[k] for k in xs]
        if not ys:
            continue
        has_data = True
        color = M_COLORS.get(method, "grey")
        style = M_LINES.get(method, ":")
        marker = M_MARKERS.get(method, "x")
        ax.plot(xs, ys, marker=marker, color=color, linestyle=style,
                linewidth=2.2, markersize=7, label=method, zorder=3)

        # Mark points that violate the budget
        for x_v, y_v in zip(xs, ys):
            if x_v > 0 and y_v > PPL_BUDGET:
                ax.plot(x_v, y_v, "x", color="#d62728", markersize=10,
                        markeredgewidth=2.5, zorder=4)

    if not has_data:
        # Placeholder until real PPL data is filled in
        ax.text(15, 100, "⚠ Fill PPL_DATA dict\nwith perplexity CSVs\nfrom Modal volume",
                ha="center", va="center", fontsize=12, color="grey",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                          edgecolor="#ffc107"))

    ax.set_xticks(RATIOS)
    ax.set_xticklabels(RATIO_LBLS)
    ax.set_xlabel("FFN pruning ratio")
    ax.set_ylabel("Perplexity (WikiText-103 test, ↓ better)")
    ax.set_title(
        "Perplexity Regression  ·  Structured FFN Pruning\n"
        "All scoring methods violate 5% budget at every ratio",
        pad=8,
    )
    ax.set_xlim(-1, 32)
    ax.legend(fontsize=9)

    # Secondary note
    ax.text(0.98, 0.97,
            "✗ = violates 5% budget",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#d62728")

    fig.savefig(OUT / "fig1_ppl_regression.pdf")
    fig.savefig(OUT / "fig1_ppl_regression.png")
    plt.close(fig)
    print("✓ fig1_ppl_regression")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Quality-efficiency frontier (PPL cost vs throughput gain)
# This is the critical chart: shows that every point that gives speedup
# comes at unacceptable PPL cost — no point in the feasible region.
# ─────────────────────────────────────────────────────────────────────────────
def fig2_quality_efficiency() -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Shade the feasible region (top-left: good speedup AND low PPL cost)
    ax.axhline(PPL_BUDGET_PCT, color="#d62728", linestyle="--", linewidth=1.8,
               label=f"5% PPL budget", zorder=2)
    ax.fill_between([0, 30], 0, PPL_BUDGET_PCT,
                    color="#2ca02c", alpha=0.08, zorder=0,
                    label="Feasible region")
    ax.fill_between([0, 30], PPL_BUDGET_PCT, 9999,
                    color="#d62728", alpha=0.05, zorder=0,
                    label="Budget violated")

    has_data = False
    for method, ppl_dict in PPL_DATA.items():
        base_ppl = ppl_dict.get(0, BASELINE_PPL)
        points = []
        for ratio in [1, 5, 10, 20, 30]:
            ppl_v = ppl_dict.get(ratio)
            if ppl_v is None:
                continue
            # throughput gain from Wanda df (or spectral df if available)
            df_key = method if method in DFS else "Wanda"
            tp_gain = 100 * (throughput(DFS[df_key], 16, 4096, ratio) -
                             throughput(DFS[df_key], 16, 4096, 0)) / \
                      throughput(DFS[df_key], 16, 4096, 0)
            ppl_cost = 100 * (ppl_v - base_ppl) / base_ppl
            points.append((tp_gain, ppl_cost, ratio))

        if not points:
            continue
        has_data = True
        color = M_COLORS.get(method, "grey")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker=M_MARKERS.get(method, "o"), color=color,
                linestyle=M_LINES.get(method, "-"), linewidth=2, markersize=8,
                label=method, zorder=3)
        for tp_g, ppl_c, r in points:
            ax.annotate(f"{r}%", (tp_g, ppl_c),
                        xytext=(tp_g + 0.3, ppl_c + 0.5),
                        fontsize=7.5, color=color)

    if not has_data:
        ax.text(10, 500, "⚠ Fill PPL_DATA dict\nwith perplexity CSVs",
                ha="center", va="center", fontsize=12, color="grey",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                          edgecolor="#ffc107"))

    ax.set_xlabel("Throughput gain vs. baseline (%)  →  better")
    ax.set_ylabel("Perplexity increase vs. baseline (%)  →  worse")
    ax.set_title(
        "Quality–Efficiency Frontier  (B=16, N=4096)\n"
        "No scoring method achieves a point in the feasible region",
        pad=8,
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.savefig(OUT / "fig2_quality_efficiency.pdf")
    fig.savefig(OUT / "fig2_quality_efficiency.png")
    plt.close(fig)
    print("✓ fig2_quality_efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Compute balance stacked bars (B=16, N=4096) — hardware characterisation
# NOTE: these gains are theoretical; all models violate PPL budget.
# ─────────────────────────────────────────────────────────────────────────────
def fig3_compute_balance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                              constrained_layout=True)

    for ax, (method, df) in zip(axes, DFS.items()):
        attn = [layer_pct(df, 16, 4096, r, "attention") for r in RATIOS]
        ffn  = [layer_pct(df, 16, 4096, r, "ffn")       for r in RATIOS]
        oth  = [layer_pct(df, 16, 4096, r, "other")     for r in RATIOS]
        x    = np.arange(len(RATIOS))
        w    = 0.6

        # Dim bars beyond 5% budget (ratio > 1% for all methods) with hatching
        bars_attn = ax.bar(x, attn, w, color=ATTN_CLR, label="Attention")
        bars_ffn  = ax.bar(x, ffn,  w, bottom=attn,  color=FFN_CLR, label="FFN")
        bars_oth  = ax.bar(x, oth,  w,
                           bottom=[a + f for a, f in zip(attn, ffn)],
                           color=OTH_CLR, label="Other")

        # Add red X overlay on all pruned ratios (all violate budget)
        for i in range(1, len(RATIOS)):   # skip baseline
            ax.text(i, 98, "✗", ha="center", va="top",
                    color="#d62728", fontsize=14, fontweight="bold")

        # Annotate attn% and ffn% inside bars
        for i in range(len(RATIOS)):
            ax.text(i, attn[i] / 2, f"{attn[i]:.0f}%",
                    ha="center", va="center", color="white", fontsize=8.5,
                    fontweight="bold")
            ax.text(i, attn[i] + ffn[i] / 2, f"{ffn[i]:.0f}%",
                    ha="center", va="center", color="white", fontsize=8.5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(RATIO_LBLS, fontsize=9)
        ax.set_title(method)
        ax.set_ylim(0, 105)
        ax.grid(axis="x", alpha=0)
        if ax is axes[0]:
            ax.set_ylabel("% of total forward-pass time")
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Compute Balance Shift  (B=16, N=4096)  ·  ✗ = PPL budget violated",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(OUT / "fig3_compute_balance.pdf")
    fig.savefig(OUT / "fig3_compute_balance.png")
    plt.close(fig)
    print("✓ fig3_compute_balance")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Attention vs FFN latency convergence (B=1, N=4096)
# Shows the hardware effect, with PPL-budget violation marked
# ─────────────────────────────────────────────────────────────────────────────
def fig4_convergence() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                              constrained_layout=True)

    for ax, (method, df) in zip(axes, DFS.items()):
        attn_ms = [layer_ms(df, 1, 4096, r, "attention") for r in RATIOS]
        ffn_ms  = [layer_ms(df, 1, 4096, r, "ffn")       for r in RATIOS]

        ax.plot(RATIOS, attn_ms, marker="o", color=ATTN_CLR,
                linewidth=2.2, markersize=7, label="Attention")
        ax.plot(RATIOS, ffn_ms,  marker="s", color=FFN_CLR,
                linewidth=2.2, markersize=7, label="FFN")

        ax.fill_between(RATIOS, attn_ms, ffn_ms,
                        where=[f > a for a, f in zip(attn_ms, ffn_ms)],
                        alpha=0.10, color=FFN_CLR)

        # Mark all pruned ratios as PPL-violating (red shading over x-axis)
        for r in RATIOS[1:]:
            ax.axvspan(r - 1.5, r + 1.5, alpha=0.06, color="#d62728", zorder=0)

        a30, f30 = attn_ms[-1], ffn_ms[-1]
        ax.annotate(
            f"Near-tie\n{a30:.0f} vs {f30:.0f} ms\n(PPL budget\nviolated)",
            xy=(30, (a30 + f30) / 2),
            xytext=(17, max(a30, f30) + 18),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9),
            fontsize=7.5, color="#d62728", ha="center",
        )

        ax.set_xticks(RATIOS)
        ax.set_xticklabels(RATIO_LBLS, fontsize=9)
        ax.set_title(method)
        if ax is axes[0]:
            ax.set_ylabel("Latency per forward pass (ms)")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Pruning ratio")

    fig.suptitle(
        "Attention vs. FFN Convergence  (B=1, N=4096)  ·  red = PPL budget violated",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(OUT / "fig4_attn_ffn_convergence.pdf")
    fig.savefig(OUT / "fig4_attn_ffn_convergence.png")
    plt.close(fig)
    print("✓ fig4_attn_ffn_convergence")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Speedup heatmap at 30% — shown as "theoretical upper bound"
# ─────────────────────────────────────────────────────────────────────────────
def fig5_speedup_heatmap() -> None:
    BATCHES  = [1, 4, 16]
    SEQ_LENS = [512, 2048, 4096]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)

    all_grids = []
    for method, df in DFS.items():
        grid = np.zeros((len(BATCHES), len(SEQ_LENS)))
        for i, B in enumerate(BATCHES):
            for j, N in enumerate(SEQ_LENS):
                grid[i, j] = throughput(df, B, N, 30) / throughput(df, B, N, 0)
        all_grids.append(grid)

    vmin = min(g.min() for g in all_grids) * 0.995
    vmax = max(g.max() for g in all_grids) * 1.005

    for ax, (method, _), grid in zip(axes, DFS.items(), all_grids):
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(SEQ_LENS)))
        ax.set_xticklabels([f"N={n}" for n in SEQ_LENS], fontsize=9)
        ax.set_yticks(range(len(BATCHES)))
        ax.set_yticklabels([f"B={b}" for b in BATCHES], fontsize=9)
        ax.set_title(method)
        ax.grid(False)

        for i in range(len(BATCHES)):
            for j in range(len(SEQ_LENS)):
                val = grid[i, j]
                tc = "white" if val > 1.17 or val < 1.03 else "black"
                ax.text(j, i, f"×{val:.2f}", ha="center", va="center",
                        fontsize=9.5, color=tc, fontweight="bold")

    plt.colorbar(im, ax=axes.tolist(),
                 label="Throughput speedup vs baseline  (PPL budget NOT met)",
                 shrink=0.85)
    fig.suptitle(
        "Theoretical Throughput Speedup at 30% FFN Pruning  ·  all B×N\n"
        "⚠ These are hardware upper bounds — all models violate the 5% PPL budget",
        fontsize=11, fontweight="bold",
    )

    fig.savefig(OUT / "fig5_speedup_heatmap.pdf")
    fig.savefig(OUT / "fig5_speedup_heatmap.png")
    plt.close(fig)
    print("✓ fig5_speedup_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Poster drop-in 2-panel: PPL regression (left) + compute balance (right)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_poster_panel() -> None:
    fig = plt.figure(figsize=(12, 4.5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    # ── Left: perplexity regression ──────────────────────────────────
    ax_l.axhline(PPL_BUDGET, color="#d62728", linestyle="--", linewidth=1.8,
                 label=f"5% budget  (PPL ≤ {PPL_BUDGET:.1f})", zorder=2)
    ax_l.fill_between([-1, 32], PPL_BUDGET, 2000,
                      color="#d62728", alpha=0.05, zorder=0)

    has_ppl = False
    for method, ppl_dict in PPL_DATA.items():
        xs = sorted(k for k, v in ppl_dict.items() if v is not None)
        ys = [ppl_dict[k] for k in xs]
        if not ys:
            continue
        has_ppl = True
        c = M_COLORS.get(method, "grey")
        ax_l.plot(xs, ys, marker=M_MARKERS.get(method, "o"), color=c,
                  linestyle=M_LINES.get(method, "-"), linewidth=2.2,
                  markersize=7, label=method, zorder=3)
        for x_v, y_v in zip(xs, ys):
            if x_v > 0 and y_v > PPL_BUDGET:
                ax_l.plot(x_v, y_v, "x", color="#d62728",
                          markersize=9, markeredgewidth=2.5, zorder=4)

    if not has_ppl:
        ax_l.text(15, 300,
                  "⚠ Perplexity data\nnot yet downloaded\n"
                  "from Modal volume.\n\n"
                  "Fill PPL_DATA in\nplot_pruning.py",
                  ha="center", va="center", fontsize=10, color="#856404",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                            edgecolor="#ffc107", linewidth=1.5))

    ax_l.set_xticks(RATIOS)
    ax_l.set_xticklabels(RATIO_LBLS)
    ax_l.set_xlabel("FFN pruning ratio")
    ax_l.set_ylabel("Perplexity (WikiText-103, ↓ better)")
    ax_l.set_title("Perplexity Regression\n(5% budget = red line)")
    ax_l.set_xlim(-1, 32)
    ax_l.legend(fontsize=8.5)

    # ── Right: compute balance (Wanda, B=16, N=4096) ─────────────────
    df_w = DFS["Wanda"]
    attn = [layer_pct(df_w, 16, 4096, r, "attention") for r in RATIOS]
    ffn  = [layer_pct(df_w, 16, 4096, r, "ffn")       for r in RATIOS]
    oth  = [layer_pct(df_w, 16, 4096, r, "other")     for r in RATIOS]
    x    = np.arange(len(RATIOS))

    ax_r.bar(x, attn, 0.6, color=ATTN_CLR, label="Attention")
    ax_r.bar(x, ffn,  0.6, bottom=attn, color=FFN_CLR, label="FFN")
    ax_r.bar(x, oth,  0.6, bottom=[a + f for a, f in zip(attn, ffn)],
             color=OTH_CLR, label="Other")

    for i in range(len(RATIOS)):
        ax_r.text(i, attn[i] / 2, f"{attn[i]:.0f}%",
                  ha="center", va="center", color="white",
                  fontsize=9, fontweight="bold")
        ax_r.text(i, attn[i] + ffn[i] / 2, f"{ffn[i]:.0f}%",
                  ha="center", va="center", color="white",
                  fontsize=9, fontweight="bold")

    # Red X on all pruned ratios
    for i in range(1, len(RATIOS)):
        ax_r.text(i, 99, "✗", ha="center", va="top",
                  color="#d62728", fontsize=14, fontweight="bold")

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(RATIO_LBLS)
    ax_r.set_ylim(0, 105)
    ax_r.set_xlabel("FFN pruning ratio")
    ax_r.set_ylabel("% of total forward-pass time")
    ax_r.set_title("Compute Balance Shift — Wanda\n(B=16, N=4096 · ✗ = PPL violated)")
    ax_r.grid(axis="x", alpha=0)
    ax_r.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Structured FFN Pruning  ·  Qwen3-8B  ·  A100 80 GB\n"
        "Hardware gains exist but perplexity budget violated at every ratio — "
        "retraining required",
        fontsize=11, fontweight="bold", y=1.01,
    )

    fig.savefig(OUT / "fig6_poster_panel.pdf")
    fig.savefig(OUT / "fig6_poster_panel.png")
    plt.close(fig)
    print("✓ fig6_poster_panel")


# ─────────────────────────────────────────────────────────────────────────────
# Print a clean data table to stdout (for quick reference / poster table)
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table() -> None:
    print("\n" + "=" * 78)
    print("SUMMARY TABLE  —  B=16, N=4096  (corrected values for poster)")
    print("=" * 78)
    header = f"{'Ratio':<10} {'Method':<18} {'Throughput':>12} {'Δ base':>8} {'VRAM (GB)':>10} {'Attn %':>8} {'FFN %':>7}"
    print(header)
    print("-" * 78)

    for method, df in DFS.items():
        base_tp = throughput(df, 16, 4096, 0)
        totals  = sel(df, 16, 4096, "total")
        base_gb = float(totals[totals.ratio_pct == 0]["peak_vram_mb"].iloc[0]) / 1024

        for ratio in RATIOS:
            tp   = throughput(df, 16, 4096, ratio)
            gb   = float(totals[totals.ratio_pct == ratio]["peak_vram_mb"].iloc[0]) / 1024
            attn = layer_pct(df, 16, 4096, ratio, "attention")
            ffn  = layer_pct(df, 16, 4096, ratio, "ffn")
            dlta = f"+{100*(tp-base_tp)/base_tp:.1f}%" if ratio > 0 else "—"
            lbl  = f"{ratio}%" if ratio > 0 else "base"
            print(f"{lbl:<10} {method:<18} {tp:>12,.0f} {dlta:>8} "
                  f"{gb:>10.2f} {attn:>7.1f}% {ffn:>6.1f}%")
        print()

    print("\nB=1, N=4096 — Attn vs FFN latency at 30% (near-tie check)")
    print("-" * 60)
    for method, df in DFS.items():
        a = layer_ms(df, 1, 4096, 30, "attention")
        f = layer_ms(df, 1, 4096, 30, "ffn")
        a_pct = layer_pct(df, 1, 4096, 30, "attention")
        f_pct = layer_pct(df, 1, 4096, 30, "ffn")
        print(f"  {method:<18}  Attn={a:.1f} ms ({a_pct:.1f}%)   "
              f"FFN={f:.1f} ms ({f_pct:.1f}%)   "
              f"Δ={abs(a-f):.1f} ms")
    print("=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_summary_table()

    # Primary quality chart — needs PPL_DATA filled in
    fig1_ppl_regression()
    fig2_quality_efficiency()

    # Hardware characterisation (annotated with PPL budget violation)
    fig3_compute_balance()
    fig4_convergence()
    fig5_speedup_heatmap()

    # Poster drop-in 2-panel (PPL + compute balance)
    fig6_poster_panel()

    print(f"\nAll figures saved to: {OUT.resolve()}")
    print("PDF + PNG versions generated for each figure.")
    missing = any(
        v is None
        for d in PPL_DATA.values()
        for k, v in d.items()
        if k > 0
    )
    if missing:
        print(
            "\n⚠  PPL_DATA has None entries — some perplexity figures are placeholders.\n"
            "   Download CSVs from Modal and fill in the PPL_DATA dict.\n"
            "   Run: modal volume get pruning-results perplexity_wanda.csv"
        )
