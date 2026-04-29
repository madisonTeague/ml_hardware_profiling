"""
Figure: Memory vs. Perplexity tradeoff — one dot per technique.
Motivates the central problem: reduce GPU memory while staying within
the 5% perplexity degradation budget.

Values at B=16, N=4096 (worst-case memory configuration).
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "figures" / "fig_memory_perplexity.png"

# ── Data ─────────────────────────────────────────────────────────────────────
# (label, peak_mem_GB, perplexity, color)
CONFIGS = [
    ("FP16 baseline",   46.5, 12.269, "#888888"),
    ("INT8KV only",     41.7, 12.269, "#4878CF"),
    ("W4A8 only",       36.2, 12.706, "#E87722"),
    ("W4A8 + INT8KV",   31.5, 12.706, "#59A14F"),
]

# Wanda 30% pruning: memory at B=16 N=4096, perplexity far off-chart (232)
WANDA = ("Wanda 30%\n(ppl=232 ↑)", 40.2, 231.6, "#E15759")

FP16_PPL   = 12.269
BUDGET_PPL = FP16_PPL * 1.05   # 12.882
YMIN, YMAX = 12.0, 13.2
XMIN, XMAX = 28.0, 51.0        # X-axis inverted below

plt.rcParams.update({"font.size": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.set_xlim(XMAX, XMIN)        # inverted: high memory on left
ax.set_ylim(YMIN, YMAX)

# ── 5% budget line ────────────────────────────────────────────────────────────
ax.axhline(BUDGET_PPL, color="#C41230", linestyle="--",
           linewidth=1.6, zorder=2)
# Label on the far-right (low-memory side) to avoid overlap
ax.text(29.0, BUDGET_PPL + 0.06, "5% perplexity budget",
        fontsize=8.5, color="#C41230", va="bottom", ha="left")

# ── Plot dots and labels ──────────────────────────────────────────────────────
# Label positions: explicit (x, y) in data coords, ha, va
label_cfg = {
    "FP16 baseline":  dict(xy=(46.5, 12.269), tx=(47.2, 12.21),  ha="center", va="top"),
    "INT8KV only":    dict(xy=(41.7, 12.269), tx=(41.7, 12.32),  ha="center", va="bottom"),
    "W4A8 only":      dict(xy=(36.2, 12.706), tx=(37.5, 12.74),  ha="right",  va="bottom"),
    "W4A8 + INT8KV":  dict(xy=(31.5, 12.706), tx=(30.0, 12.76),  ha="right",  va="bottom"),
}

for name, mem, ppl, color in CONFIGS:
    cfg = label_cfg[name]
    ax.scatter(mem, ppl, color=color, s=140, zorder=5)
    tx, ty = cfg["tx"]
    ax.text(tx, ty, name,
            fontsize=9, color=color, fontweight="bold",
            ha=cfg["ha"], va=cfg["va"])

# Wanda: clipped triangle at top edge, simple label, no arrow
wname, wmem, _, wcolor = WANDA
clip_y = YMAX - 0.04
ax.scatter(wmem, clip_y, marker="^", color=wcolor, s=120, zorder=5, clip_on=False)
ax.text(wmem, clip_y - 0.08, wname,
        fontsize=8.5, color=wcolor, ha="center", va="top")

# ── "Better" direction arrow on the axes ─────────────────────────────────────
ax.annotate("", xy=(29.5, YMIN + 0.08), xytext=(49.5, YMIN + 0.08),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.1))
ax.text(39.5, YMIN + 0.18, "less memory  →", ha="center",
        fontsize=8, color="gray")

# ── Axes labels & title ───────────────────────────────────────────────────────
ax.set_xlabel("Peak GPU Memory at $B{=}16$, $N{=}4096$ (GB)", fontsize=10)
ax.set_ylabel("WikiText-103 Perplexity  (↓ lower is better)", fontsize=10)
ax.set_title(
    "Memory–Quality Tradeoff  (B=16, N=4096)\n",
    fontsize=10, fontweight="bold",
)
ax.yaxis.grid(True, alpha=0.3, zorder=0)

plt.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved {OUT}")
