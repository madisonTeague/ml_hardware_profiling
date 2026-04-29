import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ORANGE = "#FFD9A0"
BLUE   = "#C6DCFF"
GREEN  = "#C8F0C8"
GRAY   = "#888888"

def rounded_box(ax, cx, cy, w, h, color, text, fontsize=8.5, bold=False):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="black", linewidth=0.9, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        cx, cy, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        multialignment="center", zorder=3,
    )

def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        zorder=4,
    )

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.6)
ax.axis("off")

# ── dashed hardware bounding box ──────────────────────────────────────────────
hw_box = FancyBboxPatch(
    (0.25, 0.55), 10.5, 3.15,
    boxstyle="round,pad=0.1",
    facecolor="none", edgecolor=GRAY, linewidth=1.2, linestyle="dashed", zorder=1,
)
ax.add_patch(hw_box)
ax.text(5.5, 0.3, "NVIDIA A100 SXM 80 GB via Modal Cloud",
        ha="center", va="center", fontsize=7.5, color=GRAY)

# ── model node ────────────────────────────────────────────────────────────────
MODEL_Y = 4.05
rounded_box(ax, 5.5, MODEL_Y, 3.2, 0.65, ORANGE,
            "Qwen3-8B (FP16)  |  A100 SXM 80 GB",
            fontsize=9, bold=True)

# ── experiment (compression) nodes ───────────────────────────────────────────
EXP_Y = 2.65
EXP_W, EXP_H = 2.2, 1.0
exp_xs    = [1.35, 3.75, 7.25, 9.65]
exp_texts = [
    "Exp. 1\nINT8 KV Quant.\n(Triton kernels)",
    "Exp. 2\nW4A8 Weight\nQuant. (AWQ)",
    "Exp. 3\nWanda FFN\nPruning",
    "Exp. 4\nW4A8 + INT8KV\n(combined)",
]
for cx, txt in zip(exp_xs, exp_texts):
    rounded_box(ax, cx, EXP_Y, EXP_W, EXP_H, BLUE, txt, fontsize=8)

# ── metric nodes ──────────────────────────────────────────────────────────────
MET_Y = 1.12
MET_W, MET_H = 2.2, 0.9
met_texts = [
    "HBM traffic\nKernel speedup\nRoofline AI",
    "Latency, VRAM\nAttn/FFN share\nBottleneck shift",
    "Throughput, VRAM\nAttn/FFN share\nBottleneck shift",
    "Peak GPU mem.\nWikiText-103\nperplexity",
]
for cx, txt in zip(exp_xs, met_texts):
    rounded_box(ax, cx, MET_Y, MET_W, MET_H, GREEN, txt, fontsize=7.5)

# ── arrows: model → experiments ───────────────────────────────────────────────
for cx in exp_xs:
    arrow(ax, 5.5, MODEL_Y - 0.325, cx, EXP_Y + EXP_H / 2)

# ── arrows: experiments → metrics ────────────────────────────────────────────
for cx in exp_xs:
    arrow(ax, cx, EXP_Y - EXP_H / 2, cx, MET_Y + MET_H / 2)

plt.tight_layout(pad=0.3)
out = "figures/fig_system_overview.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")
