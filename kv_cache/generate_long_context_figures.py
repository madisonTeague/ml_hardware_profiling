"""
Figures for long-context results (Exp 1 extension + Exp B).

Outputs:
  figures/fig_lc_perplexity.png   — FP16 vs INT8KV perplexity at 4K–32K context
  figures/fig_lc_bottleneck.png   — Attn/FFN share vs context length (short + long)
  figures/fig_lc_amdahl.png       — Amdahl-projected E2E INT8KV speedup vs context
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE    = Path(__file__).parent
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)

RESULTS_DIR = HERE.parent / "results"

BLUE   = "#4878CF"
ORANGE = "#E87722"
GREEN  = "#59A14F"
RED    = "#E15759"
GRAY   = "#9E9E9E"
PURPLE = "#9467BD"

plt.rcParams.update({"font.size": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False})

# ── Load data ─────────────────────────────────────────────────────────────────

with open(RESULTS_DIR / "results_lc_perplexity.json") as f:
    ppl_data = json.load(f)

# Long-context bottleneck
import csv
bn_rows = []
with open(RESULTS_DIR / "results_lc_bottleneck.csv") as f:
    for row in csv.DictReader(f):
        bn_rows.append({k: float(v) if k not in ("dtype",) else v
                        for k, v in row.items()})

# Short-context baseline from Exp 2 (FP16, B=1)
# Values from the paper's Table tab:exp2_layers
short_ctx = [
    {"ctx_len": 512,  "attn_pct": 34.7, "ffn_pct": 27.9},
    {"ctx_len": 2048, "attn_pct": 31.3, "ffn_pct": 47.4},
    {"ctx_len": 4096, "attn_pct": 33.9, "ffn_pct": 48.8},
]

# ── Fig 1: Perplexity vs context length ───────────────────────────────────────

ctx_lens_ppl = [4096, 8192, 16384, 32768]
fp16_ppl  = [ppl_data[f"fp16_ctx{n}"]   for n in ctx_lens_ppl]
i8kv_ppl  = [ppl_data[f"int8kv_ctx{n}"] for n in ctx_lens_ppl]

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(ctx_lens_ppl, fp16_ppl,  "o-", color=BLUE,   lw=2, ms=7, label="FP16")
ax.plot(ctx_lens_ppl, i8kv_ppl,  "s--", color=ORANGE, lw=2, ms=7, label="INT8KV")

for x, y1, y2 in zip(ctx_lens_ppl, fp16_ppl, i8kv_ppl):
    delta = abs(y1 - y2)
    ax.annotate(f"Δ={delta:.3f}", xy=(x, (y1+y2)/2),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7.5, color=GRAY, va="center")

ax.set_xscale("log", base=2)
ax.set_xticks(ctx_lens_ppl)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v)//1024}K" if v >= 1024 else str(int(v))))
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("WikiText-103 Perplexity  (↓ lower is better)")
ax.set_title("INT8KV Perplexity at Long Context\n"
             "Quantization noise does not accumulate — FP16 ≈ INT8KV at 32K",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_lc_perplexity.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved fig_lc_perplexity.png")


# ── Fig 2: Attention/FFN bottleneck shift across all context lengths ───────────

# Combine short-context (Exp 2) and long-context (Exp B) data
all_ctx   = [r["ctx_len"] for r in short_ctx] + [r["ctx_len"] for r in bn_rows]
all_attn  = [r["attn_pct"] for r in short_ctx] + [r["attn_pct"] for r in bn_rows]
all_ffn   = [r["ffn_pct"]  for r in short_ctx] + [r["ffn_pct"]  for r in bn_rows]

# Mark the crossover point
crossover_ctx = None
for i in range(len(all_ctx) - 1):
    if all_attn[i] < all_ffn[i] and all_attn[i+1] > all_ffn[i+1]:
        crossover_ctx = (all_ctx[i] + all_ctx[i+1]) / 2

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(all_ctx, all_attn, "o-", color=BLUE,   lw=2.2, ms=7, label="Attention")
ax.plot(all_ctx, all_ffn,  "s--", color=ORANGE, lw=2.2, ms=7, label="FFN / MLP")
ax.axhline(50, color=GRAY, linestyle=":", lw=1.2, alpha=0.6)
ax.text(600, 50.8, "50%", fontsize=8, color=GRAY)

# Mark short-context vs long-context regions
ax.axvspan(512, 4096,  alpha=0.05, color=GRAY)
ax.axvspan(4096, 32768, alpha=0.05, color=BLUE)
ax.text(1200,  10, "Exp 2\n(short ctx)", fontsize=8, color=GRAY,   ha="center")
ax.text(14000, 10, "Exp B\n(long ctx)",  fontsize=8, color=BLUE,   ha="center")

# Annotate crossover
ax.annotate("Crossover\n~12–16K tokens",
            xy=(16384, 43.5), xytext=(7000, 30),
            fontsize=8.5, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# Annotate 32K values
ax.annotate(f"Attn: {all_attn[-1]:.1f}%",
            xy=(32768, all_attn[-1]),
            xytext=(-55, 8), textcoords="offset points",
            fontsize=8.5, color=BLUE, fontweight="bold")
ax.annotate(f"FFN: {all_ffn[-1]:.1f}%",
            xy=(32768, all_ffn[-1]),
            xytext=(-55, -14), textcoords="offset points",
            fontsize=8.5, color=ORANGE, fontweight="bold")

ax.set_xscale("log", base=2)
ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384, 32768])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v)//1024}K" if v >= 1024 else str(int(v))))
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Share of total runtime (%)")
ax.set_ylim(0, 75)
ax.set_title("Attention vs FFN Runtime Share — Qwen3-8B FP16 (B=1)\n"
             "Attention overtakes FFN at ~12–16K tokens; 55.7% at 32K",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_lc_bottleneck.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved fig_lc_bottleneck.png")


# ── Fig 3: Amdahl projection — E2E INT8KV speedup vs context ──────────────────
# INT8KV kernel speedup at long context: at large N, AI >> ridge point,
# so speedup saturates near the ~19-20% seen at N=4096 in Exp 1.
# We use 19% as a conservative estimate across all long-context lengths.

KERNEL_SPEEDUP = 1.19   # from Exp 1 at N=4096

# All context lengths (short from Exp 2, long from Exp B)
proj_ctx    = [r["ctx_len"] for r in short_ctx] + [r["ctx_len"] for r in bn_rows]
proj_attn_f = [r["attn_pct"] / 100 for r in short_ctx] + \
              [r["attn_pct"] / 100 for r in bn_rows]

def amdahl(f, s):
    return 1.0 / ((1 - f) + f / s)

proj_speedup = [(amdahl(f, KERNEL_SPEEDUP) - 1) * 100 for f in proj_attn_f]

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(proj_ctx, proj_speedup, "o-", color=GREEN, lw=2.2, ms=7,
        label=f"Projected E2E speedup\n(INT8KV kernel ×{KERNEL_SPEEDUP})")

# Annotate short vs long context
ax.axvspan(512, 4096,  alpha=0.05, color=GRAY)
ax.axvspan(4096, 32768, alpha=0.05, color=GREEN)
ax.text(1200, 0.3, "short ctx\n(Exp 2)",  fontsize=8, color=GRAY,  ha="center")
ax.text(16000, 0.3, "long ctx\n(Exp B)", fontsize=8, color=GREEN, ha="center")

for x, s, f in zip(proj_ctx, proj_speedup, proj_attn_f):
    ax.annotate(f"{s:.1f}%\n(attn={f*100:.0f}%)",
                xy=(x, s), xytext=(0, 10), textcoords="offset points",
                fontsize=7.5, ha="center", color="#333333")

ax.set_xscale("log", base=2)
ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384, 32768])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v)//1024}K" if v >= 1024 else str(int(v))))
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Projected full-model E2E speedup (%)")
ax.set_ylim(0, 14)
ax.set_title("Amdahl Projection: INT8KV E2E Speedup vs Context Length\n"
             "Benefit grows from 3–6% (short ctx) to ~10% (32K)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_lc_amdahl.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved fig_lc_amdahl.png")

print("\nDone.")
