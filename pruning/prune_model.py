"""
Structured pruning of FFN layers for Experiment 3.

Magnitude-based neuron importance scoring followed by physical removal
of the least-important intermediate neurons from gate_proj, up_proj,
and down_proj.  Produces a valid HuggingFace checkpoint with a smaller
intermediate_size that can be loaded with AutoModelForCausalLM.

No training, no calibration data, no fine-tuning of any kind.

Usage:
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.2 --output ./pruned_20pct
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.1 0.2 0.3
"""

import argparse
import csv
import gc
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ------------------------------------------------------------------
# Triton fused importance kernel
# ------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _row_l2_norm_kernel(
        X_ptr, Out_ptr,
        N_ROWS, N_COLS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    ):
        """Compute L2 norm of each row: Out[row] = ||X[row, :]||_2."""
        row = tl.program_id(0)
        if row >= N_ROWS:
            return
        acc = tl.zeros([BLOCK_COLS], dtype=tl.float32)
        for start in range(0, N_COLS, BLOCK_COLS):
            cols = start + tl.arange(0, BLOCK_COLS)
            mask = cols < N_COLS
            x = tl.load(X_ptr + row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        tl.store(Out_ptr + row, tl.sqrt(tl.sum(acc)))

    @triton.jit
    def _col_l2_norm_kernel(
        X_ptr, Out_ptr,
        N_ROWS, N_COLS, BLOCK_ROWS: tl.constexpr,
    ):
        """Compute L2 norm of each column: Out[col] = ||X[:, col]||_2."""
        col = tl.program_id(0)
        if col >= N_COLS:
            return
        acc = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        for start in range(0, N_ROWS, BLOCK_ROWS):
            rows = start + tl.arange(0, BLOCK_ROWS)
            mask = rows < N_ROWS
            x = tl.load(X_ptr + rows * N_COLS + col, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        tl.store(Out_ptr + col, tl.sqrt(tl.sum(acc)))


def _triton_row_l2(weight: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 norm via Triton."""
    w = weight.contiguous()
    n_rows, n_cols = w.shape
    out = torch.empty(n_rows, device=w.device, dtype=torch.float32)
    BLOCK_COLS = triton.next_power_of_2(min(n_cols, 4096))
    _row_l2_norm_kernel[(n_rows,)](w, out, n_rows, n_cols, BLOCK_COLS)
    return out


def _triton_col_l2(weight: torch.Tensor) -> torch.Tensor:
    """Column-wise L2 norm via Triton."""
    w = weight.contiguous()
    n_rows, n_cols = w.shape
    out = torch.empty(n_cols, device=w.device, dtype=torch.float32)
    BLOCK_ROWS = triton.next_power_of_2(min(n_rows, 4096))
    _col_l2_norm_kernel[(n_cols,)](w, out, n_rows, n_cols, BLOCK_ROWS)
    return out


# ------------------------------------------------------------------
# Importance scoring
# ------------------------------------------------------------------

def compute_neuron_importance(model, use_triton: bool = True) -> dict[int, torch.Tensor]:
    """Return per-neuron importance scores for every FFN layer.

    For intermediate neuron *i* in layer *l* the score is:

        ||gate_proj.weight[i, :]||_2
      + ||up_proj.weight[i, :]||_2
      + ||down_proj.weight[:, i]||_2

    When Triton is available and use_triton=True, norms are computed
    with fused Triton kernels (single pass per matrix, fewer kernel
    launches than PyTorch's .norm()).

    Returns a dict  {layer_index: Tensor[intermediate_size]}  on CPU.
    """
    do_triton = use_triton and HAS_TRITON and torch.cuda.is_available()
    importance: dict[int, torch.Tensor] = {}

    for idx, layer in enumerate(_iter_decoder_layers(model)):
        mlp = layer.mlp
        if do_triton:
            gate_norm = _triton_row_l2(mlp.gate_proj.weight)
            up_norm = _triton_row_l2(mlp.up_proj.weight)
            down_norm = _triton_col_l2(mlp.down_proj.weight)
        else:
            gate_norm = mlp.gate_proj.weight.float().norm(dim=1)
            up_norm = mlp.up_proj.weight.float().norm(dim=1)
            down_norm = mlp.down_proj.weight.float().norm(dim=0)

        importance[idx] = (gate_norm + up_norm + down_norm).cpu()

    return importance


# ------------------------------------------------------------------
# Structural pruning
# ------------------------------------------------------------------

_TILE = 128  # Tensor Core tile width on A100/H100; keeps cuBLAS happy


def _align_to_tile(n: int, tile: int = _TILE) -> int:
    """Round *n* down to the nearest multiple of *tile*.

    cuBLAS kernel selection degrades sharply for odd or non-aligned
    intermediate dimensions (e.g. 11059 is prime -> ~2x slower than
    the aligned equivalent 11008).
    """
    return max(tile, (n // tile) * tile)


def _select_keep_indices(scores: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """Return sorted indices of neurons to *keep*."""
    n_total = scores.shape[0]
    n_keep = _align_to_tile(int(round(n_total * (1 - prune_ratio))))
    _, sorted_idx = scores.sort(descending=True)
    keep = sorted_idx[:n_keep].sort().values
    return keep


def _make_pruned_linear(old: nn.Linear, keep_idx: torch.Tensor,
                        dim: int) -> nn.Linear:
    """Build a new nn.Linear by selecting rows (dim=0) or columns (dim=1)."""
    weight = old.weight.data
    if dim == 0:
        weight = weight[keep_idx]
    else:
        weight = weight[:, keep_idx]

    new = nn.Linear(weight.shape[1], weight.shape[0],
                    bias=old.bias is not None, device=weight.device,
                    dtype=weight.dtype)
    new.weight.data.copy_(weight)

    if old.bias is not None:
        if dim == 0:
            new.bias.data.copy_(old.bias.data[keep_idx])
        else:
            new.bias.data.copy_(old.bias.data)
    return new


def prune_ffn_layer(layer, keep_indices: torch.Tensor):
    """Replace gate/up/down projections in-place with pruned versions."""
    mlp = layer.mlp
    device_idx = keep_indices.to(mlp.gate_proj.weight.device)

    mlp.gate_proj = _make_pruned_linear(mlp.gate_proj, device_idx, dim=0)
    mlp.up_proj = _make_pruned_linear(mlp.up_proj, device_idx, dim=0)
    mlp.down_proj = _make_pruned_linear(mlp.down_proj, device_idx, dim=1)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _iter_decoder_layers(model):
    """Yield decoder layers regardless of wrapper (bare vs CausalLM)."""
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers
    return layers


def _original_intermediate(model) -> int:
    return model.config.intermediate_size


# ------------------------------------------------------------------
# GEMM shape logging
# ------------------------------------------------------------------

def log_gemm_shapes(model, prune_ratio: float) -> list[dict]:
    """Record original vs current GEMM dimensions for every FFN projection."""
    rows = []
    orig_inter = model.config.intermediate_size
    hidden = model.config.hidden_size

    for idx, layer in enumerate(_iter_decoder_layers(model)):
        mlp = layer.mlp
        for name, proj in [("gate_proj", mlp.gate_proj),
                           ("up_proj", mlp.up_proj),
                           ("down_proj", mlp.down_proj)]:
            w = proj.weight
            rows.append({
                "layer": idx,
                "projection": name,
                "prune_ratio": prune_ratio,
                "original_shape": f"[{orig_inter}, {hidden}]" if name != "down_proj"
                                  else f"[{hidden}, {orig_inter}]",
                "pruned_shape": f"[{w.shape[0]}, {w.shape[1]}]",
                "original_elements": orig_inter * hidden,
                "pruned_elements": w.shape[0] * w.shape[1],
            })
    return rows


# ------------------------------------------------------------------
# W&B logging helpers
# ------------------------------------------------------------------

def _wandb_log_importance(importance: dict[int, torch.Tensor], prune_ratio: float):
    """Log per-layer importance score distributions to W&B."""
    if not HAS_WANDB or wandb.run is None:
        return

    all_scores = torch.cat(list(importance.values()))
    wandb.log({
        f"importance/global_histogram": wandb.Histogram(all_scores.numpy()),
        f"importance/global_mean": all_scores.mean().item(),
        f"importance/global_std": all_scores.std().item(),
    })

    table = wandb.Table(columns=["layer", "mean", "std", "min", "max", "median"])
    for idx, scores in importance.items():
        table.add_data(
            idx,
            scores.mean().item(),
            scores.std().item(),
            scores.min().item(),
            scores.max().item(),
            scores.median().item(),
        )
    wandb.log({f"importance/per_layer_stats": table})


def _wandb_log_gemm(gemm_rows: list[dict], prune_ratio: float):
    """Log GEMM shape table and compression chart to W&B."""
    if not HAS_WANDB or wandb.run is None:
        return

    table = wandb.Table(
        columns=["layer", "projection", "original_shape", "pruned_shape",
                 "original_elements", "pruned_elements", "compression_pct"],
    )
    for r in gemm_rows:
        comp_pct = (1 - r["pruned_elements"] / r["original_elements"]) * 100
        table.add_data(
            r["layer"], r["projection"], r["original_shape"],
            r["pruned_shape"], r["original_elements"],
            r["pruned_elements"], round(comp_pct, 2),
        )
    wandb.log({"gemm_shapes": table})

    gate_rows = [r for r in gemm_rows if r["projection"] == "gate_proj"]
    if gate_rows:
        wandb.log({
            "gemm/pruned_elements_per_layer": wandb.plot.line_series(
                xs=[r["layer"] for r in gate_rows],
                ys=[[r["pruned_elements"] / 1e6 for r in gate_rows]],
                keys=["gate_proj (M params)"],
                title=f"FFN Params per Layer (pruned {int(prune_ratio*100)}%)",
                xname="Layer",
            )
        })


def _wandb_log_prune_summary(model, prune_ratio: float, orig_inter: int):
    """Log final model summary to W&B."""
    if not HAS_WANDB or wandb.run is None:
        return

    param_count = sum(p.numel() for p in model.parameters())
    new_inter = model.config.intermediate_size
    wandb.log({
        "prune/ratio": prune_ratio,
        "prune/original_intermediate_size": orig_inter,
        "prune/pruned_intermediate_size": new_inter,
        "prune/neurons_removed_per_layer": orig_inter - new_inter,
        "prune/total_params_B": param_count / 1e9,
    })
    wandb.summary["prune_ratio"] = prune_ratio
    wandb.summary["total_params_B"] = param_count / 1e9


# ------------------------------------------------------------------
# End-to-end prune + save
# ------------------------------------------------------------------

def prune_model(model_name: str, prune_ratio: float, output_dir: str,
                torch_dtype=torch.float16,
                wandb_project: str | None = None) -> tuple:
    """Load, prune, save.  Returns (model, tokenizer, gemm_rows).

    If wandb_project is set (or WANDB_PROJECT env var), initialises
    a W&B run and logs importance distributions, GEMM shapes and
    model summary.
    """
    import os
    wb_project = wandb_project or os.environ.get("WANDB_PROJECT")
    if HAS_WANDB and wb_project and wandb.run is None:
        wandb.init(
            project=wb_project,
            name=f"prune-{int(prune_ratio*100)}pct",
            config={
                "model": model_name,
                "prune_ratio": prune_ratio,
                "method": "magnitude_l2_structured",
                "scope": "ffn_only",
            },
        )

    print(f"\n{'='*60}")
    print(f"PRUNING  model={model_name}  ratio={prune_ratio}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch_dtype, device_map="auto",
        rope_scaling={"type": "dynamic", "factor": 3.0}
    )
    model.eval()

    orig_inter = _original_intermediate(model)
    n_keep = _align_to_tile(int(round(orig_inter * (1 - prune_ratio))))
    print(f"  intermediate_size: {orig_inter} -> {n_keep}  "
          f"(removing {orig_inter - n_keep} neurons per layer)")

    use_triton = HAS_TRITON and torch.cuda.is_available()
    print(f"  Importance backend: {'Triton' if use_triton else 'PyTorch'}")
    print("  Computing neuron importance scores ...")
    importance = compute_neuron_importance(model, use_triton=use_triton)
    _wandb_log_importance(importance, prune_ratio)

    print("  Pruning FFN layers ...")
    for idx, layer in enumerate(_iter_decoder_layers(model)):
        keep = _select_keep_indices(importance[idx], prune_ratio)
        prune_ffn_layer(layer, keep)

    gemm_rows = log_gemm_shapes(model, prune_ratio)
    _wandb_log_gemm(gemm_rows, prune_ratio)

    model.config.intermediate_size = n_keep
    _wandb_log_prune_summary(model, prune_ratio, orig_inter)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  Saving pruned model to {out} ...")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    gemm_csv = out / "gemm_shapes.csv"
    _write_gemm_csv(gemm_rows, gemm_csv)
    if HAS_WANDB and wandb.run is not None:
        wandb.save(str(gemm_csv))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters after pruning: {param_count / 1e9:.3f}B")
    print(f"  Saved to {out}")

    return model, tokenizer, gemm_rows


def _write_gemm_csv(rows: list[dict], path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  GEMM shapes written to {path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structured FFN pruning (magnitude-based, no retraining)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model name")
    parser.add_argument("--ratio", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3],
                        help="Pruning ratio(s)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (auto-generated if omitted)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (enables logging)")
    args = parser.parse_args()

    all_gemm_rows = []
    for ratio in args.ratio:
        out_dir = args.output or f"./pruned_{int(ratio * 100)}pct"
        _, _, gemm_rows = prune_model(args.model, ratio, out_dir,
                                      wandb_project=args.wandb_project)
        all_gemm_rows.extend(gemm_rows)

        if HAS_WANDB and wandb.run is not None:
            wandb.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(args.ratio) > 1:
        combined = Path("gemm_shapes_all.csv")
        _write_gemm_csv(all_gemm_rows, combined)


if __name__ == "__main__":
    main()
