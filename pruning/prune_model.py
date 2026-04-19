"""
Structured pruning of FFN layers for Experiment 3.

Three importance-scoring methods are available (--method):

  wanda  (default)
    Activation-aware, Wanda-style (Sun et al. 2023).
    gate_score[i] = sqrt( gate_proj.weight[i,:]^2  @  x_rms^2 )
    up_score[i]   = sqrt( up_proj.weight[i,:]^2    @  x_rms^2 )
    down_norm[i]  = || down_proj.weight[:, i] ||_2
    score[i]      = gate_score[i] * up_score[i] * down_norm[i]
    Requires a 64-sample calibration pass; set --calib-samples 0 to
    fall back to weight-only L2 norms.

  spectral
    PruneNet-style spectral preservation (Sengupta et al. ICLR 2025).
    Calibration-free.  Uses truncated SVD of up_proj (FFN1) to measure
    each neuron's contribution to the singular-value distribution:
    svd_score[i]  = || (U * S)[i, :] ||_2   (spectral leverage)
    score[i]      = svd_score[i] * down_norm[i]
    Keeps neurons whose removal least shifts the singular-value CDF
    (approximation of PruneNet's KS-distance minimisation).

  spectral+wanda
    Per-layer L2-normalised product of both scores:
    score[i] = norm(wanda)[i] * norm(spectral)[i] * down_norm[i]
    Combines activation relevance with spectral structure preservation.

No fine-tuning of any kind.

Usage:
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.2 --output ./pruned_20pct
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.05 0.1 0.2 0.3
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.1 --method spectral
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.1 --method spectral+wanda
    python prune_model.py --model Qwen/Qwen3-8B --ratio 0.2 --calib-samples 0
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
# Calibration: collect MLP input activation statistics
# ------------------------------------------------------------------

def collect_mlp_input_stats(
    model,
    tokenizer,
    n_samples: int = 64,
    seq_len: int = 512,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
) -> dict[int, torch.Tensor]:
    """Collect per-feature RMS of MLP inputs across calibration samples.

    Runs *n_samples* non-overlapping *seq_len*-token chunks from the
    WikiText-103 train split through the model with no gradients.  For
    each FFN layer, records the per-input-feature RMS of the
    post-LayerNorm hidden states that feed into gate_proj.

    Uses online accumulation of sum-of-squares so peak extra memory is
    just one [hidden_size] float64 vector per layer — negligible vs the
    model weights.

    Returns
    -------
    dict[layer_idx -> Tensor[hidden_size]]
        Per-layer RMS on the same device as the model weights.
    """
    from datasets import load_dataset

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    layers = list(_iter_decoder_layers(model))
    n_layers = len(layers)

    sq_sum = [torch.zeros(hidden_size, dtype=torch.float64) for _ in range(n_layers)]
    token_count = [0] * n_layers

    handles = []
    for idx, layer in enumerate(layers):
        def _make_hook(i):
            def _pre(module, inp):
                # inp[0]: [batch, seq, hidden_size]
                x = inp[0].detach().float().reshape(-1, hidden_size)
                sq_sum[i].add_(x.pow(2).sum(dim=0).cpu())
                token_count[i] += x.shape[0]
            return _pre
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(_make_hook(idx)))

    # We only need n_samples*seq_len tokens. WikiText-103 averages ~150
    # tokens/article, so loading n_samples*8 rows is a safe over-provision.
    # Using a slice avoids tokenizing the full 1.8M-row train split.
    max_rows = n_samples * 8
    print(f"  Calibration: loading {dataset_name}/{dataset_config} "
          f"(first {max_rows} rows) ...")
    dataset = load_dataset(dataset_name, dataset_config,
                           split=f"train[:{max_rows}]",
                           trust_remote_code=True)
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids  # [1, total_tokens]

    print(f"  Calibration: running {n_samples} × {seq_len}-token forward passes ...")
    n_done = 0
    with torch.no_grad():
        for start in range(0, input_ids.shape[1] - seq_len + 1, seq_len):
            chunk = input_ids[:, start : start + seq_len].to(device)
            model(chunk)
            n_done += 1
            if n_done >= n_samples:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for h in handles:
        h.remove()

    x_rms: dict[int, torch.Tensor] = {}
    for i in range(n_layers):
        if token_count[i] > 0:
            x_rms[i] = (sq_sum[i] / token_count[i]).float().sqrt().to(device)

    print(f"  Calibration: {token_count[0]} tokens × {n_layers} layers collected.")
    return x_rms


# ------------------------------------------------------------------
# Importance scoring
# ------------------------------------------------------------------

def _spectral_scores_for_layer(mlp, n_sv: int = 256) -> torch.Tensor:
    """Compute PruneNet-style spectral leverage scores for one FFN layer.

    Uses truncated SVD of up_proj (FFN1) — the matrix PruneNet targets.
    For neuron i:

        svd_score[i] = || (U * S)[i, :] ||_2
                     = sqrt( sum_k  S[k]^2 * U[i,k]^2 )

    This measures how much row i drives the dominant singular directions,
    approximating PruneNet's KS-distance minimisation criterion without
    any training.  torch.svd_lowrank (randomised) keeps runtime fast
    even for large [intermediate, hidden] matrices.

    Returns a float32 CPU tensor of shape [intermediate_size].
    """
    W = mlp.up_proj.weight.float()          # [intermediate, hidden]
    q = min(n_sv, min(W.shape))
    U, S, _Vh = torch.svd_lowrank(W, q=q, niter=4)
    # U: [intermediate, q], S: [q]
    US = U * S.unsqueeze(0)                 # [intermediate, q]
    return US.pow(2).sum(dim=1).sqrt().cpu()  # [intermediate]


def compute_neuron_importance(
    model,
    x_rms: dict[int, torch.Tensor] | None = None,
    method: str = "wanda",
    use_triton: bool = True,
    n_sv: int = 256,
) -> dict[int, torch.Tensor]:
    """Return per-neuron importance scores for every FFN layer.

    method="wanda"  (activation-aware, recommended with calibration data)
        gate_score[i] = sqrt( gate_proj.weight[i,:]^2  @  x_rms^2 )
        up_score[i]   = sqrt( up_proj.weight[i,:]^2    @  x_rms^2 )
        down_norm[i]  = || down_proj.weight[:, i] ||_2
        score[i]      = gate_score[i] * up_score[i] * down_norm[i]
        Falls back to weight-only L2 norms if x_rms is None.

    method="spectral"  (PruneNet-style, calibration-free)
        svd_score[i] = || (U * S)[i, :] ||_2  from up_proj truncated SVD
        score[i]     = svd_score[i] * down_norm[i]

    method="spectral+wanda"
        Per-layer L2-normalised product of both scores:
        score[i] = norm(wanda)[i] * norm(spectral)[i] * down_norm[i]

    Returns a dict  {layer_index: Tensor[intermediate_size]}  on CPU.
    """
    if method not in ("wanda", "spectral", "spectral+wanda"):
        raise ValueError(f"Unknown method {method!r}. "
                         "Choose 'wanda', 'spectral', or 'spectral+wanda'.")

    do_triton = use_triton and HAS_TRITON and torch.cuda.is_available()
    importance: dict[int, torch.Tensor] = {}

    for idx, layer in enumerate(_iter_decoder_layers(model)):
        mlp = layer.mlp

        # down_proj column norms are shared across all methods
        if do_triton:
            down_norm = _triton_col_l2(mlp.down_proj.weight).cpu()
        else:
            down_norm = mlp.down_proj.weight.float().norm(dim=0).cpu()

        # ---- Wanda scores (activation-aware or weight-only fallback) ----
        wanda_score: torch.Tensor | None = None
        if method in ("wanda", "spectral+wanda"):
            if x_rms is not None and idx in x_rms:
                rms_sq     = x_rms[idx].float().pow(2)
                gate_score = mlp.gate_proj.weight.float().pow(2).mv(rms_sq).sqrt().cpu()
                up_score   = mlp.up_proj.weight.float().pow(2).mv(rms_sq).sqrt().cpu()
            else:
                if do_triton:
                    gate_score = _triton_row_l2(mlp.gate_proj.weight).cpu()
                    up_score   = _triton_row_l2(mlp.up_proj.weight).cpu()
                else:
                    gate_score = mlp.gate_proj.weight.float().norm(dim=1).cpu()
                    up_score   = mlp.up_proj.weight.float().norm(dim=1).cpu()
            wanda_score = gate_score * up_score  # [intermediate]

        # ---- Spectral scores (SVD leverage, calibration-free) ----
        spectral_score: torch.Tensor | None = None
        if method in ("spectral", "spectral+wanda"):
            spectral_score = _spectral_scores_for_layer(mlp, n_sv=n_sv)

        # ---- Combine ----
        if method == "wanda":
            assert wanda_score is not None
            importance[idx] = wanda_score * down_norm

        elif method == "spectral":
            assert spectral_score is not None
            importance[idx] = spectral_score * down_norm

        else:  # spectral+wanda
            assert wanda_score is not None and spectral_score is not None
            # Normalise each to unit L2 per layer before multiplying so
            # neither score dominates purely due to scale differences.
            w_norm = wanda_score / (wanda_score.norm() + 1e-8)
            s_norm = spectral_score / (spectral_score.norm() + 1e-8)
            importance[idx] = w_norm * s_norm * down_norm

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
    """Replace gate/up/down projections in-place with pruned versions.

    Qwen FFN path is:
        hidden -> gate_proj/up_proj -> elementwise(SwiGLU) -> down_proj -> hidden

    Structured pruning removes the same intermediate neuron indices from:
    - rows of gate_proj
    - rows of up_proj
    - columns of down_proj
    so tensor dimensions stay consistent for the fused FFN block.
    """
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
                method: str = "wanda",
                n_calib_samples: int = 64,
                n_sv: int = 256,
                wandb_project: str | None = None) -> tuple:
    """Load, prune, save.  Returns (model, tokenizer, gemm_rows).

    Parameters
    ----------
    method:
        Importance scoring strategy: "wanda" | "spectral" | "spectral+wanda".
    n_calib_samples:
        WikiText-103 chunks for activation calibration (wanda / spectral+wanda).
        Set to 0 to use weight-only norms.  Ignored for method="spectral".
    n_sv:
        Number of singular values to use in truncated SVD (spectral paths).
    wandb_project:
        If set (or WANDB_PROJECT env var), initialises a W&B run.
    """
    import os
    wb_project = wandb_project or os.environ.get("WANDB_PROJECT")
    if HAS_WANDB and wb_project and wandb.run is None:
        wandb.init(
            project=wb_project,
            name=f"prune-{int(prune_ratio*100)}pct-{method}",
            config={
                "model": model_name,
                "prune_ratio": prune_ratio,
                "method": method,
                "scope": "ffn_only",
                "n_calib_samples": n_calib_samples if method != "spectral" else 0,
                "n_sv": n_sv,
            },
        )

    print(f"\n{'='*60}")
    print(f"PRUNING  model={model_name}  ratio={prune_ratio}  method={method}")
    print(f"{'='*60}")

    # 1) Load baseline model/tokenizer checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch_dtype, device_map="auto",
    )
    model.eval()

    orig_inter = _original_intermediate(model)
    n_keep = _align_to_tile(int(round(orig_inter * (1 - prune_ratio))))
    print(f"  intermediate_size: {orig_inter} -> {n_keep}  "
          f"(removing {orig_inter - n_keep} neurons per layer)")

    # 2) Gather activation statistics when required by the chosen method.
    use_triton = HAS_TRITON and torch.cuda.is_available()
    x_rms = None
    needs_calib = method in ("wanda", "spectral+wanda") and n_calib_samples > 0

    if method == "spectral":
        print(f"  Importance: spectral / PruneNet-style  (n_sv={n_sv}, calibration-free)")
    elif needs_calib:
        print(f"  Importance: {method}  ({n_calib_samples} calib samples, n_sv={n_sv})")
        try:
            x_rms = collect_mlp_input_stats(
                model, tokenizer,
                n_samples=n_calib_samples,
                seq_len=512,
            )
        except Exception as e:
            print(f"  WARNING: calibration failed ({e}), falling back to weight-only.")
            x_rms = None
    else:
        print(f"  Importance: {method} weight-only  "
              f"({'Triton' if use_triton else 'PyTorch'})")

    # 3) Score each FFN intermediate neuron.
    print("  Computing neuron importance scores ...")
    importance = compute_neuron_importance(
        model, x_rms=x_rms, method=method,
        use_triton=use_triton, n_sv=n_sv,
    )
    _wandb_log_importance(importance, prune_ratio)

    # 4) Physically shrink FFN projection matrices in every decoder block.
    print("  Pruning FFN layers ...")
    for idx, layer in enumerate(_iter_decoder_layers(model)):
        keep = _select_keep_indices(importance[idx], prune_ratio)
        prune_ffn_layer(layer, keep)

    # 5) Log matrix-shape changes for hardware analysis and update config.
    gemm_rows = log_gemm_shapes(model, prune_ratio)
    _wandb_log_gemm(gemm_rows, prune_ratio)

    model.config.intermediate_size = n_keep
    _wandb_log_prune_summary(model, prune_ratio, orig_inter)

    # 6) Save a standard HF checkpoint so later profiling can reload directly.
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
        description="Structured FFN pruning — wanda / spectral / spectral+wanda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model name")
    parser.add_argument("--ratio", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3],
                        help="Pruning ratio(s) as fractions, e.g. 0.05 0.1 0.2 0.3")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (auto-generated if omitted)")
    parser.add_argument("--method", type=str, default="wanda",
                        choices=["wanda", "spectral", "spectral+wanda"],
                        help="Importance scoring method (default: wanda)")
    parser.add_argument("--calib-samples", type=int, default=64,
                        help="WikiText-103 chunks for activation calibration "
                             "(0 = weight-only; ignored for --method spectral)")
    parser.add_argument("--n-sv", type=int, default=256,
                        help="Singular values to use in truncated SVD "
                             "(spectral / spectral+wanda paths, default 256)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (enables logging)")
    args = parser.parse_args()

    all_gemm_rows = []
    for ratio in args.ratio:
        out_dir = args.output or f"./pruned_{int(ratio * 100)}pct"
        _, _, gemm_rows = prune_model(args.model, ratio, out_dir,
                                      method=args.method,
                                      n_calib_samples=args.calib_samples,
                                      n_sv=args.n_sv,
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
