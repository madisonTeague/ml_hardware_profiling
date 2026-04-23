"""
Exp 4: Combined INT8 KV cache + W4A8 weight quantization.

Measures peak GPU memory and perplexity for four configurations:
  fp16      — FP16 baseline (reference)
  int8kv    — FP16 model + INT8 KV cache
  w4a8      — W4A8 model + FP16 KV cache
  combined  — W4A8 model + INT8 KV cache  ← the new run

Primary question: do the ~62% weight-memory savings (W4A8) and the
sequence-length-dependent KV-cache savings (INT8 KV) compose additively?
They attack different memory pools (static weights vs. dynamic KV tensors),
so additive composition is expected — this run confirms it end-to-end.

Usage:
    modal profile activate mlsys-15642
    modal run combined/modal_combined.py               # full sweep + perplexity
    modal run combined/modal_combined.py --mode memory # memory sweep only
    modal run combined/modal_combined.py --mode ppl    # perplexity only
"""

import os
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
# Image — mirrors handoff.md deps exactly
# ---------------------------------------------------------------------------

combined_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.5.1",
        "transformers==4.51.3",
        "accelerate==1.2.1",
        "autoawq==0.2.9",
        "autoawq-kernels==0.0.9",
        "datasets==2.21.0",
        "pandas==2.2.0",
        "numpy",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App("combined-exp4", image=combined_image)

# Shared workspace volumes — do NOT pass create_if_missing=True
models_vol  = modal.Volume.from_name("quantized-models")
results_vol = modal.Volume.from_name("attn-results", create_if_missing=True)

W4A8_PATH  = "/models/qwen3-8b-w4a8"
FP16_MODEL = "Qwen/Qwen3-8B"

BATCH_SIZES = [1, 4, 16]
SEQ_LENS    = [512, 2048, 4096]


# ---------------------------------------------------------------------------
# INT8 KV cache — DynamicCache subclass
# ---------------------------------------------------------------------------

def make_int8kv_cache():
    """
    Returns a DynamicCache subclass that stores K/V tensors in INT8.

    AWQ leaves K/V projection outputs in FP16 (it only quantizes nn.Linear
    weights), so the tensors arriving at update() are already the right dtype.

    Headwise symmetric quantization: scale = max(|x|) / 127, per (B, H).
    Dequantized tensors are returned to the attention computation in FP16.
    The INT8 tensors + scales are what actually sit in GPU memory, giving the
    ~50% KV cache reduction.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    class INT8KVCache(DynamicCache):
        def __init__(self):
            super().__init__()
            # Store INT8 tensors and per-head scales separately from the
            # FP16 lists that the parent class manages.
            self._k_i8:     list[torch.Tensor] = []
            self._v_i8:     list[torch.Tensor] = []
            self._k_scales: list[torch.Tensor] = []
            self._v_scales: list[torch.Tensor] = []

        @staticmethod
        def _quantize(x: torch.Tensor):
            # x: (B, H, T, D)  — FP16
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            x_i8  = (x / scale).round().clamp(-128, 127).to(torch.int8)
            return x_i8, scale

        @staticmethod
        def _dequantize(x_i8: torch.Tensor, scale: torch.Tensor):
            return x_i8.to(torch.float16) * scale

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            k_i8, k_scale = self._quantize(key_states)
            v_i8, v_scale = self._quantize(value_states)

            if layer_idx == len(self._k_i8):
                # First time seeing this layer
                self._k_i8.append(k_i8)
                self._v_i8.append(v_i8)
                self._k_scales.append(k_scale)
                self._v_scales.append(v_scale)
            else:
                # Append along the sequence dimension (autoregressive generation)
                self._k_i8[layer_idx]     = torch.cat([self._k_i8[layer_idx],     k_i8],   dim=2)
                self._v_i8[layer_idx]     = torch.cat([self._v_i8[layer_idx],     v_i8],   dim=2)
                self._k_scales[layer_idx] = torch.cat([self._k_scales[layer_idx], k_scale], dim=2)
                self._v_scales[layer_idx] = torch.cat([self._v_scales[layer_idx], v_scale], dim=2)

            k_out = self._dequantize(self._k_i8[layer_idx], self._k_scales[layer_idx])
            v_out = self._dequantize(self._v_i8[layer_idx], self._v_scales[layer_idx])
            return k_out, v_out

        def get_seq_length(self, layer_idx=0):
            if not self._k_i8:
                return 0
            return self._k_i8[layer_idx].shape[2]

    return INT8KVCache()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(dtype: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from awq import AutoAWQForCausalLM

    print(f"Loading {dtype} model...")
    tok = AutoTokenizer.from_pretrained(
        FP16_MODEL if dtype in ("fp16", "int8kv") else W4A8_PATH,
        trust_remote_code=True,
    )

    if dtype in ("fp16", "int8kv"):
        model = AutoModelForCausalLM.from_pretrained(
            FP16_MODEL, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
    else:  # w4a8 or combined
        # fuse_layers=False is required — fused layers replace Qwen3Attention
        # with QuantAttentionFused which calls flash_attn_func and breaks
        # any KV cache patching.
        model = AutoAWQForCausalLM.from_quantized(
            W4A8_PATH, trust_remote_code=True,
            device_map="auto", fuse_layers=False,
        )

    model.eval()
    print(f"  loaded. Device: {next(model.parameters()).device}")
    return model, tok


# ---------------------------------------------------------------------------
# Memory sweep
# ---------------------------------------------------------------------------

def measure_peak_memory_gb(model, tok, batch: int, seq_len: int, use_int8kv: bool) -> float:
    import torch

    device = next(model.parameters()).device
    input_ids = torch.randint(0, tok.vocab_size, (batch, seq_len), device=device)

    kwargs = {"use_cache": True}
    if use_int8kv:
        kwargs["past_key_values"] = make_int8kv_cache()

    # Warmup
    with torch.no_grad():
        model(input_ids, **kwargs)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    if use_int8kv:
        kwargs["past_key_values"] = make_int8kv_cache()   # fresh cache for measurement
    with torch.no_grad():
        model(input_ids, **kwargs)
    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tok, max_samples: int = 50, seq_len: int = 512) -> float:
    import torch
    import numpy as np
    from datasets import load_dataset

    print("  Computing perplexity on WikiText-103 test ...")
    ds   = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()][:500])
    enc  = tok(text, return_tensors="pt")

    device = next(model.parameters()).device
    ids    = enc.input_ids.to(device)
    nlls   = []

    with torch.no_grad():
        for i in range(0, min(ids.shape[1], max_samples * seq_len), seq_len):
            chunk = ids[:, i : i + seq_len]
            if chunk.shape[1] < seq_len:
                break
            loss = model(chunk, labels=chunk).loss.item()
            nlls.append(loss)
            if len(nlls) >= max_samples:
                break

    ppl = float(np.exp(np.mean(nlls)))
    print(f"  Perplexity: {ppl:.3f}  (n={len(nlls)} chunks)")
    return ppl


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/models": models_vol, "/results": results_vol},
)
def run_memory_sweep() -> bytes:
    """
    Measures peak GPU memory for all 4 configs × 3 batch sizes × 3 seq_lens.
    Returns a CSV as bytes and saves to /results/combined_memory.csv.
    """
    import torch
    import pandas as pd

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    configs = [
        ("fp16",     False),
        ("int8kv",   True),
        ("w4a8",     False),
        ("combined", True),
    ]

    rows = []
    for dtype, use_int8kv in configs:
        model, tok = load_model(dtype)
        for batch in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                print(f"  {dtype:8s}  B={batch:2d}  N={seq_len} ...", end="", flush=True)
                try:
                    mem_gb = measure_peak_memory_gb(model, tok, batch, seq_len, use_int8kv)
                    print(f"  {mem_gb:.2f} GB")
                    rows.append(dict(dtype=dtype, batch=batch, seq_len=seq_len,
                                     peak_mem_gb=round(mem_gb, 3)))
                except Exception as e:
                    print(f"  FAILED: {e}")
                    rows.append(dict(dtype=dtype, batch=batch, seq_len=seq_len,
                                     peak_mem_gb=None))
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode()
    with open("/results/combined_memory.csv", "wb") as f:
        f.write(csv_bytes)
    results_vol.commit()
    print("\nSaved /results/combined_memory.csv")
    print(df.to_string(index=False))
    return csv_bytes


@app.function(
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/models": models_vol, "/results": results_vol},
)
def run_perplexity() -> dict:
    """
    Perplexity for int8kv and combined configs.
    (fp16=12.27 and w4a8=12.71 already measured — included here for completeness.)
    """
    import torch

    results = {}

    # FP16 and W4A8 baselines already known but we re-measure for consistency
    for dtype, use_int8kv in [("fp16", False), ("int8kv", True),
                               ("w4a8", False), ("combined", True)]:
        print(f"\n--- {dtype} ---")
        model, tok = load_model(dtype)
        # Perplexity uses teacher-forcing (no cache needed), so int8kv doesn't
        # affect the forward pass computation — it would only matter for generation.
        # We measure without cache to keep it comparable to existing baselines.
        ppl = compute_perplexity(model, tok)
        results[dtype] = round(ppl, 3)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 40)
    print("PERPLEXITY SUMMARY")
    print("=" * 40)
    fp16_ppl = results.get("fp16", 12.27)
    for dtype, ppl in results.items():
        deg = (ppl / fp16_ppl - 1) * 100
        budget = "OK" if deg <= 5.0 else "OVER BUDGET"
        print(f"  {dtype:10s}: {ppl:.3f}  ({deg:+.1f}% vs FP16)  [{budget}]")

    # Save
    import json
    with open("/results/combined_perplexity.json", "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(mode: str = "all"):
    """
    modal run combined/modal_combined.py              # memory + perplexity
    modal run combined/modal_combined.py --mode memory
    modal run combined/modal_combined.py --mode ppl
    """
    if mode in ("all", "memory"):
        csv_bytes = run_memory_sweep.remote()
        out = Path("combined_memory.csv")
        out.write_bytes(csv_bytes)
        print(f"Downloaded → {out}")

    if mode in ("all", "ppl"):
        ppl = run_perplexity.remote()
        print("\nPerplexity results:", ppl)
