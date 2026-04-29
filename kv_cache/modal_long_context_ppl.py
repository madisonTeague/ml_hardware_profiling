"""
Long-context evaluation for Qwen3-8B: perplexity + bottleneck profiling.

Addresses the TA's feedback: "a really long context evaluation like 32K,
which stresses KV quantization's ability to keep low perplexity and
probably shift the FFN->ATTN bottleneck."

Two experiments on the full Qwen3-8B model at N in {4096, 8192, 16384, 32768}:

  Exp A — Perplexity at long context
    FP16 vs INT8KV (DynamicCache subclass) on WikiText-103 test,
    using a sliding window with stride = context_len // 2.
    Shows whether INT8KV quantization noise accumulates at long context.

  Exp B — Bottleneck shift at long context
    torch.profiler on FP16 forward pass at each context length.
    Attention O(N^2) vs FFN O(N) — long context should drive attention
    share above 50%, motivating INT8KV even more.

Usage:
    # Both experiments (saves JSON + CSV to Modal Volume)
    modal run modal_long_context_ppl.py

    # Perplexity only
    modal run modal_long_context_ppl.py --mode perplexity

    # Bottleneck profiling only
    modal run modal_long_context_ppl.py --mode bottleneck

Results are downloaded to:
    results/results_lc_perplexity.json
    results/results_lc_bottleneck.csv
"""

import json
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image — needs transformers, autoawq, datasets on top of torch
# ---------------------------------------------------------------------------

lc_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "transformers==4.51.3",
        "accelerate>=0.27",
        "datasets>=2.18",
        "numpy",
        "pandas",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App("attn-long-context", image=lc_image)

results_vol = modal.Volume.from_name("attn-results", create_if_missing=True)

# Context lengths to sweep.  4096 overlaps with Exp 4 as a sanity anchor.
CONTEXT_LENS = [4096, 8192, 16384, 32768]
MODEL_ID = "Qwen/Qwen3-8B"   # FP16; ~16 GB on A100


# ---------------------------------------------------------------------------
# INT8 KV DynamicCache (same implementation as Exp 4)
# ---------------------------------------------------------------------------

def _make_int8kv_cache():
    """
    Returns a DynamicCache subclass that stores K/V tensors as INT8
    with per-head symmetric scaling and dequantizes on read.
    """
    from transformers import DynamicCache
    import torch

    class INT8KVCache(DynamicCache):
        """Headwise INT8 KV cache; dequantizes to FP16 before attention."""

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # Quantize headwise: scale = max(|x|) / 127 per head
            def _quant(t):
                # t: (B, H, S, D)
                scale = t.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-8) / 127.0
                q = t.div(scale).round().clamp(-128, 127).to(torch.int8)
                return q, scale.squeeze(-1).squeeze(-1)   # scale: (B, H)

            def _dequant(q, scale):
                # q: (B, H, S, D), scale: (B, H)
                return q.to(torch.float16) * scale.unsqueeze(-1).unsqueeze(-1)

            k_q, k_s = _quant(key_states)
            v_q, v_s = _quant(value_states)
            # Store quantized; parent tracks layer index in key_cache/value_cache
            super().update(k_q, v_q, layer_idx, cache_kwargs)
            # Retrieve and dequantize before returning to attention
            # (parent already stored; re-retrieve to be consistent)
            k_out = _dequant(self.key_cache[layer_idx], k_s)
            v_out = _dequant(self.value_cache[layer_idx], v_s)
            return k_out, v_out

    return INT8KVCache()


# ---------------------------------------------------------------------------
# Exp A: Perplexity at long context
# ---------------------------------------------------------------------------

@app.function(
    gpu="a100-80gb",   # 80 GB VRAM required: logits alone are ~10 GB at ctx=32768
    timeout=7200,
    volumes={"/results": results_vol},
)
def run_perplexity(
    context_lens: list[int] = CONTEXT_LENS,
    n_windows:    int = 20,       # sliding windows per context length
    out_name:     str = "results_lc_perplexity.json",
) -> str:
    """
    Evaluate WikiText-103 perplexity for FP16 and INT8KV at each context length.

    Uses a sliding window over the concatenated WikiText-103 test set
    with stride = context_len // 2 to give context_len tokens of history.
    Matches the teacher-forcing approach used in Exp 4.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {MODEL_ID} in FP16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Qwen3-8B has a 131072-token context window.  Tokenizing the full
    # concatenated WikiText-103 test set (~298K tokens) in one call triggers
    # a transformers warning and may silently truncate.  We tokenize with an
    # explicit max_length cap instead.
    MODEL_MAX = tokenizer.model_max_length or 131072
    print(f"Loading WikiText-103 test (capped at {MODEL_MAX:,} tokens) ...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
    full_text = "\n\n".join(ds["text"])
    tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_MAX,
    ).input_ids[0]
    print(f"Token budget: {len(tokens):,} (model max: {MODEL_MAX:,})")

    results = {}

    for ctx_len in context_lens:
        stride = ctx_len // 2
        # Cap windows to what actually fits in our token budget.
        # For ctx=32768 in 131072 tokens: (131072-32768)//16384 = 6 windows.
        max_windows = max(1, (len(tokens) - ctx_len) // stride)
        actual_n = min(n_windows, max_windows)
        starts = list(range(0, actual_n * stride, stride))[:actual_n]
        print(f"\n── ctx_len={ctx_len} | windows={len(starts)} "
              f"(requested {n_windows}, budget allows {max_windows}) ──")

        for dtype_label, cache_fn in [("fp16", None), ("int8kv", _make_int8kv_cache)]:
            nlls = []
            for start in starts:
                chunk = tokens[start: start + ctx_len].unsqueeze(0).to(device)
                if chunk.shape[1] < 2:
                    continue

                past = cache_fn() if cache_fn else None

                with torch.no_grad():
                    out = model(
                        chunk,
                        past_key_values=past,
                        use_cache=(past is not None),
                    )

                # Extract logits, then immediately free the rest of the model
                # output.  At ctx=32768, out.logits is ~10 GB (vocab=151936);
                # calling cross_entropy on the full tensor would copy it and OOM.
                logits = out.logits  # (1, seq_len, vocab) — keep reference
                del out
                torch.cuda.empty_cache()

                # Chunked cross-entropy: slice logits 512 tokens at a time so
                # each chunk is only ~155 MB (512 * 151936 * 2 bytes).
                # Slicing along dim 1 of a contiguous tensor is itself
                # contiguous — no extra copy, no OOM.
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                CHUNK = 512
                seq_len = logits.shape[1]
                total_nll, n_tokens = 0.0, 0
                for c in range(0, seq_len - 1, CHUNK):
                    c_end = min(c + CHUNK, seq_len - 1)
                    c_logits = logits[0, c:c_end]           # view, no copy
                    c_labels = chunk[0, c + 1:c_end + 1]   # shifted labels
                    mask = c_labels != pad_id
                    if mask.any():
                        loss_sum = torch.nn.functional.cross_entropy(
                            c_logits, c_labels,
                            ignore_index=pad_id, reduction="sum",
                        )
                        total_nll += loss_sum.item()
                        n_tokens  += mask.sum().item()

                del logits
                torch.cuda.empty_cache()
                import math as _math
                nlls.append(total_nll / n_tokens if n_tokens > 0 else _math.nan)

            import math
            ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")  # noqa: F811
            key = f"{dtype_label}_ctx{ctx_len}"
            results[key] = round(ppl, 4)
            print(f"  {dtype_label:8s} ctx={ctx_len:6d}: ppl = {ppl:.4f}")

    out_path = f"/results/{out_name}"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"\nSaved → {out_path}")
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Exp B: Attention/FFN bottleneck shift at long context
# ---------------------------------------------------------------------------

@app.function(
    gpu="a100-80gb",   # forward pass + logits at ctx=32768 is ~26 GB; use 80 GB to be safe
    timeout=3600,
    volumes={"/results": results_vol},
)
def run_bottleneck(
    context_lens: list[int] = CONTEXT_LENS,
    batch_size:   int = 1,
    out_name:     str = "results_lc_bottleneck.csv",
) -> bytes:
    """
    Profile Qwen3-8B FP16 attention vs FFN runtime share at long context.

    Uses CUDA event timing hooks instead of torch.profiler to avoid
    API compatibility issues across PyTorch versions.  Pre-hooks record
    a start event; post-hooks record the end event for each module.
    A single torch.cuda.synchronize() after the forward pass collects
    all timings with no per-layer blocking overhead.

    At N=32768, attention is O(N^2) while FFN is O(N), so attention
    share should climb well past 50% — motivating INT8KV at long context.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pandas as pd

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {MODEL_ID} in FP16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # ── Hook factories (closures capture the specific event objects) ──────────
    def start_hook(evt):
        def _hook(module, args):
            evt.record()
        return _hook

    def end_hook(evt):
        def _hook(module, args, output):
            evt.record()
        return _hook

    rows = []

    for ctx_len in context_lens:
        print(f"\n── Profiling ctx_len={ctx_len} B={batch_size} ──")
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, ctx_len), device=device
        )

        # Warm-up (no timing overhead)
        for _ in range(2):
            with torch.no_grad():
                model(input_ids, use_cache=False)
        torch.cuda.synchronize()

        # Register per-layer CUDA event pairs for attention and FFN
        attn_pairs, ffn_pairs = [], []
        hooks = []

        for layer in model.model.layers:
            # Attention timing
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            attn_pairs.append((s, e))
            hooks.append(layer.self_attn.register_forward_pre_hook(start_hook(s)))
            hooks.append(layer.self_attn.register_forward_hook(end_hook(e)))

            # FFN/MLP timing
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            ffn_pairs.append((s, e))
            hooks.append(layer.mlp.register_forward_pre_hook(start_hook(s)))
            hooks.append(layer.mlp.register_forward_hook(end_hook(e)))

        # Total-pass events
        total_s = torch.cuda.Event(enable_timing=True)
        total_e = torch.cuda.Event(enable_timing=True)

        total_s.record()
        with torch.no_grad():
            model(input_ids, use_cache=False)
        total_e.record()

        # One synchronize collects all events
        torch.cuda.synchronize()

        for h in hooks:
            h.remove()

        total_ms = total_s.elapsed_time(total_e)
        attn_ms  = sum(s.elapsed_time(e) for s, e in attn_pairs)
        ffn_ms   = sum(s.elapsed_time(e) for s, e in ffn_pairs)
        other_ms = max(0.0, total_ms - attn_ms - ffn_ms)

        attn_pct  = 100 * attn_ms  / total_ms if total_ms > 0 else 0
        ffn_pct   = 100 * ffn_ms   / total_ms if total_ms > 0 else 0
        other_pct = 100 * other_ms / total_ms if total_ms > 0 else 0

        row = dict(
            ctx_len=ctx_len, batch=batch_size, dtype="fp16",
            total_ms=round(total_ms, 2),
            attn_ms=round(attn_ms, 2),   attn_pct=round(attn_pct, 2),
            ffn_ms=round(ffn_ms, 2),     ffn_pct=round(ffn_pct, 2),
            other_ms=round(other_ms, 2), other_pct=round(other_pct, 2),
        )
        rows.append(row)
        print(f"  total={total_ms:.1f} ms | "
              f"attn={attn_pct:.1f}% | ffn={ffn_pct:.1f}% | other={other_pct:.1f}%")

        del input_ids
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_path = f"/results/{out_name}"
    df.to_csv(out_path, index=False)
    results_vol.commit()
    print(f"\nSaved → {out_path}")
    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "all",   # "all" | "perplexity" | "bottleneck"
):
    """
    Examples:
        modal run modal_long_context_ppl.py                    # both experiments
        modal run modal_long_context_ppl.py --mode perplexity
        modal run modal_long_context_ppl.py --mode bottleneck
    """
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    if mode in ("all", "perplexity"):
        print("Running long-context perplexity evaluation ...")
        ppl_json = run_perplexity.remote()
        ppl_path = out_dir / "results_lc_perplexity.json"
        ppl_path.write_text(ppl_json)
        print(f"Downloaded → {ppl_path}")
        print(ppl_json)

    if mode in ("all", "bottleneck"):
        print("\nRunning long-context bottleneck profiling ...")
        csv_bytes = run_bottleneck.remote()
        csv_path = out_dir / "results_lc_bottleneck.csv"
        csv_path.write_bytes(csv_bytes)
        print(f"Downloaded → {csv_path}")
