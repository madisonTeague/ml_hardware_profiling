"""
Layer-wise profiling for Experiment 2: Weight Quantization.

Uses CUDA event hooks on self_attn and mlp submodules to measure
actual per-layer timing. Validated on Modal A100 with torch 2.5.1.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dataclasses import dataclass
from typing import Literal


@dataclass
class ProfileConfig:
    batch_size: int = 1
    seq_len: int = 512
    dtype: Literal["fp16", "w4a8"] = "fp16"
    n_warmup: int = 3
    n_trials: int = 5


def load_model(model_path, dtype="fp16"):
    print(f"Loading {dtype} model from {model_path}...")

    if dtype == "fp16":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto",
        )
    elif dtype == "w4a8":
        from awq import AutoAWQForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_quantized(
            model_path, trust_remote_code=True, device_map="auto",
            fuse_layers=False,
        )
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hook-based layer timing (validated on Modal)
# ---------------------------------------------------------------------------

class _TimingHook:
    """Attaches pre/post hooks to record elapsed CUDA time and memory delta."""
    def __init__(self):
        self.times: list[float] = []
        self.mem_deltas: list[int] = []
        self._start = None
        self._mem_before = 0

    def pre(self, module, inp):
        self._start = torch.cuda.Event(enable_timing=True)
        self._mem_before = torch.cuda.memory_allocated()
        self._start.record()

    def post(self, module, inp, out):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        self.times.append(self._start.elapsed_time(end))
        self.mem_deltas.append(torch.cuda.memory_allocated() - self._mem_before)


def profile_model(model, tokenizer, cfg: ProfileConfig):
    device = next(model.parameters()).device

    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (cfg.batch_size, cfg.seq_len), device=device,
    )

    # Warmup
    print(f"  Warming up ({cfg.n_warmup} iters)...", end="", flush=True)
    with torch.no_grad():
        for _ in range(cfg.n_warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()
    print(" done")

    # Attach hooks to every self_attn and mlp submodule
    hooks_handles = []
    attn_hooks: list[_TimingHook] = []
    ffn_hooks: list[_TimingHook] = []

    for name, mod in model.named_modules():
        if name.endswith(".self_attn"):
            th = _TimingHook()
            attn_hooks.append(th)
            hooks_handles.append(mod.register_forward_pre_hook(th.pre))
            hooks_handles.append(mod.register_forward_hook(th.post))
        elif name.endswith(".mlp"):
            th = _TimingHook()
            ffn_hooks.append(th)
            hooks_handles.append(mod.register_forward_pre_hook(th.pre))
            hooks_handles.append(mod.register_forward_hook(th.post))

    # Also time total end-to-end + peak memory
    total_times = []
    peak_mem_bytes = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    print(f"  Profiling ({cfg.n_trials} iters)...", end="", flush=True)
    with torch.no_grad():
        for _ in range(cfg.n_trials):
            torch.cuda.reset_peak_memory_stats()
            start_ev.record()
            _ = model(input_ids)
            end_ev.record()
            torch.cuda.synchronize()
            total_times.append(start_ev.elapsed_time(end_ev))
            peak_mem_bytes.append(torch.cuda.max_memory_allocated())
    print(" done")

    # Remove hooks
    for h in hooks_handles:
        h.remove()

    # Aggregate: each hook recorded n_trials times per layer.
    # We want median across trials.
    n_layers_attn = len(attn_hooks)
    n_layers_ffn = len(ffn_hooks)

    def per_trial_sum(hooks_list, attr="times"):
        """Sum across layers for each trial, return list of per-trial totals."""
        per_trial = [0.0] * cfg.n_trials
        for th in hooks_list:
            for i, t in enumerate(getattr(th, attr)):
                per_trial[i] += t
        return per_trial

    attn_per_trial = per_trial_sum(attn_hooks)
    ffn_per_trial = per_trial_sum(ffn_hooks)
    total_per_trial = total_times

    attn_ms = np.median(attn_per_trial)
    ffn_ms = np.median(ffn_per_trial)
    total_ms = np.median(total_per_trial)
    other_ms = max(0, total_ms - attn_ms - ffn_ms)

    attn_pct = (attn_ms / total_ms * 100) if total_ms > 0 else 0
    ffn_pct = (ffn_ms / total_ms * 100) if total_ms > 0 else 0
    other_pct = (other_ms / total_ms * 100) if total_ms > 0 else 0

    # Memory: per-layer allocation deltas (median across trials)
    attn_mem = np.median(per_trial_sum(attn_hooks, "mem_deltas"))
    ffn_mem = np.median(per_trial_sum(ffn_hooks, "mem_deltas"))
    peak_mem = np.median(peak_mem_bytes)
    total_mem_mb = round(peak_mem / 1e6, 2)
    attn_mem_mb = round(attn_mem / 1e6, 2)
    ffn_mem_mb = round(ffn_mem / 1e6, 2)
    other_mem_mb = round(max(0, peak_mem - attn_mem - ffn_mem) / 1e6, 2)

    print(f"  Total: {total_ms:.2f} ms | "
          f"Attn: {attn_ms:.2f} ms ({attn_pct:.1f}%) | "
          f"FFN: {ffn_ms:.2f} ms ({ffn_pct:.1f}%) | "
          f"Other: {other_ms:.2f} ms ({other_pct:.1f}%)")
    print(f"  Peak mem: {total_mem_mb:.1f} MB | "
          f"Attn alloc: {attn_mem_mb:.1f} MB | "
          f"FFN alloc: {ffn_mem_mb:.1f} MB")

    base = dict(dtype=cfg.dtype, batch=cfg.batch_size, seq_len=cfg.seq_len)
    rows = [
        dict(**base, layer_type="attention", time_ms=attn_ms,
             time_pct=round(attn_pct, 2), mem_mb=attn_mem_mb),
        dict(**base, layer_type="ffn", time_ms=ffn_ms,
             time_pct=round(ffn_pct, 2), mem_mb=ffn_mem_mb),
        dict(**base, layer_type="other", time_ms=other_ms,
             time_pct=round(other_pct, 2), mem_mb=other_mem_mb),
        dict(**base, layer_type="total", time_ms=total_ms,
             time_pct=100.0, mem_mb=total_mem_mb),
    ]
    return rows


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def evaluate_perplexity(model, tokenizer, max_samples=50, seq_len=512):
    print("  Evaluating perplexity on WikiText-103 test set...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()][:500])
    enc = tokenizer(text, return_tensors="pt")

    device = next(model.parameters()).device
    input_ids = enc.input_ids.to(device)
    nlls = []

    with torch.no_grad():
        for i in range(0, min(len(input_ids[0]), max_samples * seq_len), seq_len):
            chunk = input_ids[:, i:i + seq_len]
            if chunk.size(1) < seq_len:
                continue
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item())
            if len(nlls) >= max_samples:
                break

    ppl = np.exp(np.mean(nlls))
    print(f"  Perplexity: {ppl:.2f} (n={len(nlls)} chunks)")
    return ppl


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def profile_sweep(model_fp16_path, model_w4a8_path, output_csv):
    batch_sizes = [1, 4, 16]
    seq_lens = [512, 2048, 4096]
    all_rows = []
    ppl_results = {}

    # --- FP16 ---
    print("\n" + "=" * 60)
    print("PROFILING FP16 MODEL")
    print("=" * 60)
    model_fp16, tok = load_model(model_fp16_path, "fp16")
    ppl_results["fp16"] = evaluate_perplexity(model_fp16, tok)

    for batch in batch_sizes:
        for seq_len in seq_lens:
            print(f"\nConfig: batch={batch}, seq_len={seq_len}")
            cfg = ProfileConfig(batch_size=batch, seq_len=seq_len, dtype="fp16")
            try:
                rows = profile_model(model_fp16, tok, cfg)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  FAILED: {e}")

    del model_fp16
    torch.cuda.empty_cache()

    # --- W4A8 ---
    print("\n" + "=" * 60)
    print("PROFILING W4A8 MODEL")
    print("=" * 60)
    model_w4a8, tok = load_model(model_w4a8_path, "w4a8")
    ppl_results["w4a8"] = evaluate_perplexity(model_w4a8, tok)

    for batch in batch_sizes:
        for seq_len in seq_lens:
            print(f"\nConfig: batch={batch}, seq_len={seq_len}")
            cfg = ProfileConfig(batch_size=batch, seq_len=seq_len, dtype="w4a8")
            try:
                rows = profile_model(model_w4a8, tok, cfg)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  FAILED: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Perplexity summary
    print("\n" + "=" * 60)
    print("PERPLEXITY")
    print("=" * 60)
    for dtype, ppl in ppl_results.items():
        print(f"  {dtype}: {ppl:.2f}")

    return df, ppl_results
