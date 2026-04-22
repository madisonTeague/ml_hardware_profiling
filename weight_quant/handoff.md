# W4A8 Checkpoint Handoff

For the combined INT8 KV + W4A8 experiment.

## Where the checkpoint lives

- **Modal workspace:** `mlsys-15642` (the shared workspace)
- **Volume:** `quantized-models`
- **Path inside the container:** `/models/qwen3-8b-w4a8`

The volume is shared across the team workspace, so as long as your active Modal profile is `mlsys-15642`, you have access. Verify with:

```bash
modal profile activate mlsys-15642
modal volume ls quantized-models qwen3-8b-w4a8
```

You should see 11 files including two `model-0000{1,2}-of-00002.safetensors` shards (~6.1 GB total).

## How the checkpoint was produced

- Base model: `Qwen/Qwen3-8B`
- Library: `autoawq==0.2.9` + `autoawq-kernels==0.0.9`
- Config: W4A8, group_size=128, GEMM kernels, zero_point=True
- Calibration: first 128 non-empty samples from WikiText-103 train split, no shuffling (see `weight_quantization.py::load_calibration_data`). Calibration is deterministic, so re-running the same script will produce a byte-equivalent checkpoint.
- Perplexity (WikiText-103 test, 50×512-token windows, measured on a previous identically-configured quantize run): **12.71** vs FP16 **12.27** (+3.6%, well under the 5% degradation budget). To re-verify on this exact checkpoint, see "Re-verify perplexity" below.
- Smoke-test peak VRAM on A100-80GB (verified on this checkpoint): **7.35 GB** after load, **6.12 GB** during a 20-token greedy generate.

## How to load it in your Modal app

**Critical:** load with `fuse_layers=False`. The default (`True`) replaces `Qwen3Attention` with AWQ's own `QuantAttentionFused` block that calls `flash_attn_func` — that breaks any KV-cache patching you'd want to do, and also requires the `flash_attn` package which isn't in our image.

```python
import modal
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# NOTE: do NOT pass create_if_missing=True. If you're accidentally on the
# wrong Modal profile, this will fail loudly instead of silently creating
# an empty volume that then mysteriously can't find the checkpoint.
models_vol = modal.Volume.from_name("quantized-models")

@app.function(gpu="A100-80GB", volumes={"/models": models_vol}, timeout=3600)
def my_combined_run():
    model = AutoAWQForCausalLM.from_quantized(
        "/models/qwen3-8b-w4a8",
        trust_remote_code=True,
        device_map="auto",
        fuse_layers=False,            # <-- IMPORTANT
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", trust_remote_code=True
    )

    # The actual nn.Module transformer is at: model.model
    # Attention layers:  model.model.model.layers[i].self_attn   (Qwen3Attention)
    # MLP layers:        model.model.model.layers[i].mlp         (Qwen3MLP)
    # Each q_proj/k_proj/v_proj/o_proj inside self_attn is a WQLinear_GEMM
    # (4-bit weights, FP16 activations).
```

## Required image deps

These match what was used to quantize and what was smoke-tested. Add to your Modal image:

```python
modal.Image.from_registry(
    "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).pip_install(
    "torch==2.5.1",
    "transformers==4.51.3",
    "accelerate==1.2.1",
    "autoawq==0.2.9",
    "autoawq-kernels==0.0.9",
    extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
)
```

If you also need datasets/pandas (for perplexity eval), add `datasets==2.21.0` and `pandas==2.2.0`.

## Notes for INT8 KV integration

- AWQ only quantizes `nn.Linear` weights; activations (including K/V projection outputs) remain FP16, so your INT8 KV path receives the same dtype it would on the FP16 baseline.
- Patch attention at `model.model.model.layers[*].self_attn`, **not** on the `AutoAWQForCausalLM` wrapper.
- W4A8 contributes ~62% peak-memory savings (mostly static weight memory). INT8 KV contributes runtime KV-cache savings. They attack different memory pools, so they should compose roughly additively — that's the headline question for the combined run.

## Sanity check / smoke test

A self-contained Modal app that loads the checkpoint and runs `generate()` lives at `weight_quant/smoke_test.py`. Run from the project root with:

```bash
modal profile activate mlsys-15642   # make sure you're on the shared workspace
modal run weight_quant/smoke_test.py
```

Expected output includes `Output: 'The capital of France is Paris...'` and module types showing `WQLinear_GEMM` for all q/k/v/o/gate/up/down projections.

## Re-verify perplexity (optional)

If you want to confirm perplexity on this exact shared-workspace checkpoint (rather than trusting the +3.6% number from the prior identically-configured run), `weight_quant/weight_quantization.py` already has a `--compare` mode that runs FP16 baseline vs W4A8 perplexity side-by-side. It just needs to be wrapped in a Modal function pointing at `/models/qwen3-8b-w4a8`. Skip unless we want it for the final report numbers.

## Ping me if

- `from_quantized` errors out — most likely an autoawq version mismatch in your image.
- Kernel composition crashes — most likely place is AWQ's `WQLinear_GEMM` interacting with whatever attention backend you're using.
- The generated output looks like garbage — could mean the wrong checkpoint loaded; double-check the path is `/models/qwen3-8b-w4a8` and the volume mounted to `/models`.
