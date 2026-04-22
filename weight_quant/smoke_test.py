"""
Smoke test for the W4A8 Qwen3-8B checkpoint on the `quantized-models` Modal volume.

Confirms:
  1. The checkpoint loads via AutoAWQ.from_quantized.
  2. Tokenizer matches the model and a forward pass runs on real input.
  3. Model generates coherent text (sanity check that quantization wasn't corrupted).
  4. Reports peak VRAM so downstream combined runs (e.g. + INT8 KV) know what to expect.

Usage:
    modal run weight_quant/smoke_test.py
"""
import modal

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.5.1",
        "transformers==4.51.3",
        "accelerate==1.2.1",
        "autoawq==0.2.9",
        "autoawq-kernels==0.0.9",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App("w4a8-smoke-test", image=image)
models_vol = modal.Volume.from_name("quantized-models")


@app.function(gpu="A100-80GB", timeout=900, volumes={"/models": models_vol})
def smoke_test(checkpoint_path: str = "/models/qwen3-8b-w4a8") -> dict:
    import os
    import torch
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading W4A8 checkpoint from: {checkpoint_path}")

    files = sorted(os.listdir(checkpoint_path))
    print(f"Files in checkpoint dir ({len(files)}):")
    for f in files:
        size_mb = os.path.getsize(os.path.join(checkpoint_path, f)) / 1e6
        print(f"  {f:<45} {size_mb:>10.1f} MB")
    total_mb = sum(os.path.getsize(os.path.join(checkpoint_path, f)) for f in files) / 1e6
    print(f"  {'TOTAL':<45} {total_mb:>10.1f} MB")

    torch.cuda.reset_peak_memory_stats()

    print("\n[1/3] Loading model with AutoAWQForCausalLM.from_quantized "
          "(fuse_layers=False so standard Qwen3Attention modules are kept "
          "— required for downstream INT8 KV swap)...")
    model = AutoAWQForCausalLM.from_quantized(
        checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
        fuse_layers=False,
    )
    load_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"      OK. Peak VRAM after load: {load_peak_gb:.2f} GB")

    print("\n[2/3] Loading tokenizer (Qwen/Qwen3-8B)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", trust_remote_code=True
    )
    print(f"      OK. Vocab size: {len(tokenizer)}")

    print("\n[3/3] Running generate() on 'The capital of France is'...")
    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.model.generate(
            ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"      Output: {text!r}")
    print(f"      Peak VRAM during generate: {gen_peak_gb:.2f} GB")

    print("\n[bonus] Inspecting AWQ wrapping...")
    inner = model.model
    first_attn = inner.model.layers[0].self_attn
    first_mlp = inner.model.layers[0].mlp
    attn_types = {n: type(m).__name__ for n, m in first_attn.named_children()}
    mlp_types = {n: type(m).__name__ for n, m in first_mlp.named_children()}
    print(f"      layer[0].self_attn children: {attn_types}")
    print(f"      layer[0].mlp children:       {mlp_types}")

    return {
        "status": "ok",
        "checkpoint_total_mb": round(total_mb, 1),
        "load_peak_vram_gb": round(load_peak_gb, 2),
        "gen_peak_vram_gb": round(gen_peak_gb, 2),
        "sample_output": text,
        "attn_module_types": attn_types,
        "mlp_module_types": mlp_types,
    }


@app.local_entrypoint()
def main():
    result = smoke_test.remote()
    print("\n=== SMOKE TEST RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")
