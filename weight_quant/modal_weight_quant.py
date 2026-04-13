"""
Modal app for weight quantization experiments (Experiment 2).

Usage:
    modal run modal_weight_quant.py                          # quantize model
    modal run modal_weight_quant.py --command profile        # layer profiling
    modal run modal_weight_quant.py --command shell          # interactive debug
"""
import modal
from pathlib import Path

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
weight_quant_image = (
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
        "numpy==1.26.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("weight_quantization.py", remote_path="/app/weight_quantization.py")
    .add_local_file("profile_layers.py", remote_path="/app/profile_layers.py")
)

app = modal.App("weight-quant-exp2", image=weight_quant_image)
models_vol = modal.Volume.from_name("quantized-models", create_if_missing=True)
results_vol = modal.Volume.from_name("weight-quant-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------
@app.function(gpu="A100", timeout=3600,
              volumes={"/models": models_vol, "/results": results_vol})
def quantize_model(
    model_name: str = "Qwen/Qwen3-8B",
    output_name: str = "qwen3-8b-w4a8",
) -> dict:
    import sys; sys.path.insert(0, "/app")
    import torch
    from weight_quantization import quantize_model as _quantize

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    output_dir = f"/models/{output_name}"

    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        print(f"Model already exists at {output_dir}, skipping.")
        return {"status": "already_exists", "path": output_dir}

    model, tokenizer = _quantize(model_name, output_dir)
    models_vol.commit()
    return {"status": "success", "path": output_dir}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@app.function(gpu="A100-80GB", timeout=3600,
              volumes={"/models": models_vol, "/results": results_vol})
def profile_layers_sweep(
    model_fp16: str = "Qwen/Qwen3-8B",
    model_w4a8: str = "/models/qwen3-8b-w4a8",
    output_name: str = "results_exp2_layers.csv",
) -> bytes:
    import sys; sys.path.insert(0, "/app")
    import torch
    from profile_layers import profile_sweep
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_path = f"/results/{output_name}"
    df, ppl = profile_sweep(model_fp16, model_w4a8, out_path)
    results_vol.commit()

    # Append perplexity as a separate small CSV
    import pandas as pd
    ppl_df = pd.DataFrame([{"dtype": k, "perplexity": v} for k, v in ppl.items()])
    ppl_path = f"/results/perplexity.csv"
    ppl_df.to_csv(ppl_path, index=False)
    results_vol.commit()

    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(command: str = "quantize"):
    if command == "quantize":
        result = quantize_model.remote()
        print(f"Result: {result}")
    elif command == "profile":
        csv_bytes = profile_layers_sweep.remote()
        Path("results_exp2_layers.csv").write_bytes(csv_bytes)
        print("Downloaded results_exp2_layers.csv")
    else:
        raise ValueError(f"Unknown command: {command}")
