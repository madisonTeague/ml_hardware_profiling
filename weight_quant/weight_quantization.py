"""
Weight quantization using AutoAWQ for Experiment 2.

This script:
1. Loads Qwen3-8B
2. Applies W4A8 quantization using AWQ
3. Validates perplexity on WikiText-103
4. Saves quantized model for later profiling

Usage:
    python weight_quantization.py --model Qwen/Qwen3-8B --output ./quantized_model
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import numpy as np


def load_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """
    Load WikiText-103 calibration data for AWQ quantization.
    
    AWQ needs a small calibration set to compute optimal quantization scales.
    Returns a list of text strings (not tokenized).
    """
    print(f"Loading {n_samples} calibration samples from WikiText-103...")
    
    # Load WikiText-103 dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    # Filter out empty entries and collect text samples
    calibration_data = []
    for entry in dataset["text"]:
        if entry.strip():  # Skip empty entries
            calibration_data.append(entry.strip())
        if len(calibration_data) >= n_samples:
            break
    
    print(f"Prepared {len(calibration_data)} calibration samples")
    return calibration_data


def quantize_model(model_name, output_dir, w_bit=4, group_size=128):
    """
    Quantize model using AutoAWQ.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")
        output_dir: Where to save quantized model
        w_bit: Weight bit-width (4 for W4A8)
        group_size: Group size for quantization (128 is standard)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model for AWQ quantization
    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Load calibration data
    calib_data = load_calibration_data(tokenizer)
    
    # Configure quantization
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": "GEMM",  # Use GEMM kernels (faster on A100)
    }
    
    print(f"Quantizing to W{w_bit}A8 with group_size={group_size}...")
    print("This may take 5-10 minutes...")
    
    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    
    # Save quantized model
    print(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Quantization complete!")
    return model, tokenizer


def evaluate_perplexity(model, tokenizer, max_samples=100, seq_len=512):
    """
    Evaluate perplexity on WikiText-103 test set.
    
    Returns:
        perplexity: float
    """
    print("Evaluating perplexity on WikiText-103 test set...")
    
    # Load test set
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(dataset["text"][:500])
    encodings = tokenizer(text, return_tensors="pt")
    
    # Compute perplexity
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = encodings.input_ids.to(device)
    nlls = []
    
    with torch.no_grad():
        for i in range(0, min(len(input_ids[0]), max_samples * seq_len), seq_len):
            chunk = input_ids[:, i:i + seq_len]
            if chunk.size(1) < seq_len:
                continue
            
            # Forward pass
            outputs = model(chunk, labels=chunk)
            nll = outputs.loss
            nlls.append(nll.item())
            
            if len(nlls) >= max_samples:
                break
    
    # Compute perplexity
    avg_nll = np.mean(nlls)
    perplexity = np.exp(avg_nll)
    
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity


def compare_models(original_name, quantized_dir):
    """
    Compare FP16 baseline vs W4A8 quantized model.
    """
    print("\n" + "="*60)
    print("COMPARISON: FP16 vs W4A8")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(original_name, trust_remote_code=True)
    
    # Load FP16 model
    print("\n1. Loading FP16 baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        original_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ppl_fp16 = evaluate_perplexity(model_fp16, tokenizer, max_samples=50)
    
    # Get model size
    param_count = sum(p.numel() for p in model_fp16.parameters())
    size_fp16_mb = param_count * 2 / 1e6  # FP16 = 2 bytes per param
    
    del model_fp16
    torch.cuda.empty_cache()
    
    # Load W4A8 model
    print("\n2. Loading W4A8 quantized model...")
    model_w4a8 = AutoAWQForCausalLM.from_quantized(
        quantized_dir,
        trust_remote_code=True,
        device_map="auto",
    )
    ppl_w4a8 = evaluate_perplexity(model_w4a8, tokenizer, max_samples=50)
    
    # Estimate quantized size (4 bits per weight + activations still FP16)
    size_w4a8_mb = param_count * 0.5 / 1e6  # W4 = 0.5 bytes per param
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Model: {original_name}")
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print()
    print(f"{'Metric':<30} {'FP16':<15} {'W4A8':<15} {'Change':<15}")
    print("-" * 60)
    print(f"{'Perplexity':<30} {ppl_fp16:<15.2f} {ppl_w4a8:<15.2f} {(ppl_w4a8/ppl_fp16-1)*100:+.2f}%")
    print(f"{'Model Size (MB)':<30} {size_fp16_mb:<15.0f} {size_w4a8_mb:<15.0f} {(size_w4a8_mb/size_fp16_mb-1)*100:+.2f}%")
    print(f"{'Theoretical Speedup (FFN)':<30} {'1.0x':<15} {'~4x':<15} {'+300%':<15}")
    print()
    print("Next steps:")
    print("1. Run layer-wise profiling to measure actual speedup")
    print("2. Check if attention becomes the bottleneck")
    print("3. Profile FFN GEMMs with NCU")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Quantize model using AutoAWQ")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./quantized_qwen_w4a8",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--w-bit",
        type=int,
        default=4,
        help="Weight bit-width (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FP16 vs W4A8 perplexity after quantization",
    )
    args = parser.parse_args()
    
    # Quantize
    model, tokenizer = quantize_model(
        args.model,
        args.output,
        w_bit=args.w_bit,
        group_size=args.group_size,
    )
    
    # Optionally compare
    if args.compare:
        compare_models(args.model, args.output)
    else:
        print("\nTo compare FP16 vs W4A8, run:")
        print(f"python {__file__} --model {args.model} --output {args.output} --compare")


if __name__ == "__main__":
    main()
