#!/usr/bin/env python3
"""
Quantize models using various backends (GPTQ, AWQ, bitsandbytes).

Usage:
    python scripts/quantize_model.py --model Qwen/Qwen1.5-MoE-A2.7B --method gptq --bits 4
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from typing import Optional


def quantize_gptq(model_id: str, output_dir: str, bits: int, group_size: int,
                  dataset: str, device: str, trust_remote_code: bool):
    """Quantize using AutoGPTQ."""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: auto-gptq not installed. Run: pip install auto-gptq")
        return False
    
    print(f"Loading model for GPTQ quantization...")
    
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        damp_percent=0.01,
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config,
        device_map="auto" if device == "auto" else device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    
    # Calibration dataset
    if dataset == "wikitext2":
        from datasets import load_dataset
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        examples = [tokenizer(text, return_tensors="pt") for text in data["text"][:128] if len(text) > 100]
    else:
        # Use simple calibration
        examples = [
            tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
            for _ in range(128)
        ]
    
    print(f"Quantizing with {len(examples)} calibration samples...")
    model.quantize(examples)
    
    print(f"Saving to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return True


def quantize_awq(model_id: str, output_dir: str, bits: int, group_size: int,
                 device: str, trust_remote_code: bool):
    """Quantize using AWQ."""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: awq not installed. Run: pip install awq")
        return False
    
    print(f"Loading model for AWQ quantization...")
    
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }
    
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "auto" else device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    
    print(f"Quantizing with AWQ...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    print(f"Saving to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return True


def quantize_bnb(model_id: str, output_dir: str, bits: int, device: str, trust_remote_code: bool):
    """Quantize using bitsandbytes (dynamic quantization)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("ERROR: transformers or bitsandbytes not installed")
        return False
    
    print(f"Loading model for bitsandbytes {bits}-bit quantization...")
    
    if bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        print(f"ERROR: bitsandbytes only supports 4-bit and 8-bit, got {bits}")
        return False
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto" if device == "auto" else device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quantize HF models")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--method", required=True, choices=["gptq", "awq", "bnb"], help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for GPTQ/AWQ")
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4", "custom"], help="Calibration dataset")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code", help="Don't trust remote code")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Bits: {args.bits}")
    print(f"Output: {output_dir}")
    
    success = False
    if args.method == "gptq":
        success = quantize_gptq(args.model, str(output_dir), args.bits, args.group_size,
                                args.dataset, args.device, args.trust_remote_code)
    elif args.method == "awq":
        success = quantize_awq(args.model, str(output_dir), args.bits, args.group_size,
                               args.device, args.trust_remote_code)
    elif args.method == "bnb":
        success = quantize_bnb(args.model, str(output_dir), args.bits, args.device, args.trust_remote_code)
    
    if success:
        print("\nQuantization completed successfully!")
    else:
        print("\nQuantization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
