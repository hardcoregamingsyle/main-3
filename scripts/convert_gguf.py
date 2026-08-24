#!/usr/bin/env python3
"""
Script to convert Hugging Face models to GGUF format for efficient CPU inference.

Usage:
    python scripts/convert_gguf.py --model Qwen/Qwen1.5-MoE-A2.7B --outtype q4_k_m --outfile model.gguf
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument("--outfile", default="model.gguf", help="Output GGUF file")
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q4_k_m", "q5_k_m", "q6_k_m"],
        default="q4_k_m",
        help="Quantization type"
    )
    parser.add_argument("--vocab-only", action="store_true", help="Only convert vocab")
    parser.add_argument("--tmpdir", help="Temporary directory for conversion")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp", help="Path to llama.cpp repo")
    args = parser.parse_args()

    llama_cpp_dir = Path(args.llama_cpp_dir)
    if not llama_cpp_dir.exists():
        print(f"Cloning llama.cpp to {llama_cpp_dir}...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)], check=True)
        subprocess.run(["make", "-C", str(llama_cpp_dir), "clean"], check=True)
        subprocess.run(["make", "-C", str(llama_cpp_dir), "llama-gguf"], check=True)
    
    convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert.py"
    
    if not convert_script.exists():
        print(f"Error: Could not find conversion script in {llama_cpp_dir}")
        sys.exit(1)

    cmd = [
        sys.executable, str(convert_script),
        args.model,
        "--outfile", args.outfile,
        "--outtype", args.outtype,
    ]
    
    if args.vocab_only:
        cmd.append("--vocab-only")
    if args.tmpdir:
        cmd.extend(["--tmpdir", args.tmpdir])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Conversion complete: {args.outfile}")


if __name__ == "__main__":
    main()
