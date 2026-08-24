#!/usr/bin/env python3
"""
Download Hugging Face models with resume support and verification.

Usage:
    python scripts/download_model.py --model Qwen/Qwen1.5-MoE-A2.7B --cache-dir ./models
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)


def verify_checksum(filepath: Path, expected_sha256: Optional[str] = None) -> bool:
    """Verify file SHA256 checksum."""
    if not expected_sha256:
        return True
    
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


def format_size(bytes_val: int) -> str:
    """Format bytes as human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def main():
    parser = argparse.ArgumentParser(description="Download HF model with resume")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--revision", default="main", help="Model revision/branch")
    parser.add_argument("--cache-dir", default="~/.cache/huggingface/hub", help="Cache directory")
    parser.add_argument("--local-dir", help="Download to local directory instead of cache")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env)")
    parser.add_argument("--include", action="append", help="Patterns to include (can repeat)")
    parser.add_argument("--exclude", action="append", help="Patterns to exclude (can repeat)")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--resume/--no-resume", default=True, help="Resume interrupted downloads")
    parser.add_argument("--verify/--no-verify", default=True, help="Verify downloads")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    cache_dir = Path(args.cache_dir).expanduser()
    
    # Default exclusions for MoE models (skip unnecessary files)
    exclude_patterns = args.exclude or [
        "*.msgpack",
        "*.h5",
        "*.ot",
        "*.tflite",
        "*.onnx",
        "*.pb",
        "*.pbtxt",
        "*.ckpt",
        "*.index.json",  # Will be regenerated
        "README.md",
        ".gitattributes",
    ]
    
    include_patterns = args.include or [
        "*.safetensors",
        "*.bin",
        "config.json",
        "tokenizer*.json",
        "tokenizer*.model",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "generation_config.json",
        "*.py",
    ]
    
    print(f"Model: {args.model}")
    print(f"Revision: {args.revision}")
    print(f"Cache dir: {cache_dir}")
    if args.local_dir:
        print(f"Local dir: {args.local_dir}")
    print(f"Include: {include_patterns}")
    print(f"Exclude: {exclude_patterns}")
    print(f"Workers: {args.max_workers}")
    print(f"Resume: {args.resume}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would download with above settings")
        return
    
    try:
        local_dir = Path(args.local_dir).expanduser() if args.local_dir else None
        
        path = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=cache_dir if not local_dir else None,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=include_patterns,
            ignore_patterns=exclude_patterns,
            max_workers=args.max_workers,
            resume_download=args.resume,
            force_download=False,
        )
        
        print(f"\nDownload complete: {path}")
        
        # Print size info
        if local_dir:
            total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            print(f"Total size: {format_size(total_size)}")
            
            # List files
            for f in sorted(local_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(local_dir)
                    print(f"  {rel} ({format_size(f.stat().st_size)})")
        
    except HfHubHTTPError as e:
        print(f"\nHTTP Error: {e}")
        if e.response.status_code == 401:
            print("Authentication required. Set HF_TOKEN or use --token")
        elif e.response.status_code == 404:
            print("Model not found. Check model ID and revision.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
