"""Command-line interface for MoE Ultra Engine."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import load_config, get_config_path
from .logging_utils import setup_logging
from .inference_engine import InferenceEngine
from .server import run_server

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to configuration file",
)
@click.option(
    "--env",
    "-e",
    type=click.Choice(["development", "staging", "production"]),
    default="development",
    help="Environment to run in",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], env: str) -> None:
    """MoE Ultra Engine - Ultra-memory-efficient inference for large models."""
    ctx.ensure_object(dict)
    ctx.obj["env"] = env
    ctx.obj["config_path"] = config or get_config_path(env)
    
    # Load configuration
    config_data = load_config(ctx.obj["config_path"])
    ctx.obj["config"] = config_data
    
    # Setup logging
    log_level = config_data.get("logging", {}).get("level", "INFO")
    setup_logging(log_level, config_data.get("logging", {}))
    logger.info(f"Loaded configuration from {ctx.obj['config_path']}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--device", "-d", type=str, default="cpu", help="Device to run on")
@click.option("--precision", "-p", type=click.Choice(["fp32", "fp16", "bf16", "int8", "int4"]), default="bf16")
@click.option("--max-context", "-m", type=int, default=4096, help="Maximum context length")
@click.option("--num-experts", "-n", type=int, default=8, help="Number of experts to load")
@click.option("--port", "-p", type=int, default=8000, help="Server port")
@click.option("--host", "-H", type=str, default="0.0.0.0", help="Host to bind")
@click.pass_context
def serve(
    ctx: click.Context,
    model_path: str,
    device: str,
    precision: str,
    max_context: int,
    num_experts: int,
    port: int,
    host: str,
) -> None:
    """Start the inference server."""
    config = ctx.obj["config"]
    
    engine_config = {
        "model_path": model_path,
        "device": device,
        "precision": precision,
        "max_context_length": max_context,
        "num_experts": num_experts,
        **config.get("inference", {}),
    }
    
    logger.info(f"Starting server on {host}:{port} with model {model_path}")
    logger.info(f"Configuration: {engine_config}")
    
    try:
        run_server(host, port, engine_config)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path())
@click.option("--output", "-o", type=click.Path(), required=True, help="Output path for converted model")
@click.option("--format", "-f", type=click.Choice(["gguf", "safetensors", "onnx"]), default="gguf")
@click.option("--quantization", "-q", type=click.Choice(["none", "q4_0", "q4_1", "q8_0", "f16"]), default="q4_0")
@click.pass_context
def convert(
    ctx: click.Context,
    model_path: str,
    output: str,
    format: str,
    quantization: str,
) -> None:
    """Convert a model to specified format."""
    from .model_converters import convert_model
    
    config = ctx.obj["config"]
    conversion_config = {
        "output_format": format,
        "quantization": quantization,
        **config.get("conversion", {}),
    }
    
    logger.info(f"Converting model {model_path} to {format} ({quantization})")
    
    try:
        result = convert_model(model_path, output, conversion_config)
        logger.info(f"Conversion complete: {result}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output path for quantized model")
@click.option("--method", "-m", type=click.Choice(["awq", "gptq", "bitsandbytes", "llama.cpp"]), default="llama.cpp")
@click.option("--bits", "-b", type=click.IntRange(4, 8), default=4)
@click.pass_context
def quantize(
    ctx: click.Context,
    model_path: str,
    output: str,
    method: str,
    bits: int,
) -> None:
    """Quantize a model for efficient inference."""
    from .model_quantizers import quantize_model
    
    config = ctx.obj["config"]
    quantization_config = {
        "method": method,
        "bits": bits,
        **config.get("quantization", {}),
    }
    
    logger.info(f"Quantizing model {model_path} using {method} ({bits}bit)")
    
    try:
        result = quantize_model(model_path, output, quantization_config)
        logger.info(f"Quantization complete: {result}")
    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path())
@click.option("--hf-token", type=str, envvar="HF_TOKEN", help="HuggingFace API token")
@click.option("--resume", is_flag=True, help="Resume interrupted download")
@click.pass_context
def download(
    ctx: click.Context,
    model_path: str,
    hf_token: Optional[str],
    resume: bool,
) -> None:
    """Download a model from HuggingFace Hub."""
    from .model_downloader import download_model
    
    config = ctx.obj["config"]
    download_config = {
        "hf_token": hf_token,
        "resume": resume,
        **config.get("download", {}),
    }
    
    logger.info(f"Downloading model {model_path}")
    
    try:
        result = asyncio.run(download_model(model_path, download_config))
        logger.info(f"Download complete: {result}")
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--warmup-steps", type=int, default=10, help="Number of warmup iterations")
@click.option("--iterations", type=int, default=100, help="Number of benchmark iterations")
@click.option("--batch-size", type=int, default=1, help="Batch size for benchmarking")
@click.option("--context-length", type=int, default=1024, help="Context length for benchmarking")
@click.option("--output", type=click.Path(), help="Output file for results")
@click.pass_context
def benchmark(
    ctx: click.Context,
    warmup_steps: int,
    iterations: int,
    batch_size: int,
    context_length: int,
    output: Optional[str],
) -> None:
    """Run performance benchmarks."""
    from .benchmark import run_benchmark
    
    config = ctx.obj["config"]
    benchmark_config = {
        "warmup_steps": warmup_steps,
        "iterations": iterations,
        "batch_size": batch_size,
        "context_length": context_length,
        **config.get("benchmark", {}),
    }
    
    logger.info(f"Running benchmark with {iterations} iterations")
    
    try:
        results = asyncio.run(run_benchmark(benchmark_config))
        
        if output:
            import json
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output}")
        else:
            logger.info(f"Benchmark results: {results}")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display system information."""
    import platform
    import subprocess
    
    config = ctx.obj["config"]
    
    info_lines = [
        f"MoE Ultra Engine v{config.get('version', 'unknown')}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"CPU: {subprocess.check_output(['nproc'], text=True).strip()} cores",
    ]
    
    try:
        import torch
        info_lines.append(f"PyTorch: {torch.__version__}")
        info_lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            info_lines.append(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        info_lines.append("PyTorch: not installed")
    
    for line in info_lines:
        click.echo(line)


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Check system health and dependencies."""
    checks = []
    
    # Check Python version
    import sys
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10)))
    
    # Check PyTorch
    try:
        import torch
        checks.append(("PyTorch installed", True))
        checks.append(("PyTorch >= 2.0", tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 0)))
    except ImportError:
        checks.append(("PyTorch installed", False))
    
    # Check CUDA
    try:
        import torch
        checks.append(("CUDA available", torch.cuda.is_available()))
    except ImportError:
        checks.append(("CUDA available", False))
    
    # Check model path
    model_path = ctx.obj["config"].get("model_path", "")
    checks.append((f"Model path exists: {model_path}", Path(model_path).exists() if model_path else False))
    
    # Check write permissions
    logs_dir = Path.cwd() / "logs"
    checks.append(("Logs directory writable", logs_dir.is_dir() and os.access(logs_dir, os.W_OK)))
    
    click.echo("\nSystem Health Check")
    click.echo("=" * 40)
    
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        click.echo(f"{status} {name}")
        if not passed:
            all_passed = False
    
    click.echo("=" * 40)
    click.echo(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    
    if not all_passed:
        sys.exit(1)


def main() -> None:
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
