"""
Command-line interface for MoE Ultra Engine.

Provides commands for model management, inference, benchmarking, and server control.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from .config import Config, load_config, merge_configs
from .engine import MoEEngine
from .logging_utils import setup_logging, get_logger


app = typer.Typer(
    name="moe-engine",
    help="Ultra-memory-efficient MoE inference engine",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
logger = get_logger(__name__)


# Global config instance
_config: Optional[Config] = None
_engine: Optional[MoEEngine] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_engine() -> MoEEngine:
    """Get or create global engine instance."""
    global _engine
    if _engine is None:
        _engine = MoEEngine(get_config())
    return _engine


@app.callback()
def callback(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    profile: str = typer.Option("default", "--profile", "-p", help="Config profile (default/prod)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
    log_format: str = typer.Option("text", "--log-format", help="Log format (json/text)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Global options for all commands."""
    if verbose:
        log_level = "DEBUG"

    setup_logging(level=log_level, log_format=log_format, enable_console=True)

    if config_file:
        global _config
        _config = load_config(config_file=config_file, profile=profile)

    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config"] = get_config()


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Server host"),
    port: Optional[int] = typer.Option(None, "--port", help="Server port"),
    workers: Optional[int] = typer.Option(None, "--workers", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
) -> None:
    """Start the inference API server."""
    config = get_config()

    # Override with CLI args
    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if workers:
        config.server.workers = workers

    console.print(Panel.fit(
        f"[bold green]Starting MoE Ultra Engine API Server[/bold green]\n"
        f"Host: {config.server.host}:{config.server.port}\n"
        f"Workers: {config.server.workers}\n"
        f"Model: {config.model.name}",
        title="Server Configuration",
        border_style="green",
    ))

    # Import here to avoid circular imports
    from api.main import create_app
    import uvicorn

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers if not reload else 1,
        reload=reload,
        log_level=config.server.log_level.lower(),
        access_log=config.server.access_log,
    )


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Input prompt"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-m", help="Max tokens to generate"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Top-p sampling"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k sampling"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    stop: List[str] = typer.Option([], "--stop", help="Stop sequences"),
) -> None:
    """Generate text from a prompt."""
    config = get_config()
    engine = get_engine()

    # Override inference params
    if max_tokens:
        config.inference.max_tokens = max_tokens
    if temperature is not None:
        config.inference.temperature = temperature
    if top_p is not None:
        config.inference.top_p = top_p
    if top_k is not None:
        config.inference.top_k = top_k
    if seed is not None:
        config.inference.seed = seed
    if stop:
        config.inference.stop_sequences = stop
    config.inference.stream = stream

    console.print(Panel.fit(
        f"[bold]Prompt:[/bold] {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
        f"[bold]Max tokens:[/bold] {config.inference.max_tokens}\n"
        f"[bold]Temperature:[/bold] {config.inference.temperature}\n"
        f"[bold]Stream:[/bold] {stream}",
        title="Generation Parameters",
        border_style="blue",
    ))

    async def _generate() -> None:
        await engine.initialize()

        if stream:
            console.print("[bold green]Output:[/bold green] ", end="")
            async for token in engine.generate_stream(prompt, config.inference):
                console.print(token, end="", highlight=False)
            console.print()  # Newline at end
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Generating...", total=None)
                result = await engine.generate(prompt, config.inference)
                progress.update(task, completed=True)

            console.print(Panel(result, title="Generated Text", border_style="green"))

        await engine.shutdown()

    asyncio.run(_generate())


@app.command()
def chat(
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-m", help="Max tokens per response"),
) -> None:
    """Interactive chat session."""
    config = get_config()
    engine = get_engine()

    if temperature is not None:
        config.inference.temperature = temperature
    if max_tokens is not None:
        config.inference.max_tokens = max_tokens

    console.print(Panel.fit(
        f"[bold green]MoE Ultra Engine Chat[/bold green]\n"
        f"Model: {config.model.name}\n"
        f"Type 'exit' or 'quit' to end session\n"
        f"Type '/clear' to clear history\n"
        f"Type '/help' for commands",
        title="Chat Session",
        border_style="green",
    ))

    if system_prompt:
        console.print(f"[dim]System: {system_prompt}[/dim]\n")

    async def _chat() -> None:
        await engine.initialize()

        history: List[Dict[str, str]] = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended[/dim]")
                break

            if user_input.lower() in ("exit", "quit"):
                break
            elif user_input == "/clear":
                history = []
                if system_prompt:
                    history.append({"role": "system", "content": system_prompt})
                console.print("[dim]History cleared[/dim]")
                continue
            elif user_input == "/help":
                console.print("[dim]Commands: /clear, /help, exit, quit[/dim]")
                continue
            elif not user_input.strip():
                continue

            history.append({"role": "user", "content": user_input})

            console.print("[bold green]Assistant:[/bold green] ", end="")
            full_response = ""
            async for token in engine.chat_stream(history, config.inference):
                console.print(token, end="", highlight=False)
                full_response += token
            console.print()

            history.append({"role": "assistant", "content": full_response})

        await engine.shutdown()

    asyncio.run(_chat())


@app.command()
def benchmark(
    prompt: str = typer.Option("The future of AI is", "--prompt", help="Benchmark prompt"),
    num_runs: int = typer.Option(5, "--runs", "-n", help="Number of benchmark runs"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-m", help="Tokens per run"),
    batch_sizes: List[int] = typer.Option([1], "--batch-sizes", "-b", help="Batch sizes to test"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
) -> None:
    """Run inference benchmarks."""
    config = get_config()
    engine = get_engine()

    console.print(Panel.fit(
        f"[bold]Benchmark Configuration[/bold]\n"
        f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}\n"
        f"Runs: {num_runs}\n"
        f"Max tokens: {max_tokens}\n"
        f"Batch sizes: {batch_sizes}",
        title="Benchmark",
        border_style="yellow",
    ))

    async def _benchmark() -> None:
        await engine.initialize()

        results = []
        for batch_size in batch_sizes:
            config.inference.batch_size = batch_size
            config.inference.max_tokens = max_tokens

            console.print(f"\n[bold]Testing batch_size={batch_size}[/bold]")
            run_times = []
            token_counts = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Batch {batch_size}", total=num_runs)

                for i in range(num_runs):
                    import time
                    start = time.perf_counter()
                    result = await engine.generate(prompt, config.inference)
                    elapsed = time.perf_counter() - start

                    tokens_generated = len(result.split())  # Rough estimate
                    run_times.append(elapsed)
                    token_counts.append(tokens_generated)

                    progress.update(task, advance=1)

            avg_time = sum(run_times) / len(run_times)
            avg_tokens = sum(token_counts) / len(token_counts)
            tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

            results.append({
                "batch_size": batch_size,
                "runs": num_runs,
                "avg_time_sec": round(avg_time, 3),
                "avg_tokens": round(avg_tokens, 1),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "times_sec": [round(t, 3) for t in run_times],
            })

            console.print(f"  Avg time: {avg_time:.3f}s")
            console.print(f"  Avg tokens: {avg_tokens:.1f}")
            console.print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")

        await engine.shutdown()

        # Display summary table
        table = Table(title="Benchmark Results")
        table.add_column("Batch Size", justify="right")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Tokens/sec", justify="right")

        for r in results:
            table.add_row(
                str(r["batch_size"]),
                f"{r['avg_time_sec']:.3f}",
                f"{r['avg_tokens']:.1f}",
                f"{r['tokens_per_sec']:.2f}",
            )
        console.print(table)

        # Save to file if requested
        if output:
            import json
            output.write_text(json.dumps({
                "model": config.model.name,
                "prompt": prompt,
                "results": results,
            }, indent=2))
            console.print(f"[green]Results saved to {output}[/green]")

    asyncio.run(_benchmark())


@app.command()
def download_model(
    model_id: str = typer.Argument(..., help="Model ID (e.g., Qwen/Qwen2.5-7B-Instruct)"),
    output_dir: Path = typer.Option(Path("./models"), "--output", "-o", help="Output directory"),
    quantization: str = typer.Option("int4", "--quant", help="Quantization (none/int8/int4/gptq/awq)"),
    revision: str = typer.Option("main", "--revision", help="Model revision/branch"),
    token: Optional[str] = typer.Option(None, "--token", help="Hugging Face token"),
) -> None:
    """Download and optionally quantize a model from Hugging Face Hub."""
    from scripts.download_model import download_model as _download_model

    console.print(Panel.fit(
        f"[bold]Downloading Model[/bold]\n"
        f"Model: {model_id}\n"
        f"Output: {output_dir}\n"
        f"Quantization: {quantization}\n"
        f"Revision: {revision}",
        title="Download Model",
        border_style="blue",
    ))

    asyncio.run(_download_model(
        model_id=model_id,
        output_dir=output_dir,
        quantization=quantization,
        revision=revision,
        token=token,
        console=console,
    ))


@app.command()
def quantize(
    input_path: Path = typer.Argument(..., help="Path to model directory"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    quantization: str = typer.Option("int4", "--quant", help="Quantization method (int4/int8/gptq/awq)"),
    group_size: int = typer.Option(128, "--group-size", help="Quantization group size"),
    calibration_data: Optional[Path] = typer.Option(None, "--calibration", help="Calibration dataset path"),
) -> None:
    """Quantize a model to lower precision."""
    from scripts.quantize_model import quantize_model as _quantize_model

    console.print(Panel.fit(
        f"[bold]Quantizing Model[/bold]\n"
        f"Input: {input_path}\n"
        f"Output: {output_path}\n"
        f"Method: {quantization}\n"
        f"Group size: {group_size}",
        title="Quantize Model",
        border_style="yellow",
    ))

    asyncio.run(_quantize_model(
        input_path=input_path,
        output_path=output_path,
        quantization=quantization,
        group_size=group_size,
        calibration_data=calibration_data,
        console=console,
    ))


@app.command()
def convert_gguf(
    input_path: Path = typer.Argument(..., help="Path to model directory"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output GGUF file"),
    quantization: str = typer.Option("q4_k_m", "--quant", help="GGUF quantization type"),
) -> None:
    """Convert model to GGUF format."""
    from scripts.convert_gguf import convert_to_gguf as _convert_gguf

    console.print(Panel.fit(
        f"[bold]Converting to GGUF[/bold]\n"
        f"Input: {input_path}\n"
        f"Output: {output_path}\n"
        f"Quantization: {quantization}",
        title="Convert to GGUF",
        border_style="magenta",
    ))

    asyncio.run(_convert_gguf(
        input_path=input_path,
        output_path=output_path,
        quantization=quantization,
        console=console,
    ))


@app.command()
def config_show(
    format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml/json)"),
) -> None:
    """Show current configuration."""
    config = get_config()

    if format.lower() == "json":
        import json
        console.print(Syntax(json.dumps(config.to_dict(), indent=2, default=str), "json"))
    else:
        import yaml
        console.print(Syntax(yaml.dump(config.to_dict(), default_flow_style=False), "yaml"))


@app.command()
def config_validate(
    config_file: Path = typer.Argument(..., help="Config file to validate"),
) -> None:
    """Validate a configuration file."""
    try:
        cfg = load_config(config_file=config_file)
        console.print(Panel(
            f"[green]✓ Configuration is valid[/green]\n"
            f"Model: {cfg.model.name}\n"
            f"Max RAM: {cfg.memory.max_ram_gb}GB\n"
            f"Server: {cfg.server.host}:{cfg.server.port}",
            title="Validation Result",
            border_style="green",
        ))
    except Exception as e:
        console.print(Panel(
            f"[red]✗ Configuration invalid:[/red]\n{str(e)}",
            title="Validation Error",
            border_style="red",
        ))
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Check system compatibility and diagnose issues."""
    import platform
    import psutil
    import torch

    console.print(Panel.fit(
        "[bold]MoE Ultra Engine - System Diagnostics[/bold]",
        border_style="blue",
    ))

    # System info
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("OS", f"{platform.system()} {platform.release()}")
    table.add_row("Architecture", platform.machine())
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "unknown")
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            table.add_row(f"  GPU {i}", f"{props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Memory
    mem = psutil.virtual_memory()
    table.add_row("Total RAM", f"{mem.total / 1e9:.1f} GB")
    table.add_row("Available RAM", f"{mem.available / 1e9:.1f} GB")
    table.add_row("RAM Usage", f"{mem.percent}%")

    # Disk
    disk = psutil.disk_usage("/")
    table.add_row("Disk Free", f"{disk.free / 1e9:.1f} GB")

    console.print(table)

    # Compatibility checks
    console.print("\n[bold]Compatibility Checks:[/bold]")
    checks = []

    # RAM check
    ram_gb = mem.total / 1e9
    if ram_gb >= 32:
        checks.append(("RAM ≥ 32GB", True, f"{ram_gb:.1f} GB available"))
    elif ram_gb >= 16:
        checks.append(("RAM ≥ 16GB", True, f"{ram_gb:.1f} GB available (limited model support)"))
    else:
        checks.append(("RAM ≥ 16GB", False, f"Only {ram_gb:.1f} GB available - insufficient"))

    # CPU check
    cpu_count = psutil.cpu_count(logical=False) or 0
    if cpu_count >= 8:
        checks.append(("CPU cores ≥ 8", True, f"{cpu_count} physical cores"))
    else:
        checks.append(("CPU cores ≥ 8", False, f"Only {cpu_count} physical cores"))

    # AVX2 check
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        flags = cpu_info.get("flags", [])
        has_avx2 = "avx2" in flags
        checks.append(("AVX2 support", has_avx2, "Required for optimal CPU inference"))
    except ImportError:
        checks.append(("AVX2 support", None, "Install 'py-cpuinfo' to check"))

    # PyTorch compile check
    has_compile = hasattr(torch, "compile")
    checks.append(("torch.compile", has_compile, "PyTorch 2.0+ required"))

    # Flash attention check
    try:
        import flash_attn
        checks.append(("Flash Attention", True, f"v{flash_attn.__version__}"))
    except ImportError:
        checks.append(("Flash Attention", False, "Install for faster attention (optional)"))

    # Display checks
    check_table = Table()
    check_table.add_column("Check", style="cyan")
    check_table.add_column("Status")
    check_table.add_column("Details", style="dim")

    for name, passed, details in checks:
        if passed is True:
            status = "[green]✓ PASS[/green]"
        elif passed is False:
            status = "[red]✗ FAIL[/red]"
        else:
            status = "[yellow]? UNKNOWN[/yellow]"
        check_table.add_row(name, status, details)

    console.print(check_table)


if __name__ == "__main__":
    app()
