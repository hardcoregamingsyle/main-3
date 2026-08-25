"""
Command-line interface for MoE Ultra Engine.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from .config import MoEConfig
from .engine import MoEEngine, create_engine


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the inference server."""
    setup_logging(args.log_level)
    
    overrides = {}
    if args.model_path:
        overrides.setdefault('engine', {})['model_path'] = args.model_path
    if args.config:
        overrides.setdefault('engine', {})['config'] = args.config
    
    try:
        with create_engine(args.config or "config/default.yaml", **overrides) as engine:
            print(f"MoE Ultra Engine started: {engine.engine_config.model_name}")
            print(f"Model: {engine.engine_config.num_layers}L x {engine.engine_config.num_experts_per_layer}E")
            print(f"Listening on {args.host}:{args.port}")
            
            # Import and run API server
            from api.main import create_app
            import uvicorn
            
            app = create_app(engine)
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    except Exception as e:
        logging.error(f"Server failed: {e}")
        return 1
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Run text generation."""
    setup_logging(args.log_level)
    
    overrides = {}
    if args.model_path:
        overrides.setdefault('engine', {})['model_path'] = args.model_path
    
    try:
        with create_engine(args.config or "config/default.yaml", **overrides) as engine:
            # Simple tokenizer (replace with actual tokenizer)
            def encode(text: str) -> List[int]:
                return [ord(c) % engine.engine_config.vocab_size for c in text]
            
            def decode(tokens: List[int]) -> str:
                return ''.join(chr(t % 256) for t in tokens)
            
            input_ids = np.array([encode(args.prompt)], dtype=np.int64)
            
            print(f"Prompt: {args.prompt}")
            print(f"Generating {args.max_tokens} tokens...")
            
            output_ids = engine.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            
            generated_text = decode(output_ids[0].tolist())
            print(f"Generated: {generated_text}")
            
            metrics = engine.get_metrics()
            print(f"\nMetrics:")
            print(f"  Tokens generated: {metrics['tokens_generated']}")
            print(f"  Avg latency: {metrics.get('avg_latency_ms_per_token', 0):.2f} ms/token")
            print(f"  Throughput: {metrics.get('tokens_per_second', 0):.2f} tokens/sec")
            print(f"  Cache hit rate: {metrics.get('cache_hit_rate', 0):.2%}")
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return 1
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark."""
    setup_logging(args.log_level)
    
    overrides = {}
    if args.model_path:
        overrides.setdefault('engine', {})['model_path'] = args.model_path
    
    try:
        with create_engine(args.config or "config/default.yaml", **overrides) as engine:
            import numpy as np
            
            print(f"Running benchmark: {args.iterations} iterations, {args.tokens_per_iter} tokens each")
            
            latencies = []
            for i in range(args.iterations):
                input_ids = np.random.randint(0, engine.engine_config.vocab_size,
                                              (1, args.prompt_length), dtype=np.int64)
                
                start = time.perf_counter()
                engine.generate(input_ids, max_new_tokens=args.tokens_per_iter)
                elapsed = time.perf_counter() - start
                
                latencies.append(elapsed / args.tokens_per_iter * 1000)  # ms per token
                
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i + 1}/{args.iterations}: {latencies[-1]:.2f} ms/token")
            
            latencies = np.array(latencies)
            print(f"\nBenchmark Results:")
            print(f"  Mean: {np.mean(latencies):.2f} ms/token")
            print(f"  Median: {np.median(latencies):.2f} ms/token")
            print(f"  Std: {np.std(latencies):.2f} ms/token")
            print(f"  Min: {np.min(latencies):.2f} ms/token")
            print(f"  Max: {np.max(latencies):.2f} ms/token")
            print(f"  P95: {np.percentile(latencies, 95):.2f} ms/token")
            print(f"  P99: {np.percentile(latencies, 99):.2f} ms/token")
            
            metrics = engine.get_metrics()
            print(f"  Cache hit rate: {metrics.get('cache_hit_rate', 0):.2%}")
            print(f"  Expert loads: {metrics.get('expert_loads', 0)}")
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate configuration."""
    setup_logging(args.log_level)
    
    try:
        config = MoEConfig.from_yaml(args.config)
        print(f"Configuration valid: {args.config}")
        print(f"Model: {config.engine.model_name}")
        print(f"Layers: {config.engine.num_layers}")
        print(f"Experts per layer: {config.engine.num_experts_per_layer}")
        print(f"Experts per token: {config.engine.num_experts_per_token}")
        print(f"Hidden size: {config.engine.hidden_size}")
        print(f"Estimated memory: {config.engine.estimate_total_memory_gb():.2f} GB")
        print(f"Available RAM: {config.engine.memory.available_ram_gb:.2f} GB")
        print(f"Expert cache: {config.engine.memory.expert_cache_gb:.2f} GB")
    except Exception as e:
        print(f"Configuration invalid: {e}")
        return 1
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MoE Ultra Engine - Ultra-memory-efficient MoE inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  moe-ultra serve --model-path /path/to/model
  moe-ultra generate --prompt "Hello world" --max-tokens 100
  moe-ultra benchmark --iterations 50 --tokens-per-iter 10
  moe-ultra validate --config config/prod.yaml
"""
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to configuration YAML")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start inference server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--model-path", help="Override model path")
    serve_parser.set_defaults(func=cmd_serve)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--prompt", required=True, help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    gen_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    gen_parser.add_argument("--model-path", help="Override model path")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    bench_parser.add_argument("--tokens-per-iter", type=int, default=10, help="Tokens per iteration")
    bench_parser.add_argument("--prompt-length", type=int, default=32, help="Prompt length")
    bench_parser.add_argument("--model-path", help="Override model path")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate configuration")
    val_parser.set_defaults(func=cmd_validate)
    
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    import time
    import numpy as np
    sys.exit(main())
