"""
Logging utilities for MoE Ultra Engine.

Provides structured JSON logging, log levels, and file output.
"""

import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextvars import ContextVar


request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter with request context."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'message', 'name', 'pathname', 'process', 'processName',
                    'relativeCreated', 'thread', 'threadName', 'exc_info',
                    'exc_text', 'stack_info', 'getMessage'
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        request_id = request_id_var.get()
        req_str = f" [{request_id[:8]}]" if request_id else ""
        
        return (
            f"{color}{timestamp}{reset}"
            f" {color}{record.levelname:<8}{reset}"
            f" {record.name}{req_str}: "
            f"{record.getMessage()}"
        )


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
) -> None:
    """Configure application logging."""
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if log_format.lower() == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredConsoleFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)
    
    logging.info(
        f"Logging configured: level={level}, format={log_format}, "
        f"file={log_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding request context to logs."""
    
    def __init__(self, request_id: Optional[str] = None, **extra):
        self.request_id = request_id or f"req_{datetime.utcnow().timestamp()}"
        self.extra = extra
        self.token = None
    
    def __enter__(self):
        self.token = request_id_var.set(self.request_id)
        # Add extra to all loggers (simplified)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_var.reset(self.token)


def log_inference_request(
    logger: logging.Logger,
    request_id: str,
    prompt_length: int,
    max_tokens: int,
    temperature: float,
) -> None:
    """Log inference request."""
    logger.info(
        "Inference request",
        extra={
            "request_id": request_id,
            "prompt_tokens": prompt_length,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "event_type": "inference_request",
        }
    )


def log_inference_response(
    logger: logging.Logger,
    request_id: str,
    tokens_generated: int,
    generation_time: float,
    tokens_per_second: float,
    finish_reason: str,
) -> None:
    """Log inference response."""
    logger.info(
        "Inference response",
        extra={
            "request_id": request_id,
            "tokens_generated": tokens_generated,
            "generation_time_ms": generation_time * 1000,
            "tokens_per_second": tokens_per_second,
            "finish_reason": finish_reason,
            "event_type": "inference_response",
        }
    )


def log_expert_load(
    logger: logging.Logger,
    layer_idx: int,
    expert_idx: int,
    load_time_ms: float,
    from_cache: bool,
) -> None:
    """Log expert weight loading."""
    logger.debug(
        "Expert loaded",
        extra={
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "load_time_ms": load_time_ms,
            "from_cache": from_cache,
            "event_type": "expert_load",
        }
    )


def log_memory_usage(
    logger: logging.Logger,
    used_gb: float,
    limit_gb: float,
    component: str,
) -> None:
    """Log memory usage."""
    logger.info(
        "Memory usage",
        extra={
            "used_gb": used_gb,
            "limit_gb": limit_gb,
            "usage_pct": (used_gb / limit_gb * 100) if limit_gb > 0 else 0,
            "component": component,
            "event_type": "memory_usage",
        }
    )
