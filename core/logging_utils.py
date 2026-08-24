"""Logging utilities for MoE Ultra Engine."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import json


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter for production environments."""
    
    def __init__(self, env: str = "production") -> None:
        super().__init__()
        self.env = env
        self.hostname = __import__("socket").gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": self.env,
            "hostname": self.hostname,
            "thread": record.thread,
            "process": record.process,
        }
        
        # Add source location
        if record.filename != "<string>":
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for terminal."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        
        color = self.COLORS.get(level, "") if self.use_colors else ""
        reset = self.RESET if self.use_colors else ""
        
        message = f"[{timestamp}] {color}{level:<8}{reset} [{record.name}] {record.getMessage()}"
        
        if record.filename != "<string>" and self.use_colors:
            message += f" {color}\033[2m({record.filename}:{record.lineno})\033[0m"
        
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        return message


def setup_logging(
    level: str = "INFO",
    config: Optional[Dict[str, Any]] = None,
    env: str = "development",
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Optional configuration dictionary
        env: Environment name (development, staging, production)
    """
    config = config or {}
    
    # Determine formatter based on environment
    if env == "production" or config.get("structured", False):
        formatter = StructuredFormatter(env)
    else:
        use_colors = config.get("colors", True) and sys.stderr.isatty()
        formatter = ConsoleFormatter(use_colors)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    root_logger.addHandler(console_handler)
    
    # Add file handler if configured
    file_config = config.get("file", {})
    if file_config.get("enabled", False):
        log_path = Path(file_config.get("path", "logs/app.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, file_config.get("level", level).upper(), logging.INFO))
        
        root_logger.addHandler(file_handler)
    
    # Set log levels for specific modules
    module_levels = config.get("modules", {})
    for module_name, module_level in module_levels.items():
        logging.getLogger(module_name).setLevel(getattr(logging, module_level.upper(), logging.INFO))
    
    # Disable verbose third-party logs in non-debug modes
    if level.upper() != "DEBUG":
        for silent_module in ["urllib3", "botocore", "azure.storage", "google.auth"]:
            logging.getLogger(silent_module).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name, typically __name__ of the module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding extra fields to log records."""
    
    def __init__(self, **kwargs: Any) -> None:
        self.extra_fields = kwargs
    
    def __enter__(self) -> None:
        pass
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
    
    def patch_logger(self, logger: logging.Logger) -> logging.Logger:
        """Patch a logger to include extra fields in all records."""
        original_makeRecord = logger.makeRecord
        
        def patchedMakeRecord(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = original_makeRecord(*args, **kwargs)
            if not hasattr(record, "extra_fields"):
                record.extra_fields = {}
            record.extra_fields.update(self.extra_fields)
            return record
        
        logger.makeRecord = patchedMakeRecord  # type: ignore[method-assign]
        return logger


def get_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    import uuid
    return str(uuid.uuid4())


def inject_correlation_id(logger: logging.Logger) -> None:
    """Inject correlation ID into all log records for this logger."""
    correlation_id = get_correlation_id()
    original_makeRecord = logger.makeRecord
    
    def patchedMakeRecord(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original_makeRecord(*args, **kwargs)
        if not hasattr(record, "correlation_id"):
            record.correlation_id = correlation_id
        return record
    
    logger.makeRecord = patchedMakeRecord  # type: ignore[method-assign]
