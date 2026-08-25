"""
Structured logging utilities for MoE Ultra Engine.

Supports JSON and text formats, file rotation, and structured context.
"""

import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from contextvars import ContextVar
import json
from datetime import datetime, timezone


# Context variable for request-scoped logging
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""

    def __init__(self, include_extra: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request/session context if available
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id

        session_id = session_id_var.get()
        if session_id:
            log_entry["session_id"] = session_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "msg", "pathname", "process", "processName",
                    "relativeCreated", "thread", "threadName", "exc_info",
                    "exc_text", "stack_info", "getMessage"
                }:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text log formatter."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            *args,
            **kwargs,
        )

    def format(self, record: logging.LogRecord) -> str:
        # Add context to message
        request_id = request_id_var.get()
        session_id = session_id_var.get()
        context_parts = []
        if request_id:
            context_parts.append(f"req={request_id[:8]}")
        if session_id:
            context_parts.append(f"sess={session_id[:8]}")

        if context_parts:
            record.msg = f"[{' '.join(context_parts)}] {record.msg}"

        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    log_rotation: str = "1 day",
    log_retention: str = "30 days",
    enable_console: bool = True,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "text")
        log_file: Path to log file (optional)
        log_rotation: Rotation interval (e.g., "1 day", "100 MB")
        log_retention: Retention period (e.g., "30 days")
        enable_console: Whether to log to console

    Returns:
        Root logger instance
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse rotation
        rotation_bytes = _parse_size(log_rotation)
        if rotation_bytes:
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=rotation_bytes,
                backupCount=_parse_retention_count(log_retention),
                encoding="utf-8",
            )
        else:
            # Time-based rotation
            when = _parse_time_rotation(log_rotation)
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_path,
                when=when,
                interval=1,
                backupCount=_parse_retention_count(log_retention),
                encoding="utf-8",
            )

        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root_logger


def _parse_size(size_str: str) -> Optional[int]:
    """Parse size string like '100 MB' to bytes."""
    size_str = size_str.strip().upper()
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                return int(float(size_str[:-len(unit)].strip()) * multiplier)
            except ValueError:
                pass
    return None


def _parse_time_rotation(rotation_str: str) -> str:
    """Parse time rotation string to TimedRotatingFileHandler 'when' parameter."""
    rotation_str = rotation_str.strip().lower()
    if rotation_str.startswith("second"):
        return "S"
    elif rotation_str.startswith("minute"):
        return "M"
    elif rotation_str.startswith("hour"):
        return "H"
    elif rotation_str.startswith("day"):
        return "D"
    elif rotation_str.startswith("week"):
        return "W0"
    elif rotation_str.startswith("month"):
        return "M"
    return "D"  # Default to daily


def _parse_retention_count(retention_str: str) -> int:
    """Parse retention string to backup count."""
    retention_str = retention_str.strip().lower()
    try:
        if "day" in retention_str:
            return int(float(retention_str.split()[0]))
        elif "week" in retention_str:
            return int(float(retention_str.split()[0])) * 7
        elif "month" in retention_str:
            return int(float(retention_str.split()[0])) * 30
    except (ValueError, IndexError):
        pass
    return 30  # Default


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.old_values: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        # Store old values and set new ones on the logger's extra
        logger = logging.getLogger()
        for key, value in self.kwargs.items():
            # We can't easily modify handler formatters at runtime,
            # so we use the contextvars approach for request/session IDs
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def set_request_id(request_id: str) -> None:
    """Set request ID for current context."""
    request_id_var.set(request_id)


def set_session_id(session_id: str) -> None:
    """Set session ID for current context."""
    session_id_var.set(session_id)


def clear_context() -> None:
    """Clear request/session context."""
    request_id_var.set(None)
    session_id_var.set(None)


# Performance logging helper
class PerformanceLogger:
    """Helper for logging performance metrics."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def __enter__(self) -> "PerformanceLogger":
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.logger.info(
                f"{self.operation} completed",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(duration_ms, 2),
                    "success": exc_type is None,
                    **self.metadata,
                },
            )

    def add_metadata(self, **kwargs: Any) -> "PerformanceLogger":
        """Add metadata to the performance log entry."""
        self.metadata.update(kwargs)
        return self
