import logging
import os
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: Optional[str | int]) -> int:
    if isinstance(level, int):
        return level

    effective_level = str(level or os.getenv("LOG_LEVEL", "INFO")).upper()
    return getattr(logging, effective_level, logging.INFO)


def configure_logging(
    *,
    level: Optional[str | int] = None,
    log_file: Optional[Path] = None,
    reset_handlers: bool = False,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure project logging once and optionally attach a file handler."""
    root_logger = logging.getLogger()
    resolved_level = _resolve_level(level)

    if reset_handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    root_logger.setLevel(resolved_level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_path = str(log_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved_log_path
            for handler in root_logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    target_logger = logging.getLogger(logger_name) if logger_name else root_logger
    target_logger.setLevel(resolved_level)
    return target_logger


def get_logger(name: str, *, level: Optional[str | int] = None) -> logging.Logger:
    """Return a named logger after ensuring the project logging is configured."""
    configure_logging(level=level)
    return logging.getLogger(name)
