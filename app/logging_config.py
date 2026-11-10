"""Logging configuration for the FastAPI application.

Provides colorized console output, timestamped file rotation, and a centralized
``configure_logging`` entry-point used by ``app.main`` and ``app.server``.
"""
from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path

from app.logging_utils import ANSI_RESET

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


class DebugAwareFormatter(logging.Formatter):
    """Append module/function info for debug-level entries."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if record.levelno <= logging.DEBUG:
            formatted = f"{formatted} [{record.name}:{record.funcName}]"
        return formatted


class ColorFormatter(DebugAwareFormatter):
    """Inject ANSI colors into the level name while keeping the base format."""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        original_level = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno)
        if color:
            record.levelname = f"{color}{original_level}{ANSI_RESET}"
        try:
            formatted = super().format(record)
        finally:
            record.levelname = original_level
        response_status = getattr(record, "response_status", None)
        if response_status is not None:
            from app.logging_utils import colorize_by_status

            colored = colorize_by_status(int(response_status))
            formatted = formatted.replace(
                f"Response: {response_status}",
                f"Response: {colored}",
                1,
            )
        return formatted


class TimestampedSizeRotatingFileHandler(logging.FileHandler):
    """Rotate log files once they exceed ``max_bytes`` with timestamped filenames."""

    def __init__(self, directory: Path, max_bytes: int, encoding: str = "utf-8") -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        initial_path = self._generate_filename()
        super().__init__(initial_path, encoding=encoding)

    def _generate_filename(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return str(self.directory / f"api-{timestamp}.log")

    def _should_rollover(self) -> bool:
        if self.max_bytes <= 0:
            return False
        if self.stream is None:
            self.stream = self._open()
        self.stream.flush()
        self.stream.seek(0, os.SEEK_END)
        return self.stream.tell() >= self.max_bytes

    def emit(self, record: logging.LogRecord) -> None:
        if self._should_rollover():
            self.doRollover()
        super().emit(record)

    def doRollover(self) -> None:
        self.acquire()
        try:
            if self.stream:
                self.stream.close()
            self.baseFilename = self._generate_filename()
            self.stream = self._open()
        finally:
            self.release()


def configure_logging(config) -> None:
    """Route logs to ``api.log`` and honor the configured log level.

    The ``config`` parameter is duck-typed so that both :class:`agent.config.AppConfig`
    and plain objects can be passed.
    """
    from app.logging_utils import colorize_process_time

    plain_formatter = DebugAwareFormatter(LOG_FORMAT)
    color_formatter = ColorFormatter(LOG_FORMAT)

    level = logging.getLevelName(config.normalized_log_level)
    if isinstance(level, str):
        level = logging.INFO

    file_handler = TimestampedSizeRotatingFileHandler(
        directory=config.log_dir,
        max_bytes=config.log_file_max_bytes,
        encoding="utf-8",
    )
    file_handler.setFormatter(plain_formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    console_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)

    for noisy in ("pdfminer", "pdfminer.psparser", "pdfminer.pdfinterp", "pdfminer.cmapdb"):
        logging.getLogger(noisy).setLevel(logging.INFO)

    for verbose in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(verbose).setLevel(logging.WARNING)
