"""
Logging utility module for the Real Estate Price Prediction project.
Provides colored console output and optional file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


# Mapping from log level to color
_LEVEL_COLORS = {
    logging.DEBUG: _Colors.CYAN,
    logging.INFO: _Colors.BRIGHT_GREEN,
    logging.WARNING: _Colors.BRIGHT_YELLOW,
    logging.ERROR: _Colors.BRIGHT_RED,
    logging.CRITICAL: _Colors.BOLD + _Colors.RED,
}


class _ColoredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log level names."""

    _FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__(fmt=self._FORMAT, datefmt=self._DATE_FORMAT)
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        if self._use_colors:
            color = _LEVEL_COLORS.get(record.levelno, _Colors.WHITE)
            record.levelname = f"{color}{record.levelname}{_Colors.RESET}"
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter without color codes — used for file handlers."""

    _FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FORMAT, datefmt=self._DATE_FORMAT)


def _supports_color(stream) -> bool:
    """Return True if *stream* supports ANSI escape codes."""
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    # Windows terminal check
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Enable VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a named logger.

    Args:
        name:     Logger name (usually ``__name__`` of the calling module).
        log_file: Optional filename (not full path) to write logs into the
                  project ``logs/`` directory.  If *None*, only console output
                  is configured.
        level:    Minimum log level (default: ``logging.INFO``).

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is already set up
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Don't bubble up to root logger

    # --- Console handler ---------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    use_colors = _supports_color(sys.stdout)
    console_handler.setFormatter(_ColoredFormatter(use_colors=use_colors))
    logger.addHandler(console_handler)

    # --- File handler (optional) -------------------------------------------
    if log_file is not None:
        logs_dir = _resolve_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / log_file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_PlainFormatter())
        logger.addHandler(file_handler)
        logger.debug("File logging enabled → %s", log_path)

    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Convenience wrapper around :func:`setup_logger`.

    Returns an existing logger if one with *name* has already been set up,
    otherwise creates a new one with console output only.

    Args:
        name:  Logger name (use ``__name__`` in calling modules).
        level: Minimum log level (default: ``logging.INFO``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return setup_logger(name, log_file=None, level=level)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_logs_dir() -> Path:
    """Return the ``logs/`` directory path relative to the project root.

    Searches upward from this file's location for a directory that contains
    a ``src/`` sub-directory (= project root), then returns ``<root>/logs``.
    Falls back to ``./logs`` if the project root cannot be determined.
    """
    this_file = Path(__file__).resolve()
    # Walk up: utils -> src -> project_root
    candidate = this_file.parent
    for _ in range(5):
        candidate = candidate.parent
        if (candidate / "src").is_dir():
            return candidate / "logs"
    # Fallback: current working directory
    return Path.cwd() / "logs"
