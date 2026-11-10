"""Shared logging helpers for color formatting."""

ANSI_RESET = "\033[0m"
STATUS_COLORS = {
    2: "\033[32m",  # success
    3: "\033[36m",  # redirects
    4: "\033[33m",  # client errors
    5: "\033[31m",  # server errors
}

_PROCESS_TIME_COLORS = (
    (100.0, "\033[32m"),  # fast
    (500.0, "\033[33m"),  # moderate
    (float("inf"), "\033[31m"),  # slow
)


def colorize_by_status(value: int) -> str:
    """Return a colorized string for HTTP status codes."""
    family = value // 100
    color = STATUS_COLORS.get(family)
    if not color:
        return str(value)
    return f"{color}{value}{ANSI_RESET}"


def colorize_process_time(duration_ms: float) -> str:
    """Colorize process times based on latency thresholds."""
    clamped = max(duration_ms, 0.0)
    for threshold, color in _PROCESS_TIME_COLORS:
        if clamped <= threshold:
            return f"{color}{clamped:.2f}ms{ANSI_RESET}"
    return f"{clamped:.2f}ms"
