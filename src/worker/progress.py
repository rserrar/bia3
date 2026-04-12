from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable


_reporter_ctx: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("worker_progress_reporter", default=None)


def set_reporter(reporter: Callable[[dict[str, Any]], None] | None) -> None:
    _reporter_ctx.set(reporter)


def clear_reporter() -> None:
    _reporter_ctx.set(None)


def report_progress(progress: dict[str, Any]) -> None:
    reporter = _reporter_ctx.get()
    if reporter is None:
        return
    try:
        reporter(progress)
    except Exception:
        pass
