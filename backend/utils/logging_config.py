"""
JSON logging configuration for voicebox backend.

Configures the root logger and all uvicorn/asyncio loggers to emit
structured JSON via python-json-logger. Call configure_json_logging()
as early as possible — before any other imports that might log.
"""

import logging
import os
import sys

from pythonjsonlogger.jsonlogger import JsonFormatter

# Uvicorn owns these logger names; we override them so nothing slips through as plain text.
_UVICORN_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
    "fastapi",
)


def configure_json_logging(log_level: str | None = None) -> None:
    """Install JSON formatter on all loggers and suppress uvicorn's own log config."""

    level_str = (log_level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    formatter = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Root logger — catches everything not explicitly handled elsewhere
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Explicitly reconfigure uvicorn's loggers so they inherit our handler
    for name in _UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.setLevel(level)
        lg.propagate = False  # prevent double-logging via root
