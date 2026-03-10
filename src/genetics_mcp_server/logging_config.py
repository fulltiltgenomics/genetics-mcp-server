"""
Structured JSON logging for GCP Cloud Logging.

On GKE, stdout is automatically captured by fluentbit and sent to Cloud Logging.
The JSON format allows Cloud Logging to parse severity and other fields.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class GCPJsonFormatter(logging.Formatter):
    """JSON formatter compatible with GCP Cloud Logging."""

    SEVERITY_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self.SEVERITY_MAP.get(record.levelno, "DEFAULT"),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


_logging_initialized = False


def setup_logging(level: str = "INFO"):
    """Configure structured JSON logging to stdout."""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(GCPJsonFormatter())
    root_logger.addHandler(handler)

    # suppress noisy loggers
    for name in ("uvicorn.access", "httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
