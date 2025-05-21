"""
Logging configuration for OpenReasoning.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import settings


# Configure root logger
def configure_logging(log_level: str = None, log_file: Optional[str] = None):
    """Configure logging for the application."""
    level = log_level or settings.log_level
    numeric_level = getattr(logging, level)

    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )

    # Set library loggers to WARNING unless in DEBUG mode
    if level != "DEBUG":
        for logger_name in ["urllib3", "asyncio", "requests", "openai"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


class JsonLogger:
    """Logger that outputs structured JSON logs."""

    def __init__(self, log_dir: str = "logs", app_name: str = "openreasoning"):
        """Initialize the JSON logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.app_name = app_name

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{self.app_name}_{timestamp}.jsonl"

    def log(self, event: str, data: Dict[str, Any]):
        """Log an event with structured data."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "app_name": self.app_name,
            "event": event,
            "data": data,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_model_request(
        self, provider: str, model: str, request_data: Dict[str, Any]
    ):
        """Log a model request."""
        self.log(
            "model_request",
            {"provider": provider, "model": model, "request": request_data},
        )

    def log_model_response(
        self, provider: str, model: str, response_data: Dict[str, Any]
    ):
        """Log a model response."""
        self.log(
            "model_response",
            {"provider": provider, "model": model, "response": response_data},
        )

    def log_error(
        self, error_message: str, error_type: str, context: Dict[str, Any] = None
    ):
        """Log an error."""
        self.log(
            "error",
            {
                "error_message": error_message,
                "error_type": error_type,
                "context": context or {},
            },
        )


# Initialize JSON logger
json_logger = JsonLogger()
