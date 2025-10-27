"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path


def setup_logging(app):
    """Configure application logging."""
    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Set log level
    log_level = app.config.get("LOG_LEVEL", "INFO")
    app.logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    app.logger.addHandler(console_handler)
    app.logger.addHandler(file_handler)
    
    app.logger.info("Logging configured successfully")
