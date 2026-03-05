import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "q-data-pipeline",
    log_file: str = "app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup a logger with console and file handlers.

    Args:
        name: Name of the logger.
        log_file: Name of the log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers are already added to avoid duplicates
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create file handler
        file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Create a default logger instance
logger = setup_logger()
