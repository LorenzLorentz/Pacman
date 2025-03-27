import logging
import logging.handlers
import sys
from typing import Optional

def get_logger(name: str = "default_logger",
               seed : int = None,
               level: int = logging.INFO,
               fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
               datefmt: str = "%Y-%m-%d %H:%M:%S",
               log_file: Optional[str] = None,
               max_bytes: int = 100 * 1024 * 1024,  # 10MB
               backup_count: int = 5,
               console: bool = False,
               file_mode: str = "a"):
    if seed:
        name += f"_SEED={seed}"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, mode=file_mode, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger