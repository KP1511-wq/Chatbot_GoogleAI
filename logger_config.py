import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name):
    """
    Creates a logger that writes to both Console and 'app.log'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # prevent adding handlers multiple times if function is called twice
    if logger.hasHandlers():
        return logger

    # 1. FORMATTER (How the log looks)
    # Format: [Time] [Level] [Module]: Message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. HANDLER A: Write to File (app.log)
    # Rotating: Max 5MB per file, keep last 3 backups
    file_handler = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. HANDLER B: Write to Console (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger