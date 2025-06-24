import logging
import os
import datetime
from logging.handlers import RotatingFileHandler

log_dir = 'logs'
time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_path = os.path.join(log_dir, f'app_{time_stamp}.log')

def setup_logger(name, level=logging.INFO, if_console = True):
    """Sets up a logger that writes to a file and optionally to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers to the same logger
    if logger.hasHandlers():
        return logger

    # File handler (always on)
    # Use RotatingFileHandler to limit log file size
    fh = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Streamer Handler
    if if_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger
