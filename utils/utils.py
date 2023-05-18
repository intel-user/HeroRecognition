import os
import logging
from logging.handlers import RotatingFileHandler

from config.config import Config


def setup_logging():
    config = Config()
    if not os.path.exists('logs'):
        os.mkdir('logs')
    new_format = "%(asctime)s | %(name)s | [%(levelname)s] | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=new_format,
        handlers=[
            RotatingFileHandler(config.log_file, encoding="utf8",
                                maxBytes=1024*10240, backupCount=10),
            logging.StreamHandler()
        ]
    )
    