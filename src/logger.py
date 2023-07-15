import os
import logging
import sys
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log"
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=logging_str,
    level = logging.INFO,
)