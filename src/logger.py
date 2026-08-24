import logging 
import os
from datetime import datetime


LOG_DIR= "logs"
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR,"app.log")

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(LOG_FILE),logging.StreamHandler()]
)