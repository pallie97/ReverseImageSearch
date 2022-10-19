import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")