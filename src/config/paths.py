import os
from pathlib import Path

# Get the current working directory (project root)
PROJECT_ROOT = Path(os.getcwd()).resolve()
CONFIG_DIR = PROJECT_ROOT/".conf"
