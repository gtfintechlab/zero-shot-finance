import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent

# Get the directory of the src folder
SRC_DIRECTORY = ROOT_DIRECTORY / "src"

# Add the src folder to sys.path
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from datetime import date

TODAY = date.today()
SEEDS = [5768, 78516, 944601]
QUANTIZATION = "fp16"
