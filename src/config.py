from datetime import date
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent

TODAY = date.today()
SEEDS = [5768, 78516, 944601]
QUANTIZATION = "fp16"
