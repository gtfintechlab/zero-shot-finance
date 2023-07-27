from datetime import date
import logging

TODAY = date.today()
SEEDS = (5768, 78516, 944601)
QUANTIZATION = "int8"  # "fp16"


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
