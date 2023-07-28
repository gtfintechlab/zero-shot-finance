import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import setup_logger

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = str(ROOT_DIRECTORY / ".model_cache")

logger = setup_logger(__name__)


def get_dolly(QUANTIZATION):
    if torch.cuda.is_available():
        cuda_n_gpus = torch.cuda.device_count()
        cuda_max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
        cuda_max_memory = {i: cuda_max_memory for i in range(cuda_n_gpus)}
        logger.info(
            f"Using k={cuda_n_gpus} CUDA GPUs with max memory {cuda_max_memory}"
        )
    else:
        logger.error(f"CUDA Unavailable!")
        raise OSError("CUDA Unavailable!")

    model_name = "databricks/dolly-v2-12b"
    logger.info(f"Loading model '{model_name}' with quantization '{QUANTIZATION}'")
    if QUANTIZATION == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    elif QUANTIZATION == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    elif QUANTIZATION == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    return model, tokenizer
