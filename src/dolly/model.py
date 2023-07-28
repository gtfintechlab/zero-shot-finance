from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import setup_logger

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = str(ROOT_DIRECTORY / ".model_cache")

logger = setup_logger(__name__)

VALID_MODELS = ["databricks/dolly-v2-12b"]

def get_model(args):
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

    if args.model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{args.model}'")

    logger.info(f"Loading model '{args.model}' with quantization '{args.quantization}'")
    if args.quantization == "default":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    elif args.quantization == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    elif args.quantization == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    elif args.quantization == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_4bit=True,
            device_map="auto",
            max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    else:
        raise ValueError(f"Invalid quantization '{args.quantization}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    return model, tokenizer
