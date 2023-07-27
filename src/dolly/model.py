import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.cuda_utils import get_gpu_with_lowest_utilization
from src.config import setup_logger
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = str(ROOT_DIRECTORY / ".model_cache")

logger = setup_logger(__name__)

# set gpu
if torch.cuda.is_available():
    # device = get_gpu_with_lowest_utilization()
    device = 0
    logger.info(f"Using CUDA device '{device}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    cuda = torch.device(f"cuda:{device}")
else:
    logger.info(f"CUDA Unavailable -- using CPU")
    torch.device("cpu")

# cuda_n_gpus = torch.cuda.device_count()
# cuda_max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
# cuda_max_memory = {i: cuda_max_memory for i in range(cuda_n_gpus)}


def get_dolly(QUANTIZATION):
    model_name = "databricks/dolly-v2-12b"

    logger.info(f"Loading model '{model_name}' with quantization '{QUANTIZATION}'")
    if QUANTIZATION == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            # max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    elif QUANTIZATION == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            # device_map="auto",
            # max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map="auto",
            # max_memory=cuda_max_memory,
            cache_dir=CACHE_DIR,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    return model, tokenizer
