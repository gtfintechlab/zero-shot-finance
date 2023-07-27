import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dolly.sentiment_analysis import cuda_max_memory


def get_dolly(QUANTIZATION):
    model_name = "databricks/dolly-v2-12b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if QUANTIZATION == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory=cuda_max_memory,
        )
    elif QUANTIZATION == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True, max_memory=cuda_max_memory
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", max_memory=cuda_max_memory
        )
    return model, tokenizer
