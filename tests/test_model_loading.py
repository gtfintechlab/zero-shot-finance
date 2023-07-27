import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from src.config import setup_logger

logger = setup_logger(__name__)
from src.dolly.model import get_dolly


@pytest.mark.parametrize(
    "quantization, expected_dtype",
    [
        ("fp16", torch.bfloat16),
        ("int8", torch.int8),  # Assuming int8 quantization sets the dtype to torch.int8
        ("none", torch.float32),  # Default dtype for models without quantization
    ],
)
def test_get_dolly(quantization, expected_dtype):
    logger.info(f"Testing get_dolly with quantization '{quantization}'")
    model, tokenizer = get_dolly(quantization)

    # Check if the returned objects are of the correct type
    assert isinstance(model, AutoModelForCausalLM)
    assert isinstance(tokenizer, AutoTokenizer)

    # Check if the model's dtype matches the expected dtype based on quantization
    for param in model.parameters():
        assert param.dtype == expected_dtype
        break  # Checking the dtype of the first parameter should suffice
