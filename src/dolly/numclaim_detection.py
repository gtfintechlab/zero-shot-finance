import os
import sys
from pathlib import Path
from time import time

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

import numpy as np
import pandas as pd
import torch
from instruct_pipeline import InstructionTextGenerationPipeline

from src.config import QUANTIZATION, SEEDS, TODAY
from src.dolly.model import get_dolly

# from utils.cuda_utils import get_gpu_with_lowest_utilization

# cuda_device = get_gpu_with_lowest_utilization()
cuda_n_gpus = torch.cuda.device_count()
cuda_max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
cuda_max_memory = {i: cuda_max_memory for i in range(cuda_n_gpus)}

# set gpu
device = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
device = (
    torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
)


model, tokenizer = get_dolly(QUANTIZATION)

# get pipeline ready for instruction text generation
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

for seed in SEEDS:
    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_t = time()
    # load test data
    test_data_path = ROOT_DIRECTORY / "data" / "test" / f"numclaim-test-{seed}.xlsx"
    data_df = pd.read_excel(test_data_path)

    sentences = data_df["text"].to_list()
    labels = data_df["label"].to_numpy()

    prompts_list = []
    for sen in sentences:
        prompt = (
            "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. Label 'INCLAIM' if consist of a claim and not just factual past or present information, or 'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
            + sen
        )
        prompts_list.append(prompt)

    res = generate_text(prompts_list)

    output_list = []

    for i in range(len(res)):
        output_list.append([labels[i], sentences[i], res[i][0]["generated_text"]])

    results = pd.DataFrame(
        output_list, columns=["true_label", "original_sent", "text_output"]
    )
    time_taken = int((time() - start_t) / 60.0)
    output_path = (
        ROOT_DIRECTORY
        / "data"
        / "llm_prompt_outputs"
        / f"dolly_{seed}_{TODAY.strftime('%d_%m_%Y')}_{time_taken}.csv"
    )
    results.to_csv(
        output_path,
        index=False,
    )
