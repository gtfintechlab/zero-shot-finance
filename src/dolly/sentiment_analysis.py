import os
from time import time

import numpy as np
import pandas as pd
import torch
from instruct_pipeline import InstructionTextGenerationPipeline

from src.config import QUANTIZATION, ROOT_DIRECTORY, SEEDS, TODAY
from src.dolly.model import get_dolly
from src.utils import get_gpu_with_lowest_utilization

# set gpu
cuda_device = get_gpu_with_lowest_utilization()
cuda_n_gpus = torch.cuda.device_count()
cuda_max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
cuda_max_memory = {i: cuda_max_memory for i in range(cuda_n_gpus)}
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# get model and tokenizer
model, tokenizer = get_dolly(QUANTIZATION)

# set data category
data_category = "FPB-sentiment-analysis-allagree"

# get pipeline ready for instruction text generation
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

for seed in SEEDS:
    # assign seed to numpy and PyTorch
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_t = time()
    # load test data
    test_data_path = (
        ROOT_DIRECTORY / "data" / "test" / f"{data_category}-test-{seed}.xlsx"
    )
    data_df = pd.read_excel(test_data_path)

    sentences = data_df["sentence"].to_list()
    labels = data_df["label"].to_numpy()

    prompts_list = []
    for sen in sentences:
        prompt = (
            "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
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
    results.to_csv(
        ROOT_DIRECTORY
        / "data"
        / "test"
        / "llm_prompt_outputs"
        / f"dolly_{seed}_{TODAY.strftime('%d_%m_%Y')}_{time_taken}.csv",
        index=False,
    )
