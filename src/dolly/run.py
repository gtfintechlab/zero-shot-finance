"""
IMPLEMENTED:
- sentiment_analysis
- numclaim_detection
- fomc_communication

NOT IMPLEMENTED (YET):
- finer_ord
"""

import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from time import time

import numpy as np
import pandas as pd
import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from tqdm.auto import tqdm
from src.config import SEEDS, TODAY
from src.dolly.model import get_model
from src.instructions import task_data_map
from src.utils.logging import setup_logger

logger = setup_logger(__name__)
from src.args import parse_args


def main(args):

    data_category = task_data_map[args.task_name]["data_category"]
    instruction = task_data_map[args.task_name]["instruction"]

    # get model and tokenizer
    model, tokenizer = get_model(args)

    # get pipeline ready for instruction text generation
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    for seed in tqdm(SEEDS):
        logger.info(f"Running inference for seed {seed}")

        # assign seed to numpy and PyTorch
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Setup directories and filepaths
        # TODO: I shouldn't need to make the data or test directory -- they should exist -- if they dont, throw an error!
        DATA_DIRECTORY = ROOT_DIRECTORY / "data"
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        TASK_DIRECTORY = DATA_DIRECTORY / args.task_name
        TASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        TEST_DIRECTORY = TASK_DIRECTORY / "test"
        TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        PROMPT_OUTPUTS = TASK_DIRECTORY / "llm_prompt_outputs" / args.quantization
        PROMPT_OUTPUTS.mkdir(parents=True, exist_ok=True)

        start_t = time()
        test_data_fp = TEST_DIRECTORY / f"{data_category}-test-{seed}.xlsx"
        logger.info(f"Loading test data from {test_data_fp}")
        data_df = pd.read_excel(test_data_fp)
        sentences = data_df["sentence"].to_list()
        logger.debug(f"Number of sentences: {len(sentences)}")
        labels = data_df["label"].to_numpy()
        logger.debug(f"Number of labels: {len(labels)}")

        prompts_list = []
        for sen in tqdm(sentences, desc="Generating prompts"):
            prompt = instruction + sen
            prompts_list.append(prompt)

        logger.info("Prompts generated. Running model inference...")
        res = generate_text(prompts_list)

        logger.info("Model inference completed. Processing outputs...")
        output_list = []
        for i in range(len(res)):
            output_list.append([labels[i], sentences[i], res[i][0]["generated_text"]])
        logger.debug(f"Number of outputs: {len(output_list)}")
        time_taken = int((time() - start_t) / 60.0)

        results = pd.DataFrame(
            output_list, columns=["true_label", "original_sent", "text_output"]
        )
        results_fp = f"dolly_{seed}_{TODAY.strftime('%d_%m_%Y')}_{time_taken}.csv"
        logger.info(f"Time taken: {time_taken} minutes")
        results.to_csv(
            PROMPT_OUTPUTS / results_fp,
            index=False,
        )
        logger.info(f"Results saved to {PROMPT_OUTPUTS / results_fp}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
