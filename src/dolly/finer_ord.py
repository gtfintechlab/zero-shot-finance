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

from src.config import QUANTIZATION, SEEDS, TODAY
from src.dolly.model import get_dolly
from src.utils.logging import setup_logger
from src.instructions import task_data_map

logger = setup_logger(__name__)

if __name__ == "__main__":
    # TODO: Have task name be a command line argument, which is mapped to the data categories
    # TODO: Make sure that numclaim uses the same dataset as sentiment analysis
    # Set task name and data category
    task_name = "finer_ord"
    data_category = task_data_map[task_name]["data_category"]
    instruction = task_data_map[task_name]["instruction"]

    # get model and tokenizer
    model, tokenizer = get_dolly(QUANTIZATION)

    # get pipeline ready for instruction text generation
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    for seed in tqdm(SEEDS):
        logger = setup_logger(f"seed_{seed}")
        # assign seed to numpy and PyTorch
        torch.manual_seed(seed)
        np.random.seed(seed)

        start_t = time()
        # load test data
        # TODO: add task name to data path
        TEST_DATA = ROOT_DIRECTORY / "data" / "test"
        TEST_DATA.mkdir(parents=True, exist_ok=True)
        test_data_path = TEST_DATA / f"{data_category}-test-{seed}.xlsx"
        logger.info(f"Loading test data from {test_data_path}")
        # TODO: make it so that the data for finer_ord is an excel sheet so I can load from that not csv
        data_df = pd.read_csv(test_data_path)

        grouped_df = (
            data_df.groupby(["doc_idx", "sent_idx"])
            .agg({"gold_label": lambda x: list(x), "gold_token": lambda x: list(x)})
            .reset_index()
        )
        grouped_df.columns = ["doc_idx", "sent_idx", "gold_label", "gold_token"]
        logger.info(f"Number of sentences: {len(grouped_df)}")
        # TODO: log a count of unique gold labels
        logger.info(f"Number of labels: <NotImplementedError>")


        prompts_list = []
        for index in range(grouped_df.shape[0]):
            token_list = grouped_df.loc[[index], ["gold_token"]].values[0, 0]
            label_list = grouped_df.loc[[index], ["gold_label"]].values[0, 0]
            sen = "\n".join(token_list)

            prompt = (
                instruction + sen
            )
            prompts_list.append(prompt)

        logger.info("Prompts generated. Running model inference...")
        res = generate_text(prompts_list)
        logger.info("Model inference completed. Processing outputs...")

        output_list = []
        for i in range(len(res)):
            token_list = grouped_df.loc[[i], ["gold_token"]].values[0, 0]
            label_list = grouped_df.loc[[i], ["gold_label"]].values[0, 0]
            sen = "\n".join(token_list)
            output_list.append([label_list, sen, res[i][0]["generated_text"]])
            logger.info(f"Number of outputs: {len(output_list)}")

        results = pd.DataFrame(
            output_list, columns=["true_label", "original_sent", "text_output"]
        )
        time_taken = int((time() - start_t) / 60.0)
        logger.info(f"Time taken: {time_taken} minutes")
        PROMPT_OUTPUTS = TEST_DATA / "llm_prompt_outputs" / task_name
        PROMPT_OUTPUTS.mkdir(parents=True, exist_ok=True)
        results_fp = f"dolly_{TODAY.strftime('%d_%m_%Y')}_{time_taken}.pkl"
        results.to_pickle(
            PROMPT_OUTPUTS / results_fp,
            index=False,
        )
