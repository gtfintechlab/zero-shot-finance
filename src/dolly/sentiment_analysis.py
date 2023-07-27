import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))
TEST_DATA = ROOT_DIRECTORY / "data" / "test"
PROMPT_OUTPUTS = TEST_DATA / "llm_prompt_outputs"

from time import time
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
from instruct_pipeline import InstructionTextGenerationPipeline

from src.config import QUANTIZATION, SEEDS, TODAY, setup_logger
from src.dolly.model import get_dolly
from src.utils import create_batches

logger = setup_logger(__name__)

if __name__ == "__main__":
    # get model and tokenizer
    model, tokenizer = get_dolly(QUANTIZATION)

    # set data category
    data_category = "FPB-sentiment-analysis-allagree"

    # get pipeline ready for instruction text generation
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    for seed in tqdm(SEEDS):
        logger = setup_logger(f"seed_{seed}")
        # assign seed to numpy and PyTorch
        torch.manual_seed(seed)
        np.random.seed(seed)

        start_t = time()
        # load test data
        test_data_path = TEST_DATA / f"{data_category}-test-{seed}.xlsx"
        logger.info(f"Loading test data from {test_data_path}")
        data_df = pd.read_excel(test_data_path)

        sentences = data_df["sentence"].to_list()
        logger.info(f"Number of sentences: {len(sentences)}")
        labels = data_df["label"].to_numpy()
        logger.info(f"Number of labels: {len(labels)}")

        prompts_list = []
        for sen in tqdm(sentences, desc="Generating prompts"):
            prompt = (
                "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: "
                + sen
            )
            prompts_list.append(prompt)

        logger.info("Prompts generated. Running model inference...")

        BATCH_SIZE = 24  # Define your desired batch size
        all_results = []
        for batch in tqdm(
            create_batches(prompts_list, BATCH_SIZE), desc="Processing batches"
        ):
            batch_results = generate_text(batch)
            all_results.extend(batch_results)
        logger.info("Model inference completed. Processing outputs...")

        output_list = []
        for i in tqdm(range(len(all_results)), desc="Processing outputs"):
            output_list.append(
                [labels[i], sentences[i], all_results[i][0]["generated_text"]]
            )

        logger.info(f"Number of outputs: {len(output_list)}")
        results = pd.DataFrame(
            output_list, columns=["true_label", "original_sent", "text_output"]
        )
        time_taken = int((time() - start_t) / 60.0)
        logger.info(f"Time taken: {time_taken} minutes")
        results.to_csv(
            PROMPT_OUTPUTS
            / f"dolly_{seed}_{TODAY.strftime('%d_%m_%Y')}_{time_taken}.csv",
            index=False,
        )
