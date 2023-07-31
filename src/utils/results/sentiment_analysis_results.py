import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent.parent
logger.debug(f"Root directory: {str(ROOT_DIRECTORY)}")
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from src.utils.results.decode import sentiment_analysis_decode
import nltk

nltk.download("punkt")


def compute_metrics(files, outputs_directory):
    acc_list = []
    f1_list = []
    missing_perc_list = []

    for file in files:
        df = pd.read_csv(outputs_directory / file)

        # Decode the predicted label
        df["predicted_label"] = df["text_output"].apply(sentiment_analysis_decode)

        # Calculate metrics
        acc_list.append(accuracy_score(df["true_label"], df["predicted_label"]))
        f1_list.append(
            f1_score(df["true_label"], df["predicted_label"], average="weighted")
        )
        missing_perc_list.append(
            (len(df[df["predicted_label"] == -1]) / df.shape[0]) * 100.0
        )

    return acc_list, f1_list, missing_perc_list


def main(args):
    LLM_OUTPUTS_DIRECTORY = (
        ROOT_DIRECTORY
        / "data"
        / args.task_name
        / "llm_prompt_outputs"
        / args.quantization
    )

    # Filter out relevant files
    files = [
        f
        for f in LLM_OUTPUTS_DIRECTORY.iterdir()
        if args.model in f.name and f.suffix == ".csv"
    ]

    acc_list, f1_list, missing_perc_list = compute_metrics(files, LLM_OUTPUTS_DIRECTORY)

    # Print results
    print("f1 score mean: ", format(np.mean(f1_list), ".4f"))
    print("f1 score std: ", format(np.std(f1_list), ".4f"))
    print(
        "Percentage of cases when didn't follow instruction: ",
        format(np.mean(missing_perc_list), ".4f"),
        "\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics for sentiment analysis results."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model used."
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        required=True,
        help="Quantization level of the model.",
    )
    parser.add_argument(
        "-t", "--task_name", type=str, required=True, help="Name of the task."
    )
    args = parser.parse_args()

    main(args)
