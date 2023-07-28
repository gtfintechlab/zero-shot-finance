import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run script with task_name and quant arguments."
    )
    parser.add_argument(
        "-t", "--task_name", type=str, required=True, help="Name of the task."
    )
    parser.add_argument(
        "-q", "--quantization", type=str, required=True, help="Quantization level."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model."
    )
    return parser.parse_args()
