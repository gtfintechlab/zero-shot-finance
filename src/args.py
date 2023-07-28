import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run script with task_name and quant arguments."
    )
    # TODO: make sure the provided task name is valid
    parser.add_argument(
        "-t", "--task_name", type=str, required=True, help="Name of the task."
    )
    # TODO: make sure the provided quantization is valid
    parser.add_argument(
        "-q", "--quantization", type=str, required=True, help="Quantization level."
    )
    # TODO: make sure the provided model is valid
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model."
    )
    return parser.parse_args()
