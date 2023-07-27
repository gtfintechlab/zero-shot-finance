import re
import subprocess

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def get_gpu_with_lowest_utilization():
    try:
        # Run the nvidia-smi command to get GPU utilization information
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        gpu_info = output.decode().strip().split("\n")

        # Parse the output to find the GPU with the lowest utilization
        lowest_utilization = float("inf")
        gpu_index = -1

        for info in gpu_info:
            # Use regex to extract index and utilization information
            match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", info)
            if match:
                index, utilization = map(int, match.groups())
                if utilization < lowest_utilization:
                    lowest_utilization = utilization
                    gpu_index = index

        return gpu_index
    except subprocess.CalledProcessError as e:
        print("Error: nvidia-smi command failed.")
        return None
