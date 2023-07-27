import re
import subprocess


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


# Test the function
if __name__ == "__main__":
    gpu_index = get_gpu_with_lowest_utilization()
    if gpu_index is not None:
        print(f"The GPU with the lowest utilization is GPU-{gpu_index}.")
    else:
        print("Failed to retrieve GPU utilization information.")
