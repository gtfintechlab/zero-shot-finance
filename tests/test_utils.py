import subprocess

import pytest

from src.utils import get_gpu_with_lowest_utilization


@pytest.fixture
def mock_nvidia_smi_output(mocker):
    # Define a sample nvidia-smi output with three GPUs and utilization values
    mock_output = b"""
    0, 10
    1, 5
    2, 20
    """
    # Mock subprocess.check_output to return the sample output
    mocker.patch("subprocess.check_output", return_value=mock_output)


@pytest.fixture
def mock_nvidia_smi_error(mocker):
    # Mock subprocess.check_output to raise a CalledProcessError
    mocker.patch(
        "subprocess.check_output",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd="nvidia-smi"),
    )


def test_get_gpu_with_lowest_utilization(mock_nvidia_smi_output):
    # Call the function
    gpu_index = get_gpu_with_lowest_utilization()

    # Assert the result
    assert gpu_index == 1


def test_get_gpu_with_lowest_utilization_error(mock_nvidia_smi_error):
    # Call the function when nvidia-smi command raises an error
    gpu_index = get_gpu_with_lowest_utilization()

    # Assert the result is None
    assert gpu_index is None
