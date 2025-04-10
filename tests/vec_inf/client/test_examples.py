"""Tests to verify the API examples function properly."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vec_inf.client import ModelStatus, ModelType, VecInfClient


@pytest.fixture
def mock_client():
    """Create a mocked VecInfClient."""
    client = MagicMock(spec=VecInfClient)

    # Set up mock responses
    mock_model1 = MagicMock()
    mock_model1.name = "test-model"
    mock_model1.family = "test-family"
    mock_model1.type = ModelType.LLM

    mock_model2 = MagicMock()
    mock_model2.name = "test-model-2"
    mock_model2.family = "test-family-2"
    mock_model2.type = ModelType.VLM

    client.list_models.return_value = [mock_model1, mock_model2]

    launch_response = MagicMock()
    launch_response.slurm_job_id = "123456"
    launch_response.model_name = "Meta-Llama-3.1-8B-Instruct"
    client.launch_model.return_value = launch_response

    status_response = MagicMock()
    status_response.status = ModelStatus.READY
    status_response.base_url = "http://gpu123:8080/v1"
    client.wait_until_ready.return_value = status_response

    metrics_response = MagicMock()
    metrics_response.metrics = {"throughput": "10.5"}
    client.get_metrics.return_value = metrics_response

    return client


@pytest.mark.skipif(
    not (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "api"
        / "basic_usage.py"
    ).exists(),
    reason="Example file not found",
)
def test_api_usage_example():
    """Test the basic API usage example."""
    example_path = (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "api"
        / "basic_usage.py"
    )

    # Create a mock client
    mock_client = MagicMock(spec=VecInfClient)

    # Set up mock responses
    mock_model = MagicMock()
    mock_model.name = "Meta-Llama-3.1-8B-Instruct"
    mock_model.type = ModelType.LLM
    mock_client.list_models.return_value = [mock_model]

    launch_response = MagicMock()
    launch_response.slurm_job_id = "123456"
    mock_client.launch_model.return_value = launch_response

    status_response = MagicMock()
    status_response.status = ModelStatus.READY
    status_response.base_url = "http://gpu123:8080/v1"
    mock_client.wait_until_ready.return_value = status_response

    metrics_response = MagicMock()
    metrics_response.metrics = {"throughput": "10.5"}
    mock_client.get_metrics.return_value = metrics_response

    # Mock the VecInfClient class
    with (
        patch("vec_inf.client.VecInfClient", return_value=mock_client),
        patch("builtins.print"),
        example_path.open() as f,
    ):
        exec(f.read())

    # Verify the client methods were called
    mock_client.list_models.assert_called_once()
    mock_client.launch_model.assert_called_once()
    mock_client.wait_until_ready.assert_called_once()
    mock_client.get_metrics.assert_called_once()
    mock_client.shutdown_model.assert_called_once()
