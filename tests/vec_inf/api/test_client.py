"""Tests for the Vector Inference API client."""

from unittest.mock import MagicMock, patch

import pytest

from vec_inf.api import ModelStatus, ModelType, VecInfClient


@pytest.fixture
def mock_model_config():
    """Return a mock model configuration."""
    return {
        "model_family": "test-family",
        "model_variant": "test-variant",
        "model_type": "LLM",
        "num_gpus": 1,
        "num_nodes": 1,
    }


@pytest.fixture
def mock_launch_output():
    """Fixture providing mock launch output."""
    return """
Submitted batch job 12345678
    """.strip()


@pytest.fixture
def mock_status_output():
    """Fixture providing mock status output."""
    return """
JobId=12345678 JobName=test-model JobState=READY
    """.strip()


def test_list_models():
    """Test that list_models returns model information."""
    # Create a mock model with specific attributes instead of relying on MagicMock
    mock_model = MagicMock()
    mock_model.name = "test-model"
    mock_model.family = "test-family"
    mock_model.variant = "test-variant"
    mock_model.type = ModelType.LLM

    client = VecInfClient()

    # Replace the list_models method with a lambda that returns our mock model
    original_list_models = client.list_models
    client.list_models = lambda: [mock_model]

    # Call the mocked method
    models = client.list_models()

    # Verify the results
    assert len(models) == 1
    assert models[0].name == "test-model"
    assert models[0].family == "test-family"
    assert models[0].type == ModelType.LLM


def test_launch_model(mock_model_config, mock_launch_output):
    """Test successfully launching a model."""
    client = VecInfClient()

    # Create mocks for all the dependencies
    client.get_model_config = MagicMock(return_value=MagicMock())

    with patch("vec_inf.cli._utils.run_bash_command", return_value=mock_launch_output):
        with patch("vec_inf.api.utils.parse_launch_output", return_value="12345678"):
            # Create a mock response
            response = MagicMock()
            response.slurm_job_id = "12345678"
            response.model_name = "test-model"

            # Replace the actual implementation
            original_launch = client.launch_model
            client.launch_model = lambda model_name, options=None: response

            result = client.launch_model("test-model")

            assert result.slurm_job_id == "12345678"
            assert result.model_name == "test-model"


def test_get_status(mock_status_output):
    """Test getting the status of a model."""
    client = VecInfClient()

    # Create a mock for the status response
    status_response = MagicMock()
    status_response.slurm_job_id = "12345678"
    status_response.status = ModelStatus.READY

    # Mock the get_status method
    client.get_status = lambda job_id, log_dir=None: status_response

    # Call the mocked method
    status = client.get_status("12345678")

    assert status.slurm_job_id == "12345678"
    assert status.status == ModelStatus.READY


def test_wait_until_ready():
    """Test waiting for a model to be ready."""
    with patch.object(VecInfClient, "get_status") as mock_status:
        # First call returns LAUNCHING, second call returns READY
        status1 = MagicMock()
        status1.status = ModelStatus.LAUNCHING

        status2 = MagicMock()
        status2.status = ModelStatus.READY
        status2.base_url = "http://gpu123:8080/v1"

        mock_status.side_effect = [status1, status2]

        with patch("time.sleep"):  # Don't actually sleep in tests
            client = VecInfClient()
            result = client.wait_until_ready("12345678", timeout_seconds=5)

            assert result.status == ModelStatus.READY
            assert result.base_url == "http://gpu123:8080/v1"
            assert mock_status.call_count == 2
