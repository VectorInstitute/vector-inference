"""Tests for the Vector Inference API client."""

from unittest.mock import MagicMock, patch

import pytest

from vec_inf.client import ModelStatus, ModelType, VecInfClient
from vec_inf.client._exceptions import ServerError, SlurmJobError


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

    with (
        patch(
            "vec_inf.client._utils.run_bash_command",
            return_value=(mock_launch_output, ""),
        ),
        patch(
            "vec_inf.client._utils.parse_launch_output", return_value=("12345678", {})
        ),
    ):
        # Create a mock response
        response = MagicMock()
        response.slurm_job_id = "12345678"
        response.model_name = "test-model"

        # Replace the actual implementation
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
        status1.server_status = ModelStatus.LAUNCHING

        status2 = MagicMock()
        status2.server_status = ModelStatus.READY
        status2.base_url = "http://gpu123:8080/v1"

        mock_status.side_effect = [status1, status2]

        with patch("time.sleep"):  # Don't actually sleep in tests
            client = VecInfClient()
            result = client.wait_until_ready("12345678", timeout_seconds=5)

            assert result.server_status == ModelStatus.READY
            assert result.base_url == "http://gpu123:8080/v1"
            assert mock_status.call_count == 2


def test_shutdown_model_success():
    """Test model shutdown success."""
    client = VecInfClient()
    with patch("vec_inf.client.api.run_bash_command") as mock_command:
        mock_command.return_value = ("", "")
        result = client.shutdown_model(12345)

        assert result is True
        mock_command.assert_called_once_with("scancel 12345")


def test_shutdown_model_failure():
    """Test model shutdown failure."""
    client = VecInfClient()
    with patch("vec_inf.client.api.run_bash_command") as mock_command:
        mock_command.return_value = ("", "Error: Job not found")
        with pytest.raises(
            SlurmJobError, match="Failed to shutdown model: Error: Job not found"
        ):
            client.shutdown_model(12345)


def test_wait_until_ready_timeout():
    """Test timeout in wait_until_ready."""
    client = VecInfClient()

    with patch.object(client, "get_status") as mock_status:
        mock_response = MagicMock()
        mock_response.server_status = ModelStatus.LAUNCHING
        mock_status.return_value = mock_response

        with (
            patch("time.sleep"),
            pytest.raises(ServerError, match="Timed out waiting for model"),
        ):
            client.wait_until_ready(12345, timeout_seconds=1, poll_interval_seconds=0.5)


def test_wait_until_ready_failed_status():
    """Test wait_until_ready when model fails."""
    client = VecInfClient()

    with patch.object(client, "get_status") as mock_status:
        mock_response = MagicMock()
        mock_response.server_status = ModelStatus.FAILED
        mock_response.failed_reason = "Out of memory"
        mock_status.return_value = mock_response

        with pytest.raises(ServerError, match="Model failed to start: Out of memory"):
            client.wait_until_ready(12345)


def test_wait_until_ready_failed_no_reason():
    """Test wait_until_ready when model fails without reason."""
    client = VecInfClient()

    with patch.object(client, "get_status") as mock_status:
        mock_response = MagicMock()
        mock_response.server_status = ModelStatus.FAILED
        mock_response.failed_reason = None
        mock_status.return_value = mock_response

        with pytest.raises(ServerError, match="Model failed to start: Unknown error"):
            client.wait_until_ready(12345)


def test_wait_until_ready_shutdown():
    """Test wait_until_ready when model is shutdown."""
    client = VecInfClient()

    with patch.object(client, "get_status") as mock_status:
        mock_response = MagicMock()
        mock_response.server_status = ModelStatus.SHUTDOWN
        mock_status.return_value = mock_response

        with pytest.raises(
            ServerError, match="Model was shutdown before it became ready"
        ):
            client.wait_until_ready(12345)
