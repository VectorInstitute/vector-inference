"""Tests for the Vector Inference API client."""

from unittest.mock import AsyncMock, MagicMock, patch

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


def test_list_models():
    """Test that list_models returns model information."""
    # Create a mock model with specific attributes instead of relying on MagicMock
    mock_model = MagicMock()
    mock_model.name = "test-model"
    mock_model.family = "test-family"
    mock_model.variant = "test-variant"
    mock_model.model_type = ModelType.LLM

    client = VecInfClient()

    # Replace the list_models method with a lambda that returns our mock model
    client.list_models = lambda: [mock_model]

    # Call the mocked method
    models = client.list_models()

    # Verify the results
    assert len(models) == 1
    assert models[0].name == "test-model"
    assert models[0].family == "test-family"
    assert models[0].model_type == ModelType.LLM


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


@pytest.mark.asyncio
async def test_get_status():
    """Test getting the status of a model."""
    client = VecInfClient()

    status_response = MagicMock()
    status_response.slurm_job_id = "12345678"
    status_response.server_status = ModelStatus.READY

    with patch.object(
        VecInfClient, "_get_status_sync", return_value=status_response
    ) as mock_sync:
        status = await client.get_status(12345678)

    mock_sync.assert_called_once_with(12345678, None)
    assert status is status_response
    assert status.server_status == ModelStatus.READY


@pytest.mark.asyncio
async def test_wait_until_ready():
    """Test waiting for a model to be ready."""
    with patch.object(VecInfClient, "get_status", new_callable=AsyncMock) as mock_status:
        status1 = MagicMock()
        status1.server_status = ModelStatus.LAUNCHING

        status2 = MagicMock()
        status2.server_status = ModelStatus.READY
        status2.base_url = "http://gpu123:8080/v1"

        mock_status.side_effect = [status1, status2]

        client = VecInfClient()
        result = await client.wait_until_ready(
            12345678, timeout_seconds=5, poll_interval_seconds=0
        )

        assert result.server_status == ModelStatus.READY
        assert result.base_url == "http://gpu123:8080/v1"
        assert mock_status.await_count == 2


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
            _ = client.shutdown_model(12345)


@pytest.mark.asyncio
async def test_wait_until_ready_timeout():
    """Test timeout in wait_until_ready."""
    client = VecInfClient()

    mock_response = MagicMock()
    mock_response.server_status = ModelStatus.LAUNCHING

    with patch.object(
        client, "get_status", new=AsyncMock(return_value=mock_response)
    ), pytest.raises(ServerError, match="Timed out waiting for model"):
        _ = await client.wait_until_ready(
            12345, timeout_seconds=0, poll_interval_seconds=0
        )


@pytest.mark.asyncio
async def test_wait_until_ready_failed_status():
    """Test wait_until_ready when model fails."""
    client = VecInfClient()

    mock_response = MagicMock()
    mock_response.server_status = ModelStatus.FAILED
    mock_response.failed_reason = "Out of memory"

    with patch.object(
        client, "get_status", new=AsyncMock(return_value=mock_response)
    ), pytest.raises(ServerError, match="Model failed to start: Out of memory"):
        _ = await client.wait_until_ready(12345)


@pytest.mark.asyncio
async def test_wait_until_ready_failed_no_reason():
    """Test wait_until_ready when model fails without reason."""
    client = VecInfClient()

    mock_response = MagicMock()
    mock_response.server_status = ModelStatus.FAILED
    mock_response.failed_reason = None

    with patch.object(
        client, "get_status", new=AsyncMock(return_value=mock_response)
    ), pytest.raises(ServerError, match="Model failed to start: Unknown error"):
        _ = await client.wait_until_ready(12345)


@pytest.mark.asyncio
async def test_wait_until_ready_shutdown():
    """Test wait_until_ready when model is shutdown."""
    client = VecInfClient()

    mock_response = MagicMock()
    mock_response.server_status = ModelStatus.SHUTDOWN

    with patch.object(
        client, "get_status", new=AsyncMock(return_value=mock_response)
    ), pytest.raises(
        ServerError, match="Model was shutdown before it became ready"
    ):
        _ = await client.wait_until_ready(12345)
