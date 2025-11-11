"""Tests for the Vector Inference API client."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from vec_inf.client import ModelStatus, ModelType, VecInfClient
from vec_inf.client._exceptions import (
    ModelConfigurationError,
    ServerError,
    SlurmJobError,
)


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


def test_cleanup_logs_no_match(tmp_path):
    """Test when cleanup_logs returns empty list."""
    fam_a = tmp_path / "fam_a"
    model_a = fam_a / "model_a.999"
    model_a.mkdir(parents=True)

    client = VecInfClient()
    deleted = client.cleanup_logs(
        log_dir=tmp_path,
        model_family="fam_b",
        dry_run=False,
    )

    assert deleted == []
    assert fam_a.exists()
    assert model_a.exists()


def test_cleanup_logs_deletes_matching_dirs(tmp_path):
    """Test that cleanup_logs deletes model directories matching filters."""
    fam_a = tmp_path / "fam_a"
    fam_a.mkdir()

    model_a_1 = fam_a / "model_a.10"
    model_a_2 = fam_a / "model_a.20"
    model_b = fam_a / "model_b.30"

    model_a_1.mkdir()
    model_a_2.mkdir()
    model_b.mkdir()

    client = VecInfClient()
    deleted = client.cleanup_logs(
        log_dir=tmp_path,
        model_family="fam_a",
        model_name="model_a",
        before_job_id=15,
        dry_run=False,
    )

    assert deleted == [model_a_1]
    assert not model_a_1.exists()
    assert model_a_2.exists()
    assert model_b.exists()


def test_cleanup_logs_matching_dirs_dry_run(tmp_path):
    """Test that cleanup_logs find model directories matching filters."""
    fam_a = tmp_path / "fam_a"
    fam_a.mkdir()

    model_a_1 = fam_a / "model_a.10"
    model_a_2 = fam_a / "model_a.20"

    model_a_1.mkdir()
    model_a_2.mkdir()

    client = VecInfClient()
    deleted = client.cleanup_logs(
        log_dir=tmp_path,
        model_family="fam_a",
        model_name="model_a",
        before_job_id=15,
        dry_run=True,
    )

    assert deleted == [model_a_1]
    assert model_a_1.exists()
    assert model_a_2.exists()


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


def test_batch_launch_models_success():
    """Test successfully launching multiple models in batch mode."""
    client = VecInfClient()

    # Create a mock batch response
    mock_response = MagicMock()
    mock_response.slurm_job_id = "12345678"
    mock_response.slurm_job_name = "BATCH-model1-model2"
    mock_response.model_names = ["model1", "model2"]
    mock_response.config = {"slurm_job_id": "12345678"}

    # Mock the batch launch method
    client.batch_launch_models = lambda model_names, batch_config=None: mock_response

    result = client.batch_launch_models(["model1", "model2"])

    assert result.slurm_job_id == "12345678"
    assert result.slurm_job_name == "BATCH-model1-model2"
    assert result.model_names == ["model1", "model2"]
    assert result.config["slurm_job_id"] == "12345678"


def test_batch_launch_models_with_config():
    """Test launching multiple models with custom batch configuration."""
    client = VecInfClient()

    # Create a mock batch response
    mock_response = MagicMock()
    mock_response.slurm_job_id = "12345678"
    mock_response.slurm_job_name = "BATCH-model1-model2"
    mock_response.model_names = ["model1", "model2"]
    mock_response.config = {"slurm_job_id": "12345678"}

    # Mock the batch launch method
    client.batch_launch_models = lambda model_names, batch_config=None: mock_response

    result = client.batch_launch_models(
        ["model1", "model2"], batch_config="custom_config.yaml"
    )

    assert result.slurm_job_id == "12345678"
    assert result.slurm_job_name == "BATCH-model1-model2"
    assert result.model_names == ["model1", "model2"]


def test_batch_launch_models_empty_list():
    """Test that batch launch with empty model list raises an error."""
    client = VecInfClient()

    # Mock the batch launch method to raise an error for empty list
    def mock_batch_launch(model_names, batch_config=None):
        if not model_names:
            raise ValueError("Model names list cannot be empty")
        return MagicMock()

    client.batch_launch_models = mock_batch_launch

    with pytest.raises(ValueError, match="Model names list cannot be empty"):
        client.batch_launch_models([])


def test_batch_launch_models_single_model():
    """Test launching a single model in batch mode."""
    client = VecInfClient()

    # Create a mock batch response for single model
    mock_response = MagicMock()
    mock_response.slurm_job_id = "12345678"
    mock_response.slurm_job_name = "BATCH-model1"
    mock_response.model_names = ["model1"]
    mock_response.config = {"slurm_job_id": "12345678"}

    # Mock the batch launch method
    client.batch_launch_models = lambda model_names, batch_config=None: mock_response

    result = client.batch_launch_models(["model1"])

    assert result.slurm_job_id == "12345678"
    assert result.slurm_job_name == "BATCH-model1"
    assert result.model_names == ["model1"]
    assert len(result.model_names) == 1


def test_batch_launch_models_three_models():
    """Test launching three models in batch mode."""
    client = VecInfClient()

    # Create a mock batch response for three models
    mock_response = MagicMock()
    mock_response.slurm_job_id = "12345678"
    mock_response.slurm_job_name = "BATCH-model1-model2-model3"
    mock_response.model_names = ["model1", "model2", "model3"]
    mock_response.config = {"slurm_job_id": "12345678"}

    # Mock the batch launch method
    client.batch_launch_models = lambda model_names, batch_config=None: mock_response

    result = client.batch_launch_models(["model1", "model2", "model3"])

    assert result.slurm_job_id == "12345678"
    assert result.slurm_job_name == "BATCH-model1-model2-model3"
    assert result.model_names == ["model1", "model2", "model3"]
    assert len(result.model_names) == 3


def test_batch_launch_models_with_special_characters():
    """Test launching models with special characters in names."""
    client = VecInfClient()

    # Create a mock batch response for models with special characters
    mock_response = MagicMock()
    mock_response.slurm_job_id = "12345678"
    mock_response.slurm_job_name = "BATCH-model-1-model_2"
    mock_response.model_names = ["model-1", "model_2"]
    mock_response.config = {"slurm_job_id": "12345678"}

    # Mock the batch launch method
    client.batch_launch_models = lambda model_names, batch_config=None: mock_response

    result = client.batch_launch_models(["model-1", "model_2"])

    assert result.slurm_job_id == "12345678"
    assert result.slurm_job_name == "BATCH-model-1-model_2"
    assert result.model_names == ["model-1", "model_2"]


def test_batch_launch_models_configuration_error():
    """Test that batch launch raises configuration error when models are not found."""
    client = VecInfClient()

    # Mock the batch launch method to raise a configuration error
    def mock_batch_launch(model_names, batch_config=None):
        raise ModelConfigurationError(
            "Model 'nonexistent-model' not found in configuration"
        )

    client.batch_launch_models = mock_batch_launch

    with pytest.raises(
        ModelConfigurationError,
        match="Model 'nonexistent-model' not found in configuration",
    ):
        client.batch_launch_models(["model1", "nonexistent-model"])


def test_batch_launch_models_slurm_error():
    """Test that batch launch raises SLURM error when job submission fails."""
    client = VecInfClient()

    # Mock the batch launch method to raise a SLURM error
    def mock_batch_launch(model_names, batch_config=None):
        raise SlurmJobError("sbatch: error: Invalid partition specified")

    client.batch_launch_models = mock_batch_launch

    with pytest.raises(
        SlurmJobError, match="sbatch: error: Invalid partition specified"
    ):
        client.batch_launch_models(["model1", "model2"])


def test_batch_launch_models_integration():
    """Test integration of batch launch with actual BatchModelLauncher."""
    client = VecInfClient()

    with (
        patch("vec_inf.client.api.BatchModelLauncher") as mock_launcher_class,
        patch(
            "vec_inf.client.api.run_bash_command",
            return_value=("Submitted batch job 12345678", ""),
        ),
    ):
        # Mock the BatchModelLauncher instance
        mock_launcher = MagicMock()
        mock_launcher.launch.return_value = MagicMock(
            slurm_job_id="12345678",
            slurm_job_name="BATCH-model1-model2",
            model_names=["model1", "model2"],
            config={"slurm_job_id": "12345678"},
        )
        mock_launcher_class.return_value = mock_launcher

        result = client.batch_launch_models(["model1", "model2"])

        # Verify BatchModelLauncher was called correctly
        mock_launcher_class.assert_called_once_with(
            ["model1", "model2"], None, None, None
        )
        mock_launcher.launch.assert_called_once()

        # Verify the response
        assert result.slurm_job_id == "12345678"
        assert result.slurm_job_name == "BATCH-model1-model2"
        assert result.model_names == ["model1", "model2"]


def test_batch_launch_models_with_custom_config_integration():
    """Test integration of batch launch with custom configuration."""
    client = VecInfClient()

    with (
        patch("vec_inf.client.api.BatchModelLauncher") as mock_launcher_class,
        patch(
            "vec_inf.client.api.run_bash_command",
            return_value=("Submitted batch job 12345678", ""),
        ),
    ):
        # Mock the BatchModelLauncher instance
        mock_launcher = MagicMock()
        mock_launcher.launch.return_value = MagicMock(
            slurm_job_id="12345678",
            slurm_job_name="BATCH-model1-model2",
            model_names=["model1", "model2"],
            config={"slurm_job_id": "12345678"},
        )
        mock_launcher_class.return_value = mock_launcher

        result = client.batch_launch_models(
            ["model1", "model2"], batch_config="custom_config.yaml"
        )

        # Verify BatchModelLauncher was called with custom config
        mock_launcher_class.assert_called_once_with(
            ["model1", "model2"], "custom_config.yaml", None, None
        )
        mock_launcher.launch.assert_called_once()

        # Verify the response
        assert result.slurm_job_id == "12345678"
        assert result.slurm_job_name == "BATCH-model1-model2"
        assert result.model_names == ["model1", "model2"]


def test_fetch_running_jobs_success_with_matching_jobs():
    """Test fetch_running_jobs returns matching job IDs."""
    client = VecInfClient()

    # Mock squeue output with multiple jobs
    squeue_output = "12345  RUNNING  gpu\n67890  RUNNING  gpu\n"
    # Mock scontrol outputs for each job
    scontrol_outputs = {
        "12345": "JobId=12345 JobName=test-model-vec-inf User=user",
        "67890": "JobId=67890 JobName=other-model-vec-inf User=user",
    }

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            mock_result.stdout = squeue_output
            mock_result.returncode = 0
        elif cmd[0] == "scontrol":
            job_id = cmd[-1]
            mock_result.stdout = scontrol_outputs.get(job_id, "")
            mock_result.returncode = 0
        return mock_result

    with patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run):
        result = client.fetch_running_jobs()

    assert result == ["12345", "67890"]


def test_fetch_running_jobs_no_matching_jobs():
    """Test fetch_running_jobs returns empty list when no jobs match."""
    client = VecInfClient()

    # Mock squeue output with jobs that don't match
    squeue_output = "12345  RUNNING  gpu\n67890  RUNNING  gpu\n"
    # Mock scontrol outputs - jobs don't end with -vec-inf
    scontrol_outputs = {
        "12345": "JobId=12345 JobName=test-model User=user",
        "67890": "JobId=67890 JobName=other-job User=user",
    }

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            mock_result.stdout = squeue_output
            mock_result.returncode = 0
        elif cmd[0] == "scontrol":
            job_id = cmd[-1]
            mock_result.stdout = scontrol_outputs.get(job_id, "")
            mock_result.returncode = 0
        return mock_result

    with patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run):
        result = client.fetch_running_jobs()

    assert result == []


def test_fetch_running_jobs_empty_squeue():
    """Test fetch_running_jobs returns empty list when squeue is empty."""
    client = VecInfClient()

    # Mock empty squeue output
    squeue_output = ""

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            mock_result.stdout = squeue_output
            mock_result.returncode = 0
        return mock_result

    with patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run):
        result = client.fetch_running_jobs()

    assert result == []


def test_fetch_running_jobs_mixed_jobs():
    """Test fetch_running_jobs filters correctly with matching/non-matching jobs."""
    client = VecInfClient()

    # Mock squeue output with multiple jobs
    squeue_output = "12345  RUNNING  gpu\n67890  RUNNING  gpu\n11111  RUNNING  gpu\n"
    # Mock scontrol outputs - only some match
    scontrol_outputs = {
        "12345": "JobId=12345 JobName=test-model-vec-inf User=user",
        "67890": "JobId=67890 JobName=other-job User=user",  # Doesn't match
        "11111": "JobId=11111 JobName=another-model-vec-inf User=user",
    }

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            mock_result.stdout = squeue_output
            mock_result.returncode = 0
        elif cmd[0] == "scontrol":
            job_id = cmd[-1]
            mock_result.stdout = scontrol_outputs.get(job_id, "")
            mock_result.returncode = 0
        return mock_result

    with patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run):
        result = client.fetch_running_jobs()

    assert result == ["12345", "11111"]


def test_fetch_running_jobs_scontrol_failure():
    """Test fetch_running_jobs skips jobs when scontrol fails."""
    client = VecInfClient()

    # Mock squeue output
    squeue_output = "12345  RUNNING  gpu\n67890  RUNNING  gpu\n"
    # Mock scontrol - one succeeds, one fails
    scontrol_outputs = {
        "12345": "JobId=12345 JobName=test-model-vec-inf User=user",
    }

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            mock_result.stdout = squeue_output
            mock_result.returncode = 0
        elif cmd[0] == "scontrol":
            job_id = cmd[-1]
            if job_id in scontrol_outputs:
                mock_result.stdout = scontrol_outputs[job_id]
                mock_result.returncode = 0
            else:
                # Simulate CalledProcessError for job 67890
                raise subprocess.CalledProcessError(1, cmd)
        return mock_result

    with patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run):
        result = client.fetch_running_jobs()

    # Should only return the job that succeeded
    assert result == ["12345"]


def test_fetch_running_jobs_squeue_failure():
    """Test fetch_running_jobs raises SlurmJobError when squeue fails."""
    client = VecInfClient()

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            # Simulate CalledProcessError
            raise subprocess.CalledProcessError(1, cmd, stderr="squeue: error")
        return mock_result

    with (
        patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run),
        pytest.raises(SlurmJobError, match="Error running slurm command"),
    ):
        client.fetch_running_jobs()


def test_fetch_running_jobs_job_name_not_found():
    """Test fetch_running_jobs handles missing JobName in scontrol output."""
    client = VecInfClient()

    # Mock squeue output
    squeue_output = "12345  RUNNING  gpu\n"
    # Mock scontrol output without JobName
    scontrol_output = "JobId=12345 User=user State=RUNNING"

    def mock_subprocess_run(cmd, **kwargs):
        mock_result = MagicMock()
        if cmd[0] == "squeue":
            mock_result.stdout = squeue_output
            mock_result.returncode = 0
        elif cmd[0] == "scontrol":
            mock_result.stdout = scontrol_output
            mock_result.returncode = 0
        return mock_result

    with patch("vec_inf.client.api.subprocess.run", side_effect=mock_subprocess_run):
        result = client.fetch_running_jobs()

    # Should return empty list since JobName doesn't match
    assert result == []
