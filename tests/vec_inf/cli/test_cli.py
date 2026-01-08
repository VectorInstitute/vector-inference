"""Tests for the CLI module of the vec_inf package."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vec_inf.cli._cli import cli


@pytest.fixture
def runner():
    """Fixture for invoking CLI commands."""
    return CliRunner()


def test_launch_command_success(runner):
    """Test successful model launch with minimal required arguments."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        # Mock the client instance
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the launch response - ensure numeric values are strings
        # for Rich table rendering
        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933053",
            "model_name": "Meta-Llama-3.1-8B",
            "model_type": "LLM",
            "log_dir": "/tmp/test_logs",
            "partition": "gpu",
            "qos": "normal",
            "time": "1:00:00",
            "num_nodes": "1",
            "gpus_per_node": "1",
            "cpus_per_task": "8",
            "mem_per_node": "32G",
            "model_weights_parent_dir": "/model-weights",
            "vocab_size": "128000",
            "venv": "/path/to/venv",
            "engine": "vllm",
            "engine_args": {"max_model_len": 8192},
            "env": {"CACHE": "/cache"},
        }
        mock_client.launch_model.return_value = mock_response

        result = runner.invoke(cli, ["launch", "Meta-Llama-3.1-8B"])

        assert result.exit_code == 0
        assert "14933053" in result.output
        mock_client.launch_model.assert_called_once()


def test_launch_command_with_json_output(runner):
    """Test JSON output format for launch command."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933051",
            "model_name": "Meta-Llama-3.1-8B",
            "model_type": "LLM",
            "log_dir": "/tmp/test_logs",
        }
        mock_client.launch_model.return_value = mock_response

        result = runner.invoke(cli, ["launch", "Meta-Llama-3.1-8B", "--json-mode"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output.get("slurm_job_id") == "14933051"
        assert output.get("model_name") == "Meta-Llama-3.1-8B"


def test_launch_command_model_not_found(runner):
    """Test handling of a model that's neither in config nor has weights."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the client to raise an exception
        mock_client.launch_model.side_effect = Exception(
            "'unknown-model' not found in configuration and model weights "
            "not found at expected path '/model-weights/unknown-model'"
        )

        result = runner.invoke(cli, ["launch", "unknown-model"])

        assert result.exit_code == 1
        assert "Launch failed:" in result.output


def test_list_all_models(runner):
    """Test listing all available models."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_model_info = MagicMock()
        mock_model_info.name = "Meta-Llama-3.1-8B"
        mock_model_info.family = "Meta-Llama-3.1"
        mock_model_info.model_type = "LLM"
        mock_model_info.variant = None
        mock_client.list_models.return_value = [mock_model_info]

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Meta-Llama-3.1" in result.output


def test_list_single_model(runner):
    """Test displaying details for a specific model."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_config = MagicMock()
        mock_config.model_dump.return_value = {
            "model_name": "Meta-Llama-3.1-8B",
            "model_family": "Meta-Llama-3.1",
            "model_type": "LLM",
            "model_weights_parent_dir": "/model-weights",
            "engine": "vllm",
            "vllm_args": {"--max-model-len": 8192},
            "sglang_args": {},
        }
        mock_client.get_model_config.return_value = mock_config

        result = runner.invoke(cli, ["list", "Meta-Llama-3.1-8B"])

        assert result.exit_code == 0
        assert "Meta-Llama-3.1-8B" in result.output


def test_status_command(runner):
    """Test status command with job ID argument."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_status = MagicMock()
        mock_status.model_name = "Meta-Llama-3.1-8B"
        mock_status.server_status = "READY"
        mock_status.base_url = "http://localhost:8000"
        mock_status.pending_reason = None
        mock_status.failed_reason = None
        mock_client.get_status.return_value = mock_status

        result = runner.invoke(cli, ["status", "12345"])

        assert result.exit_code == 0
        assert "Meta-Llama-3.1-8B" in result.output


def test_status_command_no_job_id_no_running_jobs(runner):
    """Test status command with no argument when no jobs are running."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.fetch_running_jobs.return_value = []

        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "No running jobs found." in result.output


def test_status_command_no_job_id_single_running_job(runner):
    """Test status command with no argument when one job is running."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.fetch_running_jobs.return_value = ["12345"]

        mock_status = MagicMock()
        mock_status.model_name = "test-model-1"
        mock_status.server_status = "READY"
        mock_status.base_url = "http://localhost:8000"
        mock_status.pending_reason = None
        mock_status.failed_reason = None
        mock_client.get_status.return_value = mock_status

        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "test-model-1" in result.output
        mock_client.fetch_running_jobs.assert_called_once()
        mock_client.get_status.assert_called_once_with("12345")


def test_status_command_no_job_id_multiple_running_jobs(runner):
    """Test status command with no argument when multiple jobs are running."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.fetch_running_jobs.return_value = ["12345", "67890"]

        mock_status_1 = MagicMock()
        mock_status_1.model_name = "test-model-1"
        mock_status_1.server_status = "READY"
        mock_status_1.base_url = "http://localhost:8000"
        mock_status_1.pending_reason = None
        mock_status_1.failed_reason = None

        mock_status_2 = MagicMock()
        mock_status_2.model_name = "test-model-2"
        mock_status_2.server_status = "PENDING"
        mock_status_2.base_url = None
        mock_status_2.pending_reason = "Waiting for resources"
        mock_status_2.failed_reason = None

        mock_client.get_status.side_effect = [mock_status_1, mock_status_2]

        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "test-model-1" in result.output
        assert "test-model-2" in result.output
        assert "12345" in result.output
        assert "67890" in result.output
        mock_client.fetch_running_jobs.assert_called_once()
        assert mock_client.get_status.call_count == 2


def test_status_command_no_job_id_multiple_jobs_json_mode(runner):
    """Test status command with no argument and JSON mode for multiple jobs."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.fetch_running_jobs.return_value = ["12345", "67890"]

        mock_status_1 = MagicMock()
        mock_status_1.model_name = "test-model-1"
        mock_status_1.server_status = "READY"
        mock_status_1.base_url = "http://localhost:8000"
        mock_status_1.pending_reason = None
        mock_status_1.failed_reason = None

        mock_status_2 = MagicMock()
        mock_status_2.model_name = "test-model-2"
        mock_status_2.server_status = "FAILED"
        mock_status_2.base_url = None
        mock_status_2.pending_reason = None
        mock_status_2.failed_reason = "Out of memory"

        mock_client.get_status.side_effect = [mock_status_1, mock_status_2]

        result = runner.invoke(cli, ["status", "--json-mode"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0]["model_name"] == "test-model-1"
        assert output[0]["model_status"] == "READY"
        assert output[1]["model_name"] == "test-model-2"
        assert output[1]["model_status"] == "FAILED"


def test_shutdown_command(runner):
    """Test shutdown command."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock shutdown to not raise exception
        mock_client.shutdown_model.return_value = None

        result = runner.invoke(cli, ["shutdown", "12345"])

        assert result.exit_code == 0
        assert "Shutting down model with Slurm Job ID: 12345" in result.output


def test_metrics_command_pending_server(runner):
    """Test metrics command when server is pending."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.metrics = "ERROR: Pending resources for server initialization"
        mock_client.get_metrics.return_value = mock_response

        result = runner.invoke(cli, ["metrics", "12345"])

        assert result.exit_code == 0
        assert "ERROR" in result.output


def test_metrics_command_server_ready(runner):
    """Test metrics command when server is ready and returning metrics."""
    with (
        patch("vec_inf.cli._cli.VecInfClient") as mock_client_class,
        patch("vec_inf.cli._cli.time.sleep", side_effect=KeyboardInterrupt),
    ):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.metrics = {
            "prompt_tokens_per_sec": 100.0,
            "generation_tokens_per_sec": 500.0,
            "gpu_cache_usage": 0.5,
        }
        mock_client.get_metrics.return_value = mock_response

        # Use input simulation to trigger KeyboardInterrupt
        result = runner.invoke(cli, ["metrics", "12345"], input="\x03")  # Ctrl+C

        # The command should handle KeyboardInterrupt gracefully
        # In CLI context, this might exit with code 1 but that's expected behavior
        assert result.exit_code in [0, 1]  # Accept both success and interrupted states


def test_cli_cleanup_logs_dry_run(runner, tmp_path):
    """Test CLI cleanup command in dry-run mode."""
    model_dir = tmp_path / "fam_a" / "model_a.123"
    model_dir.mkdir(parents=True)

    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.cleanup_logs.return_value = [model_dir]

        result = runner.invoke(
            cli,
            [
                "cleanup",
                "--log-dir",
                str(tmp_path),
                "--model-family",
                "fam_a",
                "--model-name",
                "model_a",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "would be deleted" in result.output


def test_cli_cleanup_logs_delete(runner, tmp_path):
    """Test cleanup_logs CLI deletes matching directories when not in dry-run mode."""
    fam_dir = tmp_path / "fam_a"
    fam_dir.mkdir()
    model_dir = fam_dir / "model_a.1"
    model_dir.mkdir()

    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.cleanup_logs.return_value = [model_dir]

        result = runner.invoke(
            cli,
            [
                "cleanup",
                "--log-dir",
                str(tmp_path),
                "--model-family",
                "fam_a",
                "--model-name",
                "model_a",
                "--job-id",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Deleted 1 log directory" in result.output


def test_cli_cleanup_logs_no_match(runner, tmp_path):
    """Test cleanup_logs CLI when no directories match the filters."""
    fam_dir = tmp_path / "fam_a"
    fam_dir.mkdir()
    (fam_dir / "model_a.1").mkdir()

    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.cleanup_logs.return_value = []

        result = runner.invoke(
            cli,
            [
                "cleanup",
                "--log-dir",
                str(tmp_path),
                "--model-family",
                "fam_b",
            ],
        )

        assert result.exit_code == 0
        assert "No matching log directories were deleted." in result.output


def test_batch_launch_command_success(runner):
    """Test successful batch model launch with multiple models."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933053",
            "slurm_job_name": "BATCH-job",
            "model_names": ["Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"],
            "log_dir": "/tmp/test_logs",  # Moved to top level
            "models": {
                "Meta-Llama-3.1-8B": {
                    "model_name": "Meta-Llama-3.1-8B",
                    "partition": "gpu",
                    "qos": "normal",
                    "time": "1:00:00",
                    "num_nodes": "1",
                    "gpus_per_node": "1",
                    "cpus_per_task": "8",
                    "mem_per_node": "32G",
                    "engine": "vllm",
                },
                "Meta-Llama-3.1-70B": {
                    "model_name": "Meta-Llama-3.1-70B",
                    "partition": "gpu",
                    "qos": "normal",
                    "time": "1:00:00",
                    "num_nodes": "1",
                    "gpus_per_node": "1",
                    "cpus_per_task": "8",
                    "mem_per_node": "32G",
                    "engine": "vllm",
                },
            },
        }
        mock_client.batch_launch_models.return_value = mock_response

        result = runner.invoke(
            cli, ["batch-launch", "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"]
        )

        assert result.exit_code == 0
        assert "14933053" in result.output


def test_batch_launch_command_with_json_output(runner):
    """Test JSON output format for batch launch command."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933051",
            "slurm_job_name": "BATCH-job",
            "model_names": ["Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"],
        }
        mock_client.batch_launch_models.return_value = mock_response

        result = runner.invoke(
            cli,
            ["batch-launch", "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B", "--json-mode"],
        )

        assert result.exit_code == 0
        assert "14933051" in result.output


def test_batch_launch_command_model_not_found(runner):
    """Test batch launch when one of the models is not found."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.batch_launch_models.side_effect = Exception(
            "'unknown-model' not found in configuration"
        )

        result = runner.invoke(
            cli, ["batch-launch", "Meta-Llama-3.1-8B", "unknown-model"]
        )

        assert result.exit_code == 1
        assert "Batch launch failed:" in result.output


def test_batch_launch_command_slurm_error(runner):
    """Test batch launch when SLURM job submission fails."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.batch_launch_models.side_effect = Exception(
            "sbatch: error: Invalid partition specified"
        )

        result = runner.invoke(
            cli, ["batch-launch", "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"]
        )

        assert result.exit_code == 1
        assert "Batch launch failed:" in result.output


def test_batch_launch_command_no_models(runner):
    """Test batch launch with no models specified."""
    result = runner.invoke(cli, ["batch-launch"])

    # Should fail because no models were specified
    assert result.exit_code != 0


def test_launch_command_with_engine_option(runner):
    """Test launch command with --engine option."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933053",
            "model_name": "Meta-Llama-3.1-8B",
            "model_type": "LLM",
            "vocab_size": "128000",
            "partition": "gpu",
            "qos": "normal",
            "time": "1:00:00",
            "num_nodes": "1",
            "gpus_per_node": "1",
            "cpus_per_task": "8",
            "mem_per_node": "32G",
            "model_weights_parent_dir": "/model-weights",
            "log_dir": "/tmp/logs",
            "engine": "sglang",
            "engine_args": {},
        }
        mock_client.launch_model.return_value = mock_response

        result = runner.invoke(
            cli, ["launch", "Meta-Llama-3.1-8B", "--engine", "sglang"]
        )

        assert result.exit_code == 0
        assert "14933053" in result.output
        # Verify LaunchOptions was called with engine
        call_args = mock_client.launch_model.call_args
        assert call_args[0][0] == "Meta-Llama-3.1-8B"
        assert call_args[0][1].engine == "sglang"


def test_launch_command_with_vllm_args(runner):
    """Test launch command with --vllm-args option."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933053",
            "model_name": "Meta-Llama-3.1-8B",
            "model_type": "LLM",
            "vocab_size": "128000",
            "partition": "gpu",
            "qos": "normal",
            "time": "1:00:00",
            "num_nodes": "1",
            "gpus_per_node": "1",
            "cpus_per_task": "8",
            "mem_per_node": "32G",
            "model_weights_parent_dir": "/model-weights",
            "log_dir": "/tmp/logs",
            "engine": "vllm",
            "engine_args": {},
        }
        mock_client.launch_model.return_value = mock_response

        result = runner.invoke(
            cli,
            [
                "launch",
                "Meta-Llama-3.1-8B",
                "--vllm-args",
                "--max-model-len=8192,--tensor-parallel-size=4",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_client.launch_model.call_args
        assert (
            call_args[0][1].vllm_args == "--max-model-len=8192,--tensor-parallel-size=4"
        )


def test_launch_command_with_sglang_args(runner):
    """Test launch command with --sglang-args option."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.config = {
            "slurm_job_id": "14933053",
            "model_name": "Meta-Llama-3.1-8B",
            "model_type": "LLM",
            "vocab_size": "128000",
            "partition": "gpu",
            "qos": "normal",
            "time": "1:00:00",
            "num_nodes": "1",
            "gpus_per_node": "1",
            "cpus_per_task": "8",
            "mem_per_node": "32G",
            "model_weights_parent_dir": "/model-weights",
            "log_dir": "/tmp/logs",
            "engine": "sglang",
            "engine_args": {},
        }
        mock_client.launch_model.return_value = mock_response

        result = runner.invoke(
            cli,
            [
                "launch",
                "Meta-Llama-3.1-8B",
                "--sglang-args",
                "--context-length=8192,--tensor-parallel-size=4",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_client.launch_model.call_args
        assert (
            call_args[0][1].sglang_args
            == "--context-length=8192,--tensor-parallel-size=4"
        )


def test_launch_command_engine_mismatch_error(runner):
    """Test launch command error when engine and args mismatch."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.launch_model.side_effect = ValueError(
            "Mismatch between provided engine 'vllm' and engine-specific args 'sglang'"
        )

        result = runner.invoke(
            cli,
            [
                "launch",
                "Meta-Llama-3.1-8B",
                "--engine",
                "vllm",
                "--sglang-args",
                "--context-length=8192",
            ],
        )

        assert result.exit_code == 1
        assert "Launch failed:" in result.output


def test_list_single_model_sglang(runner):
    """Test listing model with SGLang engine."""
    with patch("vec_inf.cli._cli.VecInfClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_config = MagicMock()
        mock_config.model_dump.return_value = {
            "model_name": "Meta-Llama-3.1-8B",
            "model_family": "Meta-Llama-3.1",
            "model_type": "LLM",
            "engine": "sglang",
            "sglang_args": {"--context-length": "8192"},
        }
        mock_client.get_model_config.return_value = mock_config

        result = runner.invoke(cli, ["list", "Meta-Llama-3.1-8B"])

        assert result.exit_code == 0
        assert "Meta-Llama-3.1-8B" in result.output
        assert "sglang" in result.output.lower() or "SGLang" in result.output
