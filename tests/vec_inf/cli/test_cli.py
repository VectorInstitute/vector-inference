"""Tests for the CLI module of the vec_inf package."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vec_inf.cli._cli import cli


@pytest.fixture
def runner():
    """Fixture for invoking CLI commands."""
    return CliRunner()


@pytest.fixture
def mock_launch_output():
    """Fixture providing consistent mock output structure."""

    def _output(job_id):
        return f"""
Job Name: Meta-Llama-3.1-8B
Partition: a40
Num Nodes: 1
GPUs per Node: 1
QOS: llm
Walltime: 08:00:00
Model Type: LLM
Task: generate
Data Type: auto
Max Model Length: 131072
Max Num Seqs: 256
Vocabulary Size: 128256
Pipeline Parallelism: True
Enforce Eager: False
Log Directory: /h/llm/.vec-inf-logs/Meta-Llama-3.1
Model Weights Parent Directory: /model-weights
Submitted batch job {job_id}
        """.strip()

    return _output


@pytest.fixture
def mock_status_output():
    """Fixture providing consistent mock status output."""

    def _output(job_id, job_state):
        return f"""
JobId={job_id} JobName=Meta-Llama-3.1-8B JobState={job_state} QOS=llm NumNodes=1-1 NumCPUs=16 NumTasks=1
        """.strip()

    return _output


def test_launch_command_success(runner, mock_launch_output):
    """Test successful model launch with minimal required arguments."""
    with patch("vec_inf.cli._utils.run_bash_command") as mock_run:
        expected_job_id = "14933053"
        mock_run.return_value = mock_launch_output(expected_job_id)
        result = runner.invoke(cli, ["launch", "Meta-Llama-3.1-8B"])
        assert result.exit_code == 0
        assert expected_job_id in result.output
        mock_run.assert_called_once()


def test_launch_command_with_json_output(runner, mock_launch_output):
    """Test JSON output format for launch command."""
    with patch("vec_inf.cli._utils.run_bash_command") as mock_run:
        expected_job_id = "14933051"
        mock_run.return_value = mock_launch_output(expected_job_id)
        result = runner.invoke(cli, ["launch", "Meta-Llama-3.1-8B", "--json-mode"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output.get("slurm_job_id") == expected_job_id
        assert output.get("job_name") == "Meta-Llama-3.1-8B"
        assert output.get("model_type") == "LLM"
        assert output.get("log_directory") == "/h/llm/.vec-inf-logs/Meta-Llama-3.1"


def test_launch_command_invalid_model(runner):
    """Test error handling for unknown model."""
    result = runner.invoke(cli, ["launch", "unknown-model"])
    assert result.exit_code == 1
    assert "Model 'unknown-model' not found in configuration" in result.output


def test_list_all_models(runner):
    """Test listing all available models."""
    result = runner.invoke(cli, ["list"])

    assert result.exit_code == 0
    assert "Meta-Llama-3.1-8B" in result.output


def test_list_single_model(runner):
    """Test displaying details for a specific model."""
    result = runner.invoke(cli, ["list", "Meta-Llama-3.1-8B"])

    assert result.exit_code == 0
    assert "Meta-Llama-3.1-8B" in result.output
    assert "LLM" in result.output
    assert "/model-weights" in result.output
