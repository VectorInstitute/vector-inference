"""Tests for the utility functions in the CLI module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from vec_inf.cli._utils import (
    MODEL_READY_SIGNATURE,
    create_table,
    get_base_url,
    is_server_running,
    load_config,
    model_health_check,
    read_slurm_log,
    run_bash_command,
)


@pytest.fixture
def mock_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


def test_run_bash_command_success():
    """Test that run_bash_command returns the output of the command."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("test output", "")
        mock_popen.return_value = mock_process
        result, stderr = run_bash_command("echo test")
        assert result == "test output"
        assert stderr == ""


def test_run_bash_command_error():
    """Test run_bash_command with error output."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "error output")
        mock_popen.return_value = mock_process
        result, stderr = run_bash_command("invalid_command")
        assert result == ""
        assert stderr == "error output"


def test_read_slurm_log_found(mock_log_dir):
    """Test that read_slurm_log reads the content of a log file."""
    test_content = ["line1\n", "line2\n"]
    log_file = mock_log_dir / "test_job.123" / "test_job.123.err"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("".join(test_content))
    result = read_slurm_log("test_job", 123, "err", mock_log_dir)
    assert result == test_content


def test_read_slurm_log_not_found():
    """Test read_slurm_log, return an error message if the log file is not found."""
    result = read_slurm_log("missing_job", 456, "err", "/nonexistent")
    assert (
        result == "LOG FILE NOT FOUND: /nonexistent/missing_job.456/missing_job.456.err"
    )


@pytest.mark.parametrize(
    "log_content,expected",
    [
        ([MODEL_READY_SIGNATURE], "RUNNING"),
        (["ERROR: something wrong"], ("FAILED", "ERROR: something wrong")),
        ([], "LAUNCHING"),
        (["some other content"], "LAUNCHING"),
    ],
)
def test_is_server_running_statuses(log_content, expected):
    """Test that is_server_running returns the correct status."""
    with patch("vec_inf.cli._utils.read_slurm_log") as mock_read:
        mock_read.return_value = log_content
        result = is_server_running("test_job", 123, None)
        assert result == expected


def test_get_base_url_found():
    """Test that get_base_url returns the correct base URL."""
    test_dict = {"server_address": "http://localhost:8000"}
    with patch("vec_inf.cli._utils.read_slurm_log") as mock_read:
        mock_read.return_value = test_dict
        result = get_base_url("test_job", 123, None)
        assert result == "http://localhost:8000"


def test_get_base_url_not_found():
    """Test get_base_url when URL is not found in logs."""
    with patch("vec_inf.cli._utils.read_slurm_log") as mock_read:
        mock_read.return_value = {"random_key": "123"}
        result = get_base_url("test_job", 123, None)
        assert result == "URL NOT FOUND"


@pytest.mark.parametrize(
    "url,status_code,expected",
    [
        ("http://localhost:8000", 200, ("READY", 200)),
        ("http://localhost:8000", 500, ("FAILED", 500)),
        ("not_a_url", None, ("FAILED", "not_a_url")),
    ],
)
def test_model_health_check(url, status_code, expected):
    """Test model_health_check with various scenarios."""
    with patch("vec_inf.cli._utils.get_base_url") as mock_url:
        mock_url.return_value = url
        if url.startswith("http"):
            with patch("requests.get") as mock_get:
                mock_get.return_value.status_code = status_code
                result = model_health_check("test_job", 123, None)
                assert result == expected
        else:
            result = model_health_check("test_job", 123, None)
            assert result == expected


def test_model_health_check_request_exception():
    """Test model_health_check when request raises an exception."""
    with (
        patch("vec_inf.cli._utils.get_base_url") as mock_url,
        patch("requests.get") as mock_get,
    ):
        mock_url.return_value = "http://localhost:8000"
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        result = model_health_check("test_job", 123, None)
        assert result == ("FAILED", "Connection error")


def test_create_table_with_header():
    """Test that create_table creates a table with the correct header."""
    table = create_table("Key", "Value")
    assert table.columns[0].header == "Key"
    assert table.columns[1].header == "Value"
    assert table.show_header is True


def test_create_table_without_header():
    """Test create_table without header."""
    table = create_table(show_header=False)
    assert table.show_header is False


def test_load_config_default_only():
    """Test loading the actual default configuration file from the filesystem."""
    configs = load_config()

    # Verify at least one known model exists
    model_names = {m.model_name for m in configs}
    assert "c4ai-command-r-plus" in model_names

    # Verify full configuration of a sample model
    model = next(m for m in configs if m.model_name == "c4ai-command-r-plus")
    assert model.model_family == "c4ai-command-r"
    assert model.model_type == "LLM"
    assert model.gpus_per_node == 4
    assert model.num_nodes == 2
    assert model.max_model_len == 8192
    assert model.pipeline_parallelism is True


def test_load_config_with_user_override(tmp_path, monkeypatch):
    """Test user config overriding default values."""
    # Create user config with override and new model
    user_config = tmp_path / "user_config.yaml"
    user_config.write_text("""\
models:
  c4ai-command-r-plus:
    gpus_per_node: 8
  new-model:
    model_family: new-family
    model_type: VLM
    gpus_per_node: 4
    num_nodes: 1
    vocab_size: 256000
    max_model_len: 4096
""")

    with monkeypatch.context() as m:
        m.setenv("VEC_INF_CONFIG", str(user_config))
        configs = load_config()
        config_map = {m.model_name: m for m in configs}

    # Verify override (merged with defaults)
    assert config_map["c4ai-command-r-plus"].gpus_per_node == 8
    assert config_map["c4ai-command-r-plus"].num_nodes == 2
    assert config_map["c4ai-command-r-plus"].vocab_size == 256000

    # Verify new model
    new_model = config_map["new-model"]
    assert new_model.model_family == "new-family"
    assert new_model.model_type == "VLM"
    assert new_model.gpus_per_node == 4
    assert new_model.vocab_size == 256000


def test_load_config_invalid_user_model(tmp_path):
    """Test validation of user-provided model configurations."""
    invalid_config = tmp_path / "bad_config.yaml"
    invalid_config.write_text("""\
models:
  invalid-model:
    model_family: ""
    model_type: INVALID_TYPE
    num_gpus: 0
    num_nodes: -1
""")

    with (
        pytest.raises(ValueError) as excinfo,
        patch.dict(os.environ, {"VEC_INF_CONFIG": str(invalid_config)}),
    ):
        load_config()

    assert "validation error" in str(excinfo.value).lower()
    assert "model_type" in str(excinfo.value)
    assert "num_gpus" in str(excinfo.value)
