"""Tests for the utility functions in the CLI module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from vec_inf.cli._utils import (
    MODEL_READY_SIGNATURE,
    SERVER_ADDRESS_SIGNATURE,
    create_table,
    get_base_url,
    get_latest_metric,
    is_server_running,
    load_default_args,
    load_models_df,
    model_health_check,
    read_slurm_log,
    run_bash_command,
)


@pytest.fixture
def sample_models_csv(tmp_path):
    """Create a sample models CSV file."""
    csv_content = """model_name,model_type,param1,param2
model_a,type1,value1,value2
model_b,type2,value3,value4"""
    path = tmp_path / "models.csv"
    path.write_text(csv_content)
    return path


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
        result = run_bash_command("echo test")
        assert result == "test output"


def test_run_bash_command_error():
    """Test run_bash_command with error output."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "error output")
        mock_popen.return_value = mock_process
        result = run_bash_command("invalid_command")
        assert result == ""


def test_read_slurm_log_found(mock_log_dir):
    """Test that read_slurm_log reads the content of a log file."""
    test_content = ["line1\n", "line2\n"]
    log_file = mock_log_dir / "test_job.123.err"
    log_file.write_text("".join(test_content))
    result = read_slurm_log("test_job", 123, "err", mock_log_dir)
    assert result == test_content


def test_read_slurm_log_not_found():
    """Test read_slurm_log, return an error message if the log file is not found."""
    result = read_slurm_log("missing_job", 456, "err", "/nonexistent")
    assert result == "LOG_FILE_NOT_FOUND"


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
    test_line = f"{SERVER_ADDRESS_SIGNATURE}http://localhost:8000\n"
    with patch("vec_inf.cli._utils.read_slurm_log") as mock_read:
        mock_read.return_value = [test_line]
        result = get_base_url("test_job", 123, None)
        assert result == "http://localhost:8000"


def test_get_base_url_not_found():
    """Test get_base_url when URL is not found in logs."""
    with patch("vec_inf.cli._utils.read_slurm_log") as mock_read:
        mock_read.return_value = ["some other content"]
        result = get_base_url("test_job", 123, None)
        assert result == "URL_NOT_FOUND"


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


def test_load_models_df(sample_models_csv):
    """Test that load_models_df loads the models CSV file correctly."""
    models_dir = sample_models_csv.parent / "models"
    models_dir.mkdir()
    (models_dir / "models.csv").write_text(sample_models_csv.read_text())

    with (
        patch.object(os.path, "dirname") as mock_dirname,
        patch.object(os.path, "realpath") as mock_realpath,
    ):
        mock_realpath.return_value = str(models_dir / "dummy_file.py")
        mock_dirname.return_value = str(models_dir.parent)
        df = load_models_df()
        assert df.shape == (2, 4)
        assert "model_name" in df.columns


def test_load_default_args(sample_models_csv):
    """Test that load_default_args returns the default arguments for a model."""
    models_dir = sample_models_csv.parent / "models"
    models_dir.mkdir()
    (models_dir / "models.csv").write_text(sample_models_csv.read_text())

    with (
        patch.object(os.path, "dirname") as mock_dirname,
        patch.object(os.path, "realpath") as mock_realpath,
    ):
        mock_realpath.return_value = str(models_dir / "dummy_file.py")
        mock_dirname.return_value = str(models_dir.parent)
        df = load_models_df()
        args = load_default_args(df, "model_a")
        assert args == {"model_type": "type1", "param1": "value1", "param2": "value2"}


@pytest.mark.parametrize(
    "log_lines,expected",
    [
        (
            ["2023-01-01 [INFO] Avg prompt throughput: 5.2, Avg token throughput: 3.1"],
            {"Avg prompt throughput": "5.2", "Avg token throughput": "3.1"},
        ),
        (["No metrics here"], {}),
        ([], {}),
        (["Invalid metric format"], {}),
    ],
)
def test_get_latest_metric(log_lines, expected):
    """Test that get_latest_metric returns the latest metric entry."""
    result = get_latest_metric(log_lines)
    assert result == expected
