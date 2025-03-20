"""Tests for the CLI module of the vec_inf package."""

import json
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import requests
import yaml
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
        return (f"Submitted batch job {job_id}", "")

    return _output


@pytest.fixture
def mock_status_output():
    """Fixture providing consistent mock status output."""

    def _output(job_id, job_state):
        return f"""
JobId={job_id} JobName=Meta-Llama-3.1-8B UserId=userid(54321) GroupId=groupid(12321) MCS_label=N/A Priority=2173 Nice=0 Account=vector QOS=m2 JobState={job_state} Reason=None
        """.strip()

    return _output


@pytest.fixture
def mock_exists():
    """Fixture providing path existence checks."""

    def _exists(path):
        # Return False for CACHED_CONFIG to fall back to default config
        return not str(path).endswith("vec-inf-shared/models.yaml")

    return _exists


@pytest.fixture
def test_config_dir():
    """Fixture providing the path to the test config directory."""
    # Go up to project root, then into vec_inf/config
    return Path(__file__).resolve().parent.parent.parent.parent / "vec_inf" / "config"


@pytest.fixture
def path_exists(mock_exists, test_config_dir):
    """Fixture providing path existence checks."""

    def _exists(p):
        # Allow access to the default config file
        if str(p).endswith("config/models.yaml"):
            return True
        # Allow access to the default log directory
        if str(p).endswith(".vec-inf-logs"):
            return True
        # Use mock_exists for other paths
        return mock_exists(p)

    return _exists


@pytest.fixture
def debug_helper(test_config_dir):
    """Fixture providing debug helper functions and tracked file operations."""

    class DebugHelper:
        def __init__(self):
            self.open_calls = []
            self.config_file = test_config_dir / "models.yaml"
            with open(self.config_file, "r") as f:
                self.config_content = f.read()
                self.yaml_content = yaml.safe_load(self.config_content)

        def print_debug_info(self, result):
            """Print debug information for test results."""
            print("\n=== TEST ERROR DETAILS ===")
            print(f"Config file path: {self.config_file}")
            print(f"Config file exists: {self.config_file.exists()}")
            print(f"Config dir contents: {list(test_config_dir.iterdir())}")
            print(f"Exit Code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output:\n{result.output}")

            print("\n=== FILE OPEN CALLS ===")
            for args, kwargs in self.open_calls:
                print(f"Open called with: args={args}, kwargs={kwargs}")

            if hasattr(result.exception, "__traceback__"):
                print("\n=== STACK TRACE ===")
                print("".join(traceback.format_tb(result.exception.__traceback__)))

            # Try to parse and print JSON output if present
            try:
                if result.output:
                    print("\n=== PARSED OUTPUT ===")
                    # Try direct JSON parsing first
                    try:
                        parsed = json.loads(result.output)
                    except json.JSONDecodeError:
                        # Fall back to ast.literal_eval for Python dict format
                        import ast

                        parsed = ast.literal_eval(result.output)
                    print("Keys found:", list(parsed.keys()))
                    print("Full parsed output:", parsed)
            except Exception as e:
                print(f"Failed to parse output: {e}")

        def tracked_mock_open(self, *args, **kwargs):
            """Track file open operations and return mock."""
            self.open_calls.append((args, kwargs))
            return mock_open(read_data=self.config_content)(*args, **kwargs)

    return DebugHelper()


@pytest.fixture
def test_paths():
    """Fixture providing common test paths."""
    return {
        "log_dir": Path("/tmp/test_vec_inf_logs"),
        "weights_dir": Path("/model-weights"),
        "unknown_model": Path("/model-weights/unknown-model"),
    }


@pytest.fixture
def mock_truediv(test_paths):
    """Fixture providing path joining mock."""

    def _mock_truediv(*args):
        # Handle the case where it's called with just one argument
        if len(args) == 1:
            other = args[0]
            return test_paths.get(other, Path(str(other)))

        # Normal case with self and other
        self, other = args
        specific_paths = {
            (str(test_paths["weights_dir"]), "unknown-model"): test_paths[
                "unknown_model"
            ],
            (str(test_paths["log_dir"]), other): test_paths["log_dir"] / other,
            (
                str(test_paths["log_dir"] / "model_family_placeholder"),
                other,
            ): test_paths["log_dir"] / "model_family_placeholder" / other,
            ("/home/user", ".vec-inf-logs"): test_paths["log_dir"],
        }

        return specific_paths.get((str(self), other), Path(str(self)) / str(other))

    return _mock_truediv


def create_path_exists(test_paths, path_exists, exists_paths=None):
    """Create a path existence checker.

    Args:
        test_paths: Dictionary containing test paths
        path_exists: Default path existence checker
        exists_paths: Optional list of paths that should exist
    """

    def _custom_path_exists(p):
        str_path = str(p)
        # First check if path should explicitly exist
        if exists_paths is not None:
            for path in exists_paths:
                if str_path == str(path):
                    return True
        # Special handling for model weights paths
        if str_path == str(test_paths["unknown_model"]):
            # Model weights path existence depends on exists_paths
            return (
                exists_paths is not None and test_paths["unknown_model"] in exists_paths
            )
        if str_path == str(test_paths["weights_dir"]):
            # Model weights directory existence depends on exists_paths
            return (
                exists_paths is not None and test_paths["weights_dir"] in exists_paths
            )
        # Fall back to default path_exists for other paths
        return path_exists(p)

    return _custom_path_exists


@pytest.fixture
def base_patches(test_paths, mock_truediv, debug_helper):
    """Fixture providing common patches for tests."""
    return [
        patch("pathlib.Path.mkdir"),
        patch("builtins.open", debug_helper.tracked_mock_open),
        patch("pathlib.Path.open", debug_helper.tracked_mock_open),
        patch("pathlib.Path.expanduser", return_value=test_paths["log_dir"]),
        patch("pathlib.Path.resolve", return_value=debug_helper.config_file.parent),
        patch(
            "pathlib.Path.parent", return_value=debug_helper.config_file.parent.parent
        ),
        patch("pathlib.Path.__truediv__", side_effect=mock_truediv),
        patch("pathlib.Path.iterdir", return_value=[]),  # Mock empty directory listing
        patch("json.dump"),
        patch("pathlib.Path.touch"),
        patch("vec_inf.cli._helper.Path", return_value=test_paths["weights_dir"]),
        patch(
            "pathlib.Path.home", return_value=Path("/home/user")
        ),  # Mock home directory
    ]


@pytest.fixture
def apply_base_patches(base_patches):
    """Fixture to apply all base patches."""
    with ExitStack() as stack:
        # Apply all patches
        for patch_obj in base_patches:
            stack.enter_context(patch_obj)
        yield


def test_launch_command_success(runner, mock_launch_output, path_exists, debug_helper):
    """Test successful model launch with minimal required arguments."""
    test_log_dir = Path("/tmp/test_vec_inf_logs")

    with (
        patch("vec_inf.cli._utils.run_bash_command") as mock_run,
        patch("pathlib.Path.mkdir"),
        patch("builtins.open", debug_helper.tracked_mock_open),
        patch("pathlib.Path.open", debug_helper.tracked_mock_open),
        patch("pathlib.Path.exists", new=path_exists),
        patch("pathlib.Path.expanduser", return_value=test_log_dir),
        patch("pathlib.Path.resolve", return_value=debug_helper.config_file.parent),
        patch(
            "pathlib.Path.parent", return_value=debug_helper.config_file.parent.parent
        ),
        patch("json.dump"),
        patch("pathlib.Path.touch"),
        patch("pathlib.Path.__truediv__", return_value=test_log_dir),
    ):
        expected_job_id = "14933053"
        mock_run.return_value = mock_launch_output(expected_job_id)

        result = runner.invoke(cli, ["launch", "Meta-Llama-3.1-8B"])
        debug_helper.print_debug_info(result)

        assert result.exit_code == 0
        assert expected_job_id in result.output
        mock_run.assert_called_once()


def test_launch_command_with_json_output(
    runner, mock_launch_output, path_exists, debug_helper
):
    """Test JSON output format for launch command."""
    test_log_dir = Path("/tmp/test_vec_inf_logs")
    with (
        patch("vec_inf.cli._utils.run_bash_command") as mock_run,
        patch("pathlib.Path.mkdir"),
        patch("builtins.open", debug_helper.tracked_mock_open),
        patch("pathlib.Path.open", debug_helper.tracked_mock_open),
        patch("pathlib.Path.exists", new=path_exists),
        patch("pathlib.Path.expanduser", return_value=test_log_dir),
        patch("pathlib.Path.resolve", return_value=debug_helper.config_file.parent),
        patch(
            "pathlib.Path.parent", return_value=debug_helper.config_file.parent.parent
        ),
        patch("json.dump"),
        patch("pathlib.Path.touch"),
        patch("pathlib.Path.__truediv__", return_value=test_log_dir),
    ):
        expected_job_id = "14933051"
        mock_run.return_value = mock_launch_output(expected_job_id)

        result = runner.invoke(cli, ["launch", "Meta-Llama-3.1-8B", "--json-mode"])
        debug_helper.print_debug_info(result)

        assert result.exit_code == 0

        # Try to fix single quotes to double quotes if needed
        try:
            output = json.loads(result.output)
        except json.JSONDecodeError:
            # If direct parsing fails, try to fix the format
            import ast

            # First convert string to dict using ast.literal_eval
            output_dict = ast.literal_eval(result.output)
            # Then convert back to proper JSON string
            output = json.loads(json.dumps(output_dict))

        assert output.get("slurm_job_id") == expected_job_id
        assert output.get("model_name") == "Meta-Llama-3.1-8B"
        assert output.get("model_type") == "LLM"
        assert str(test_log_dir) in output.get("log_dir", "")


def test_launch_command_model_not_in_config_with_weights(
    runner, mock_launch_output, path_exists, debug_helper, test_paths, base_patches
):
    """Test handling of a model that's not in config but has weights."""
    custom_path_exists = create_path_exists(
        test_paths,
        path_exists,
        exists_paths=[
            test_paths["unknown_model"],
            test_paths["weights_dir"],
        ],  # Ensure both paths exist
    )

    with ExitStack() as stack:
        # Apply all base patches
        for patch_obj in base_patches:
            stack.enter_context(patch_obj)
        # Apply specific patches for this test
        mock_run = stack.enter_context(patch("vec_inf.cli._utils.run_bash_command"))
        stack.enter_context(patch("pathlib.Path.exists", new=custom_path_exists))

        expected_job_id = "14933051"
        mock_run.return_value = mock_launch_output(expected_job_id)

        result = runner.invoke(cli, ["launch", "unknown-model"])
        debug_helper.print_debug_info(result)

        assert result.exit_code == 0
        assert (
            "Warning: 'unknown-model' configuration not found in config"
            in result.output
        )


def test_launch_command_model_not_found(
    runner, path_exists, debug_helper, test_paths, base_patches
):
    """Test handling of a model that's neither in config nor has weights."""

    def custom_path_exists(p):
        str_path = str(p)
        # Always return False for model weights paths
        if str_path == str(test_paths["unknown_model"]) or str_path == str(
            test_paths["weights_dir"]
        ):
            return False
        # Allow access to the default config file
        return str_path.endswith("config/models.yaml")

    with ExitStack() as stack:
        # Apply all base patches except the Path mock
        for patch_obj in base_patches[:-1]:  # Skip the last patch which is Path mock
            stack.enter_context(patch_obj)

        # Apply specific patches for this test
        stack.enter_context(patch("pathlib.Path.exists", new=custom_path_exists))

        # Mock Path to return the weights dir path
        stack.enter_context(
            patch("vec_inf.cli._helper.Path", return_value=test_paths["weights_dir"])
        )

        result = runner.invoke(cli, ["launch", "unknown-model"])
        debug_helper.print_debug_info(result)

        assert result.exit_code == 1
        assert (
            "'unknown-model' not found in configuration and model weights not found"
            in result.output
        )


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


def test_metrics_command_pending_server(
    runner, mock_status_output, path_exists, debug_helper, apply_base_patches
):
    """Test metrics command when server is pending."""
    with (
        patch("vec_inf.cli._utils.run_bash_command") as mock_run,
        patch("pathlib.Path.exists", new=path_exists),
        patch("vec_inf.cli._utils.get_base_url", return_value="URL NOT FOUND"),
    ):
        job_id = 12345
        mock_run.return_value = (mock_status_output(job_id, "PENDING"), "")

        result = runner.invoke(cli, ["metrics", str(job_id)])
        debug_helper.print_debug_info(result)

        assert result.exit_code == 0
        assert "Server State" in result.output
        assert "PENDING" in result.output
        assert (
            "Metrics endpoint unavailable - Pending resources for server"
            in result.output
        )


def test_metrics_command_server_not_ready(
    runner, mock_status_output, path_exists, debug_helper, apply_base_patches
):
    """Test metrics command when server is running but not ready."""
    with (
        patch("vec_inf.cli._utils.run_bash_command") as mock_run,
        patch("pathlib.Path.exists", new=path_exists),
        patch("vec_inf.cli._utils.get_base_url", return_value="Server not ready"),
    ):
        job_id = 12345
        mock_run.return_value = (mock_status_output(job_id, "RUNNING"), "")

        result = runner.invoke(cli, ["metrics", str(job_id)])
        debug_helper.print_debug_info(result)

        assert result.exit_code == 0
        assert "Server State" in result.output
        assert "RUNNING" in result.output
        assert "Server not ready" in result.output


@patch("vec_inf.cli._helper.requests.get")
def test_metrics_command_server_ready(
    mock_get, runner, mock_status_output, path_exists, debug_helper, apply_base_patches
):
    """Test metrics command when server is ready and returning metrics."""
    metrics_response = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="test-model"} 100.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="test-model"} 500.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="test-model"} 0.5
"""
    mock_response = mock_get.return_value
    mock_response.text = metrics_response
    mock_response.status_code = 200

    with (
        patch("vec_inf.cli._utils.run_bash_command") as mock_run,
        patch("pathlib.Path.exists", new=path_exists),
        patch("vec_inf.cli._utils.get_base_url", return_value="http://test:8000/v1"),
        patch("time.sleep", side_effect=KeyboardInterrupt),  # Break the infinite loop
    ):
        job_id = 12345
        mock_run.return_value = (mock_status_output(job_id, "RUNNING"), "")

        result = runner.invoke(cli, ["metrics", str(job_id)], catch_exceptions=False)
        debug_helper.print_debug_info(result)

        # KeyboardInterrupt is expected and ok
        assert "Prompt Throughput" in result.output
        assert "Generation Throughput" in result.output
        assert "GPU Cache Usage" in result.output
        assert "50.0%" in result.output  # 0.5 converted to percentage


@patch("vec_inf.cli._helper.requests.get")
def test_metrics_command_request_failed(
    mock_get, runner, mock_status_output, path_exists, debug_helper, apply_base_patches
):
    """Test metrics command when request to metrics endpoint fails."""
    mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

    with (
        patch("vec_inf.cli._utils.run_bash_command") as mock_run,
        patch("pathlib.Path.exists", new=path_exists),
        patch("vec_inf.cli._utils.get_base_url", return_value="http://test:8000/v1"),
        patch("time.sleep", side_effect=KeyboardInterrupt),  # Break the infinite loop
    ):
        job_id = 12345
        mock_run.return_value = (mock_status_output(job_id, "RUNNING"), "")

        result = runner.invoke(cli, ["metrics", str(job_id)], catch_exceptions=False)
        debug_helper.print_debug_info(result)

        # KeyboardInterrupt is expected and ok
        assert "Server State" in result.output
        assert "RUNNING" in result.output
        assert (
            "Metrics request failed, `metrics` endpoint might not be ready"
            in result.output
        )
        assert "Connection refused" in result.output
