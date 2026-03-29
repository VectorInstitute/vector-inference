"""Tests for the CLI helper classes."""

import json
from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.table import Table

from vec_inf.cli._helper import (
    BatchLaunchResponseFormatter,
    LaunchResponseFormatter,
    ListCmdDisplay,
    ListStatusDisplay,
    MetricsResponseFormatter,
    StatusResponseFormatter,
)
from vec_inf.client import ModelConfig, ModelInfo, StatusResponse


class TestLaunchResponseFormatter:
    """Test cases for LaunchResponseFormatter."""

    def test_format_table_output(self):
        """Test formatting launch response as a table."""
        model_name = "Meta-Llama-3.1-8B"
        params = {
            "slurm_job_id": "14933053",
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
            "venv": "/path/to/venv",
            "engine": "vllm",
            "engine_args": {"max_model_len": 8192, "enable_prefix_caching": True},
            "env": {"CACHE": "/cache"},
        }

        formatter = LaunchResponseFormatter(model_name, params)
        table = formatter.format_table_output()

        # Just check that it returns a Table object
        assert isinstance(table, Table)
        # Check that it has some rows
        assert len(table.rows) > 0

    def test_format_table_output_with_minimal_params(self):
        """Test formatting with minimal required parameters."""
        model_name = "test-model"
        params = {
            "slurm_job_id": "12345",
            "model_type": "LLM",
            "vocab_size": "50000",
            "partition": "cpu",
            "qos": "low",
            "time": "30:00",
            "num_nodes": "1",
            "gpus_per_node": "0",
            "cpus_per_task": "4",
            "mem_per_node": "16G",
            "model_weights_parent_dir": "/weights",
            "log_dir": "/logs",
            "venv": "/path/to/venv",
            "engine": "vllm",
            "engine_args": {},
            "env": {},
        }

        formatter = LaunchResponseFormatter(model_name, params)
        table = formatter.format_table_output()

        assert isinstance(table, Table)
        assert len(table.rows) > 0


class TestBatchLaunchResponseFormatter:
    """Test cases for BatchLaunchResponseFormatter."""

    def test_format_table_output(self):
        """Test formatting batch launch response as a table."""
        params = {
            "slurm_job_id": "14933053",
            "slurm_job_name": "BATCH-job",
            "model_names": ["model1", "model2"],
            "log_dir": "/tmp/logs",  # Moved to top level
            "models": {
                "model1": {
                    "model_name": "model1",
                    "partition": "gpu",
                    "qos": "normal",
                    "time": "1:00:00",
                    "num_nodes": "1",
                    "gpus_per_node": "1",
                    "cpus_per_task": "8",
                    "mem_per_node": "32G",
                    "engine": "vllm",
                },
                "model2": {
                    "model_name": "model2",
                    "partition": "gpu",
                    "qos": "high",
                    "time": "2:00:00",
                    "num_nodes": "2",
                    "gpus_per_node": "2",
                    "cpus_per_task": "16",
                    "mem_per_node": "64G",
                    "engine": "vllm",
                },
            },
        }

        formatter = BatchLaunchResponseFormatter(params)
        table = formatter.format_table_output()

        assert isinstance(table, Table)
        assert len(table.rows) > 0

    def test_format_table_output_single_model(self):
        """Test formatting batch launch with single model."""
        params = {
            "slurm_job_id": "12345",
            "slurm_job_name": "SINGLE-job",
            "model_names": ["single-model"],
            "log_dir": "/logs",  # Moved to top level
            "models": {
                "single-model": {
                    "model_name": "single-model",
                    "partition": "cpu",
                    "qos": "low",
                    "time": "30:00",
                    "num_nodes": "1",
                    "gpus_per_node": "0",
                    "cpus_per_task": "4",
                    "mem_per_node": "16G",
                    "engine": "vllm",
                }
            },
        }

        formatter = BatchLaunchResponseFormatter(params)
        table = formatter.format_table_output()

        assert isinstance(table, Table)
        assert len(table.rows) > 0


class TestStatusResponseFormatter:
    """Test cases for StatusResponseFormatter."""

    def test_output_table_ready_status(self):
        """Test formatting status response for ready server."""
        status_info = StatusResponse(
            model_name="Meta-Llama-3.1-8B",
            log_dir="/tmp/logs",
            server_status="READY",
            job_state="RUNNING",
            raw_output="JobState=RUNNING",
            base_url="http://localhost:8000",
            pending_reason=None,
            failed_reason=None,
        )

        formatter = StatusResponseFormatter(status_info)
        table = formatter.output_table()

        assert isinstance(table, Table)
        assert len(table.rows) > 0

    def test_output_table_pending_status(self):
        """Test formatting status response for pending server."""
        status_info = StatusResponse(
            model_name="test-model",
            log_dir="/tmp/logs",
            server_status="PENDING",
            job_state="PENDING",
            raw_output="JobState=PENDING",
            base_url=None,
            pending_reason="Waiting for resources",
            failed_reason=None,
        )

        formatter = StatusResponseFormatter(status_info)
        table = formatter.output_table()

        assert isinstance(table, Table)
        assert len(table.rows) > 0

    def test_output_table_failed_status(self):
        """Test formatting status response for failed server."""
        status_info = StatusResponse(
            model_name="failed-model",
            log_dir="/tmp/logs",
            server_status="FAILED",
            job_state="FAILED",
            raw_output="JobState=FAILED",
            base_url=None,
            pending_reason=None,
            failed_reason="Out of memory",
        )

        formatter = StatusResponseFormatter(status_info)
        table = formatter.output_table()

        assert isinstance(table, Table)
        assert len(table.rows) > 0

    @patch("click.echo")
    def test_output_json_ready_status(self, mock_echo):
        """Test JSON output for ready server status."""
        status_info = StatusResponse(
            model_name="Meta-Llama-3.1-8B",
            log_dir="/tmp/logs",
            server_status="READY",
            job_state="RUNNING",
            raw_output="JobState=RUNNING",
            base_url="http://localhost:8000",
            pending_reason=None,
            failed_reason=None,
        )

        formatter = StatusResponseFormatter(status_info)
        formatter.output_json()

        mock_echo.assert_called_once()
        # Just check that it was called with a json compatible string
        output = mock_echo.call_args[0][0]
        json_dict = json.loads(output)
        assert isinstance(json_dict, dict)
        assert "model_name" in json_dict

    @patch("click.echo")
    def test_output_json_with_error_reasons(self, mock_echo):
        """Test JSON output with error reasons."""
        status_info = StatusResponse(
            model_name="test-model",
            log_dir="/tmp/logs",
            server_status="FAILED",
            job_state="FAILED",
            raw_output="JobState=FAILED",
            base_url=None,
            pending_reason="Resource allocation",
            failed_reason="Configuration error",
        )

        formatter = StatusResponseFormatter(status_info)
        formatter.output_json()

        mock_echo.assert_called_once()
        output = mock_echo.call_args[0][0]
        json_dict = json.loads(output)
        assert isinstance(json_dict, dict)
        assert "pending_reason" in json_dict
        assert "failed_reason" in json_dict


class TestMetricsResponseFormatter:
    """Test cases for MetricsResponseFormatter."""

    def test_format_metrics_basic(self):
        """Test formatting basic metrics."""
        metrics = {
            "prompt_tokens_per_sec": 100.5,
            "generation_tokens_per_sec": 250.2,
            "requests_running": 3,
            "requests_waiting": 1,
            "requests_swapped": 0,
            "gpu_cache_usage": 0.75,
            "cpu_cache_usage": 0.25,
            "total_prompt_tokens": 1000,
            "total_generation_tokens": 2500,
            "successful_requests_total": 10,
        }

        formatter = MetricsResponseFormatter(metrics)
        formatter.format_metrics()

        assert isinstance(formatter.table, Table)
        assert len(formatter.table.rows) > 0

    def test_format_metrics_with_prefix_caching(self):
        """Test formatting metrics with prefix caching enabled."""
        metrics = {
            "prompt_tokens_per_sec": 100.0,
            "generation_tokens_per_sec": 200.0,
            "requests_running": 2,
            "requests_waiting": 0,
            "requests_swapped": 0,
            "gpu_cache_usage": 0.5,
            "cpu_cache_usage": 0.1,
            "gpu_prefix_cache_hit_rate": 0.8,
            "cpu_prefix_cache_hit_rate": 0.6,
            "total_prompt_tokens": 500,
            "total_generation_tokens": 1000,
            "successful_requests_total": 5,
        }

        formatter = MetricsResponseFormatter(metrics)
        formatter.format_metrics()

        assert formatter.enabled_prefix_caching is True
        assert isinstance(formatter.table, Table)
        assert len(formatter.table.rows) > 0

    def test_format_metrics_with_latency(self):
        """Test formatting metrics with latency information."""
        metrics = {
            "prompt_tokens_per_sec": 150.0,
            "generation_tokens_per_sec": 300.0,
            "requests_running": 1,
            "requests_waiting": 0,
            "requests_swapped": 0,
            "gpu_cache_usage": 0.3,
            "cpu_cache_usage": 0.1,
            "avg_request_latency": 2.5,
            "total_prompt_tokens": 750,
            "total_generation_tokens": 1500,
            "successful_requests_total": 8,
        }

        formatter = MetricsResponseFormatter(metrics)
        formatter.format_metrics()

        assert isinstance(formatter.table, Table)
        assert len(formatter.table.rows) > 0

    def test_format_failed_metrics(self):
        """Test formatting failed metrics with error message."""
        error_message = "ERROR: Server not ready"

        formatter = MetricsResponseFormatter(error_message)
        formatter.format_failed_metrics(error_message)

        assert isinstance(formatter.table, Table)
        assert len(formatter.table.rows) > 0

    def test_set_metrics_with_string(self):
        """Test _set_metrics with string input."""
        formatter = MetricsResponseFormatter("Error message")
        assert formatter.metrics == {}

    def test_set_metrics_with_dict(self):
        """Test _set_metrics with dictionary input."""
        metrics_dict = {"prompt_tokens_per_sec": 100.0}
        formatter = MetricsResponseFormatter(metrics_dict)
        assert formatter.metrics == metrics_dict

    def test_check_prefix_caching_enabled(self):
        """Test prefix caching detection when enabled."""
        metrics = {"gpu_prefix_cache_hit_rate": 0.8}
        formatter = MetricsResponseFormatter(metrics)
        assert formatter.enabled_prefix_caching is True

    def test_check_prefix_caching_disabled(self):
        """Test prefix caching detection when disabled."""
        metrics = {"prompt_tokens_per_sec": 100.0}
        formatter = MetricsResponseFormatter(metrics)
        assert formatter.enabled_prefix_caching is False


class TestListCmdDisplay:
    """Test cases for ListCmdDisplay."""

    def test_init(self):
        """Test ListCmdDisplay initialization."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=False)

        assert display.console is console
        assert display.json_mode is False
        assert display.model_config is None
        assert display.model_names == []

    def test_init_json_mode(self):
        """Test ListCmdDisplay initialization with JSON mode."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=True)

        assert display.json_mode is True

    def test_format_single_model_output_table(self):
        """Test formatting single model output as table."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=False)

        # Create a mock ModelConfig
        mock_config = MagicMock(spec=ModelConfig)
        mock_config.model_dump.return_value = {
            "model_name": "Meta-Llama-3.1-8B",
            "model_family": "Meta-Llama-3.1",
            "model_type": "LLM",
            "model_weights_parent_dir": "/model-weights",
            "vllm_args": {"max_model_len": 8192},
            "venv": "/path/to/venv",
            "log_dir": "/logs",
        }

        result = display._format_single_model_output(mock_config)

        assert isinstance(result, Table)
        assert len(result.rows) > 0

    def test_format_single_model_output_json(self):
        """Test formatting single model output as JSON."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=True)

        # Create a mock ModelConfig
        mock_config = MagicMock(spec=ModelConfig)
        mock_config.model_dump.return_value = {
            "model_name": "Meta-Llama-3.1-8B",
            "model_family": "Meta-Llama-3.1",
            "model_type": "LLM",
            "model_weights_parent_dir": "/model-weights",
            "vllm_args": {"max_model_len": 8192},
            "venv": "/path/to/venv",
            "log_dir": "/logs",
        }

        result = display._format_single_model_output(mock_config)

        assert isinstance(result, str)  # Changed from dict to str
        # Parse the JSON string to check content
        result_dict = json.loads(result)
        assert "model_name" in result_dict
        assert result_dict["model_name"] == "Meta-Llama-3.1-8B"

    def test_format_all_models_output(self):
        """Test formatting all models output."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=False)

        # Create mock ModelInfo objects
        model_info1 = MagicMock(spec=ModelInfo)
        model_info1.name = "Meta-Llama-3.1-8B"
        model_info1.family = "Meta-Llama-3.1"
        model_info1.model_type = "LLM"
        model_info1.variant = None

        model_info2 = MagicMock(spec=ModelInfo)
        model_info2.name = "CLIP-ViT-B-32"
        model_info2.family = "CLIP"
        model_info2.model_type = "VLM"
        model_info2.variant = "ViT-B-32"

        model_infos = [model_info1, model_info2]

        result = display._format_all_models_output(model_infos)

        assert isinstance(result, list)
        assert len(result) == 2

    @patch("click.echo")
    def test_display_single_model_output_json(self, mock_echo):
        """Test displaying single model output in JSON mode."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=True)

        mock_config = MagicMock(spec=ModelConfig)
        mock_config.model_dump.return_value = {
            "model_name": "test-model",
            "model_family": "test-family",
            "model_type": "LLM",
            "model_weights_parent_dir": "/weights",
            "vllm_args": {},
            "venv": "/venv",
            "log_dir": "/logs",
        }

        display.display_single_model_output(mock_config)

        mock_echo.assert_called_once()

    def test_display_single_model_output_table(self):
        """Test displaying single model output in table mode."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=False)

        mock_config = MagicMock(spec=ModelConfig)
        mock_config.model_dump.return_value = {
            "model_name": "test-model",
            "model_family": "test-family",
            "model_type": "LLM",
            "model_weights_parent_dir": "/weights",
            "vllm_args": {},
            "venv": "/venv",
            "log_dir": "/logs",
        }

        # Mock the console.print method
        with patch.object(console, "print") as mock_print:
            display.display_single_model_output(mock_config)
            mock_print.assert_called_once()

    @patch("click.echo")
    def test_display_all_models_output_json(self, mock_echo):
        """Test displaying all models output in JSON mode."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=True)

        model_info = MagicMock(spec=ModelInfo)
        model_info.name = "test-model"
        model_infos = [model_info]

        display.display_all_models_output(model_infos)

        mock_echo.assert_called_once()

    def test_display_all_models_output_table(self):
        """Test displaying all models output in table mode."""
        console = Console()
        display = ListCmdDisplay(console, json_mode=False)

        model_info = MagicMock(spec=ModelInfo)
        model_info.name = "test-model"
        model_info.family = "test-family"
        model_info.model_type = "LLM"
        model_info.variant = None
        model_infos = [model_info]

        # Mock the console.print method
        with patch.object(console, "print") as mock_print:
            display.display_all_models_output(model_infos)
            mock_print.assert_called_once()


class TestListStatusDisplay:
    """Test cases for ListStatusDisplay."""

    def test_init(self):
        """Test ListStatusDisplay initialization."""
        job_ids = ["12345", "67890"]
        statuses = [
            StatusResponse(
                model_name="test-model-1",
                log_dir="/tmp/logs",
                server_status="READY",
                job_state="RUNNING",
                raw_output="JobState=RUNNING",
                base_url="http://localhost:8000",
                pending_reason=None,
                failed_reason=None,
            ),
            StatusResponse(
                model_name="test-model-2",
                log_dir="/tmp/logs",
                server_status="PENDING",
                job_state="PENDING",
                raw_output="JobState=PENDING",
                base_url=None,
                pending_reason="Waiting for resources",
                failed_reason=None,
            ),
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=False)

        assert display.job_ids == job_ids
        assert display.statuses == statuses
        assert display.json_mode is False
        assert isinstance(display.table, Table)

    def test_init_json_mode(self):
        """Test ListStatusDisplay initialization with JSON mode."""
        job_ids = ["12345"]
        statuses = [
            StatusResponse(
                model_name="test-model",
                log_dir="/tmp/logs",
                server_status="READY",
                job_state="RUNNING",
                raw_output="JobState=RUNNING",
                base_url="http://localhost:8000",
                pending_reason=None,
                failed_reason=None,
            )
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=True)

        assert display.json_mode is True

    def test_display_multiple_status_output_table_mode(self):
        """Test displaying multiple statuses in table mode."""
        console = Console()
        job_ids = ["12345", "67890"]
        statuses = [
            StatusResponse(
                model_name="test-model-1",
                log_dir="/tmp/logs",
                server_status="READY",
                job_state="RUNNING",
                raw_output="JobState=RUNNING",
                base_url="http://localhost:8000",
                pending_reason=None,
                failed_reason=None,
            ),
            StatusResponse(
                model_name="test-model-2",
                log_dir="/tmp/logs",
                server_status="PENDING",
                job_state="PENDING",
                raw_output="JobState=PENDING",
                base_url=None,
                pending_reason="Waiting for resources",
                failed_reason=None,
            ),
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=False)

        with patch.object(console, "print") as mock_print:
            display.display_multiple_status_output(console)
            mock_print.assert_called_once()
            # Verify the table was printed
            assert mock_print.call_args[0][0] == display.table

    def test_display_multiple_status_output_json_mode(self):
        """Test displaying multiple statuses in JSON mode."""
        console = Console()
        job_ids = ["12345", "67890"]
        statuses = [
            StatusResponse(
                model_name="test-model-1",
                log_dir="/tmp/logs",
                server_status="READY",
                job_state="RUNNING",
                raw_output="JobState=RUNNING",
                base_url="http://localhost:8000",
                pending_reason=None,
                failed_reason=None,
            ),
            StatusResponse(
                model_name="test-model-2",
                log_dir="/tmp/logs",
                server_status="FAILED",
                job_state="FAILED",
                raw_output="JobState=FAILED",
                base_url=None,
                pending_reason=None,
                failed_reason="Out of memory",
            ),
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=True)

        with patch("click.echo") as mock_echo:
            display.display_multiple_status_output(console)
            mock_echo.assert_called_once()

            # Verify JSON output
            output = mock_echo.call_args[0][0]
            json_data = json.loads(output)
            assert isinstance(json_data, list)
            assert len(json_data) == 2
            assert json_data[0]["model_name"] == "test-model-1"
            assert json_data[0]["model_status"] == "READY"
            assert json_data[0]["base_url"] == "http://localhost:8000"
            assert json_data[1]["model_name"] == "test-model-2"
            assert json_data[1]["model_status"] == "FAILED"
            assert json_data[1]["base_url"] is None

    def test_display_multiple_status_output_empty_list(self):
        """Test displaying empty status list."""
        console = Console()
        job_ids = []
        statuses = []

        display = ListStatusDisplay(job_ids, statuses, json_mode=False)

        with patch.object(console, "print") as mock_print:
            display.display_multiple_status_output(console)
            mock_print.assert_called_once()

    def test_display_multiple_status_output_empty_list_json(self):
        """Test displaying empty status list in JSON mode."""
        console = Console()
        job_ids = []
        statuses = []

        display = ListStatusDisplay(job_ids, statuses, json_mode=True)

        with patch("click.echo") as mock_echo:
            display.display_multiple_status_output(console)
            mock_echo.assert_called_once()

            output = mock_echo.call_args[0][0]
            json_data = json.loads(output)
            assert isinstance(json_data, list)
            assert len(json_data) == 0

    def test_display_multiple_status_output_single_status(self):
        """Test displaying single status."""
        console = Console()
        job_ids = ["12345"]
        statuses = [
            StatusResponse(
                model_name="single-model",
                log_dir="/tmp/logs",
                server_status="READY",
                job_state="RUNNING",
                raw_output="JobState=RUNNING",
                base_url="http://localhost:8000",
                pending_reason=None,
                failed_reason=None,
            )
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=False)

        with patch.object(console, "print") as mock_print:
            display.display_multiple_status_output(console)
            mock_print.assert_called_once()
            # Verify table has one row
            assert len(display.table.rows) == 1

    def test_display_multiple_status_output_with_none_base_url(self):
        """Test displaying statuses with None base_url."""
        console = Console()
        job_ids = ["12345"]
        statuses = [
            StatusResponse(
                model_name="pending-model",
                log_dir="/tmp/logs",
                server_status="PENDING",
                job_state="PENDING",
                raw_output="JobState=PENDING",
                base_url=None,
                pending_reason="Resource allocation",
                failed_reason=None,
            )
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=False)

        with patch.object(console, "print") as mock_print:
            display.display_multiple_status_output(console)
            mock_print.assert_called_once()
            # Verify the row was added (None base_url should be handled gracefully)
            assert len(display.table.rows) == 1
            # Verify table has correct number of columns
            assert (
                len(display.table.columns) == 4
            )  # Job ID, Model Name, Status, Base URL

    def test_display_multiple_status_output_json_with_none_values(self):
        """Test JSON output with None values."""
        console = Console()
        job_ids = ["12345"]
        statuses = [
            StatusResponse(
                model_name="pending-model",
                log_dir="/tmp/logs",
                server_status="PENDING",
                job_state="PENDING",
                raw_output="JobState=PENDING",
                base_url=None,
                pending_reason="Waiting",
                failed_reason=None,
            )
        ]

        display = ListStatusDisplay(job_ids, statuses, json_mode=True)

        with patch("click.echo") as mock_echo:
            display.display_multiple_status_output(console)
            mock_echo.assert_called_once()

            output = mock_echo.call_args[0][0]
            json_data = json.loads(output)
            assert json_data[0]["base_url"] is None
            assert json_data[0]["model_status"] == "PENDING"
