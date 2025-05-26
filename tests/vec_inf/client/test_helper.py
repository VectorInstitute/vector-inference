"""Unit tests for helper components in the vec_inf.client module."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import requests

from vec_inf.client._exceptions import (
    MissingRequiredFieldsError,
    ModelConfigurationError,
    ModelNotFoundError,
    SlurmJobError,
)
from vec_inf.client._helper import (
    ModelLauncher,
    ModelRegistry,
    ModelStatusMonitor,
    PerformanceMetricsCollector,
)
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import (
    ModelStatus,
    ModelType,
    StatusResponse,
)


class TestModelLauncher:
    """Tests for the ModelLauncher class."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Fixture providing a basic model configuration for tests."""
        return ModelConfig(
            model_name="test-model",
            model_family="test-family",
            model_type="LLM",
            gpus_per_node=1,
            num_nodes=1,
            vocab_size=32000,
            model_weights_parent_dir=Path("/path/to/models"),
            vllm_args={},
        )

    @pytest.fixture
    def mock_configs(self, model_config):
        """Fixture providing a list of model configurations."""
        return [model_config]

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_existing_config(self, mock_load_config, mock_configs):
        """Test launcher initializes correctly when config exists."""
        mock_load_config.return_value = mock_configs
        launcher = ModelLauncher("test-model", {})

        assert launcher.model_name == "test-model"
        assert launcher.model_config.model_name == "test-model"
        assert launcher.model_config.model_family == "test-family"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_init_with_missing_config_and_existing_weights(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test fallback to dummy config when config is missing but weights exist."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        with pytest.warns(UserWarning):
            launcher = ModelLauncher("unknown-model", {})

        assert launcher.model_name == "unknown-model"
        assert launcher.model_config.model_name == "unknown-model"
        assert launcher.model_config.model_family == "model_family_placeholder"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_init_with_missing_config_and_missing_weights(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test error is raised when both config and weights are missing."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = False

        with pytest.raises(ModelConfigurationError):
            ModelLauncher("unknown-model", {})

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_empty_configs(self, mock_load_config):
        """Test error is raised when config list is empty."""
        mock_load_config.return_value = []

        with pytest.raises(ModelNotFoundError):
            ModelLauncher("test-model", {})

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_vllm_args(self, mock_path_exists, mock_load_config, mock_configs):
        """Test vLLM args are parsed correctly into key-value pairs."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        vllm_args = launcher._process_vllm_args(
            "--tensor-parallel-size=4,--quantization=awq,-O3"
        )

        assert vllm_args["--tensor-parallel-size"] == "4"
        assert vllm_args["--quantization"] == "awq"
        assert vllm_args["--compilation-config"] == "3"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_merges_config_and_cli_args(
        self, mock_load_config, model_config
    ):
        """Test _get_launch_params merges config and CLI vllm_args correctly."""
        mock_load_config.return_value = [model_config]
        cli_kwargs = {"vllm_args": "--num-scheduler-steps=16", "num_nodes": 4}

        launcher = ModelLauncher("test-model", cli_kwargs)
        params = launcher.params

        assert params["num_nodes"] == "4"
        assert params["vllm_args"]["--num-scheduler-steps"] == "16"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_with_multi_gpu_no_tp(
        self, mock_load_config, model_config
    ):
        """Test the case tensor parallelism is missing in multi-GPU setup."""
        updated_config = model_config.model_copy(
            update={
                "gpus_per_node": 2,
            }
        )
        mock_load_config.return_value = [updated_config]

        with pytest.raises(MissingRequiredFieldsError) as excinfo:
            ModelLauncher("test-model", {})

        assert "--tensor-parallel-size" in str(excinfo.value)

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.SlurmScriptGenerator")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.open")
    @patch("pathlib.Path.rename")
    def test_launch(
        self,
        mock_rename,
        mock_open,
        mock_touch,
        mock_mkdir,
        mock_script_gen,
        mock_run_bash,
        mock_load_config,
        mock_configs,
    ):
        """Test successful model launch returns expected response and creates logs."""
        mock_open.return_value = mock.mock_open().return_value
        mock_load_config.return_value = mock_configs
        mock_script_gen_instance = MagicMock()
        mock_script_gen_instance.write_to_log_dir.return_value = Path(
            "/path/to/slurm_script.sh"
        )
        mock_script_gen.return_value = mock_script_gen_instance
        mock_run_bash.return_value = ("Submitted batch job 12345", "")

        launcher = ModelLauncher("test-model", {})
        response = launcher.launch()

        assert response.slurm_job_id == 12345
        assert response.model_name == "test-model"
        assert "slurm_job_id" in response.config
        assert response.config["slurm_job_id"] == "12345"

        mock_mkdir.assert_called()
        mock_touch.assert_called()
        mock_rename.assert_called()

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_launch_with_slurm_error(
        self, mock_run_bash, mock_load_config, model_config
    ):
        """Test launch raises error on SLURM submission failure."""
        mock_load_config.return_value = [model_config]
        mock_run_bash.return_value = ("", "sbatch: error: Invalid partition specified")

        launcher = ModelLauncher("test-model", {})
        with pytest.raises(SlurmJobError):
            launcher.launch()


class TestModelStatusMonitor:
    """Tests for the ModelStatusMonitor class."""

    @pytest.fixture
    def mock_scontrol_output(self):
        """Fixture returning mock `scontrol` output for a running SLURM job."""
        return "JobId=12345 JobName=test-model UserId=user(1000) GroupId=group(1000) MCS_label=N/A Priority=1 Nice=0 Account=account QOS=normal JobState=RUNNING Reason=None Dependency=(null) Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 RunTime=00:10:00 TimeLimit=1-00:00:00 TimeMin=N/A SubmitTime=2023-01-01T12:00:00 EligibleTime=2023-01-01T12:00:00 AccrueTime=2023-01-01T12:00:00 StartTime=2023-01-01T12:01:00 EndTime=2023-01-02T12:01:00 Deadline=N/A PreemptEligibleTime=2023-01-01T12:01:00 PreemptTime=None SuspendTime=None SecsPreSuspend=0 LastSchedEval=2023-01-01T12:00:30 Partition=gpu AllocNode:Sid=login01:123 ReqNodeList=(null) ExcNodeList=(null) NodeList=gpu01 BatchHost=gpu01 NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:* TRES=cpu=8,mem=32G,node=1,billing=8,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=1 MinMemoryNode=32G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/path/to/slurm_script.sh WorkDir=/home/user StdErr=/path/to/log/test-family/test-model.12345/test-model.12345.err StdIn=/dev/null StdOut=/path/to/log/test-family/test-model.12345/test-model.12345.out Power="

    @pytest.fixture
    def mock_pending_scontrol_output(self):
        """Fixture returning mock scontrol output for a pending job."""
        return "JobId=12345 JobName=test-model UserId=user(1000) GroupId=group(1000) MCS_label=N/A Priority=1 Nice=0 Account=account QOS=normal JobState=PENDING Reason=Resources NodeList=(null) BatchHost=gpu01 NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:* TRES=cpu=8,mem=32G,node=1,billing=8,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=1 MinMemoryNode=32G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/path/to/slurm_script.sh WorkDir=/home/user StdErr=/path/to/log/test-family/test-model.12345/test-model.12345.err StdIn=/dev/null StdOut=/path/to/log/test-family/test-model.12345/test-model.12345.out Power="

    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_init_sets_fields_correctly(self, mock_run_bash, mock_scontrol_output):
        """Test __init__ sets job ID, output, and initial status info."""
        mock_run_bash.return_value = (mock_scontrol_output, "")

        monitor = ModelStatusMonitor(slurm_job_id=12345, log_dir="/path/to/logs")

        assert monitor.slurm_job_id == 12345
        assert monitor.output == mock_scontrol_output
        assert monitor.log_dir == "/path/to/logs"
        assert isinstance(monitor.status_info.raw_output, str)
        assert monitor.status_info.job_state == "RUNNING"
        assert monitor.status_info.model_name == "test-model"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_init_with_slurm_error(self, mock_run_bash):
        """Test __init__ raises error when SLURM command returns stderr."""
        mock_run_bash.return_value = ("", "scontrol: error: Invalid job id specified")

        with pytest.raises(SlurmJobError):
            ModelStatusMonitor(99999)

    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_process_pending_state(self, mock_run_bash, mock_pending_scontrol_output):
        """Test pending job sets server status and pending reason correctly."""
        mock_run_bash.return_value = (mock_pending_scontrol_output, "")

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.PENDING
        assert status.pending_reason == "Resources"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.utils.is_server_running")
    @patch("vec_inf.client._helper.utils.model_health_check")
    @patch("vec_inf.client._helper.utils.get_base_url")
    def test_process_running_state_server_ready(
        self,
        mock_get_base_url,
        mock_health_check,
        mock_is_server_running,
        mock_run_bash,
        mock_scontrol_output,
    ):
        """Test READY status and base URL are set when server is healthy and running."""
        mock_run_bash.return_value = (mock_scontrol_output, "")
        mock_is_server_running.return_value = "RUNNING"
        mock_health_check.return_value = (ModelStatus.READY, 200)
        mock_get_base_url.return_value = "http://gpu01:8000/v1"

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.READY
        assert status.base_url == "http://gpu01:8000/v1"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.utils.is_server_running")
    def test_process_running_state_server_failed(
        self, mock_is_server_running, mock_run_bash, mock_scontrol_output
    ):
        """Test that FAILED status and reason are set when server startup fails."""
        mock_run_bash.return_value = (mock_scontrol_output, "")
        mock_is_server_running.return_value = (
            ModelStatus.FAILED,
            "RuntimeError: CUDA out of memory",
        )

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.FAILED
        assert status.failed_reason == "RuntimeError: CUDA out of memory"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.utils.is_server_running")
    @patch("vec_inf.client._helper.utils.model_health_check")
    def test_process_running_state_health_check_failed(
        self,
        mock_health_check,
        mock_is_server_running,
        mock_run_bash,
        mock_scontrol_output,
    ):
        """Test FAILED status is set when health check fails with non-200 response."""
        mock_run_bash.return_value = (mock_scontrol_output, "")
        mock_is_server_running.return_value = "RUNNING"
        mock_health_check.return_value = (ModelStatus.FAILED, "500")

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.FAILED
        assert status.failed_reason == "500"


@pytest.fixture
def mock_status_response():
    """Fixture returning a mock StatusResponse instance."""
    return StatusResponse(
        model_name="test-model",
        server_status=ModelStatus.UNAVAILABLE,
        job_state="RUNNING",
        raw_output="mock scontrol output",
        base_url="http://gpu01:8000/v1",
        pending_reason=None,
        failed_reason=None,
    )


class TestPerformanceMetricsCollector:
    """Unit tests for the PerformanceMetricsCollector class."""

    @pytest.fixture
    def mock_metrics_text(self):
        """Fixture returning a sample metrics text used for parsing tests."""
        return """\
vllm:prompt_tokens_total{engine="0",model_name="test-model"} 5000.0
vllm:generation_tokens_total{engine="0",model_name="test-model"} 1000.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="test-model"} 6.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="test-model"} 2.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="test-model"} 1.5
vllm:request_success_total{engine="0",finished_reason="length",model_name="test-model"} 2.0
vllm:num_requests_running{engine="0",model_name="test-model"} 1.0
vllm:num_requests_waiting{engine="0",model_name="test-model"} 0.0
vllm:gpu_cache_usage_perc{engine="0",model_name="test-model"} 0.123
"""

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    def test_init(self, mock_status_monitor, mock_status_response):
        """Test __init__ sets default state and initializes from model status."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )

        collector = PerformanceMetricsCollector(12345)

        assert collector.slurm_job_id == 12345
        assert collector._prev_prompt_tokens == 0.0
        assert collector._prev_generation_tokens == 0.0
        assert collector._last_updated is None
        assert collector.enabled_prefix_caching is False

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.utils.read_slurm_log")
    def test_init_check_prefix_caching_enabled(
        self, mock_read_slurm_log, mock_status_monitor, mock_status_response
    ):
        """Test __init__ when enable prefix caching is True."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )
        mock_read_slurm_log.return_value = {"enable_prefix_caching": True}

        collector = PerformanceMetricsCollector(12345)

        assert collector.slurm_job_id == 12345
        assert collector._prev_prompt_tokens == 0.0
        assert collector._prev_generation_tokens == 0.0
        assert collector._last_updated is None
        assert collector.enabled_prefix_caching is True

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.utils.get_base_url")
    def test_build_metrics_url_pending(
        self, mock_get_base_url, mock_status_monitor, mock_status_response
    ):
        """Test metrics URL returns placeholder when job is pending."""
        mock_status_response.job_state = ModelStatus.PENDING
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )

        collector = PerformanceMetricsCollector(12345)

        assert collector.metrics_url == "Pending resources for server initialization"

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.utils.get_base_url")
    def test_build_metrics_url_running(
        self, mock_get_base_url, mock_status_monitor, mock_status_response
    ):
        """Test metrics URL is built correctly when job is running."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )
        mock_get_base_url.return_value = "http://gpu01:8000/v1"

        collector = PerformanceMetricsCollector(12345)

        assert collector.metrics_url == "http://gpu01:8000/metrics"

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    def test_parse_metrics(
        self, mock_status_monitor, mock_metrics_text, mock_status_response
    ):
        """Test parsing of raw metrics text into structured dictionary values."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )

        collector = PerformanceMetricsCollector(12345)

        parsed = collector._parse_metrics(mock_metrics_text)

        assert parsed["total_prompt_tokens"] == 5000.0
        assert parsed["total_generation_tokens"] == 1000.0
        assert parsed["request_latency_sum"] == 6.0
        assert parsed["request_latency_count"] == 2.0
        assert parsed["requests_running"] == 1.0
        assert parsed["requests_waiting"] == 0.0

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.requests.get")
    @patch("time.time")
    def test_fetch_metrics_second_call(
        self,
        mock_time,
        mock_requests_get,
        mock_status_monitor,
        mock_metrics_text,
        mock_status_response,
    ):
        """Test throughput calculation between two fetch_metrics calls."""
        mock_time.return_value = 1000.0
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )
        mock_response = MagicMock()
        mock_response.text = mock_metrics_text
        mock_requests_get.return_value = mock_response

        collector = PerformanceMetricsCollector(12345)
        collector.metrics_url = "http://gpu01:8000/metrics"
        collector.fetch_metrics()

        mock_time.return_value = 1006.0
        updated_metrics = """\
vllm:prompt_tokens_total{engine="0",model_name="test-model"} 5048.0
vllm:generation_tokens_total{engine="0",model_name="test-model"} 1012.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="test-model"} 6.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="test-model"} 2.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="test-model"} 1.5
vllm:request_success_total{engine="0",finished_reason="length",model_name="test-model"} 2.0
vllm:num_requests_running{engine="0",model_name="test-model"} 1.0
vllm:num_requests_waiting{engine="0",model_name="test-model"} 0.0
vllm:gpu_cache_usage_perc{engine="0",model_name="test-model"} 0.123
"""
        mock_response.text = updated_metrics
        metrics = collector.fetch_metrics()

        assert metrics["prompt_tokens_per_sec"] == pytest.approx(
            (5048.0 - 5000.0) / 6.0
        )
        assert metrics["generation_tokens_per_sec"] == pytest.approx(
            (1012.0 - 1000.0) / 6.0
        )
        assert metrics["total_prompt_tokens"] == 5048.0
        assert metrics["total_generation_tokens"] == 1012.0
        assert metrics["request_latency_sum"] == 6.0
        assert metrics["request_latency_count"] == 2.0
        assert metrics["queue_time_sum"] == 1.5
        assert metrics["successful_requests_total"] == 2.0
        assert metrics["requests_running"] == 1.0
        assert metrics["requests_waiting"] == 0.0
        assert metrics["gpu_cache_usage"] == 0.123

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.requests.get")
    def test_fetch_metrics_request_exception(
        self, mock_requests_get, mock_status_monitor
    ):
        """Test that fetch_metrics handles request exceptions."""
        mock_status = MagicMock()
        mock_status_monitor.return_value.process_model_status.return_value = mock_status
        mock_requests_get.side_effect = requests.RequestException("Connection refused")

        collector = PerformanceMetricsCollector(12345)
        collector.metrics_url = "http://gpu01:8000/metrics"

        result = collector.fetch_metrics()

        assert isinstance(result, str)
        assert "Metrics request failed" in result
        assert "Connection refused" in result


class TestModelRegistry:
    """Unit tests for the ModelRegistry class."""

    @pytest.fixture
    def mock_configs(self):
        """Return a list of mock ModelConfig for testing."""
        return [
            ModelConfig(
                model_name="model1",
                model_family="family1",
                model_variant="variant1",
                model_type="LLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=32000,
                model_weights_parent_dir=Path("/path/to/models"),
            ),
            ModelConfig(
                model_name="model2",
                model_family="family2",
                model_variant="variant2",
                model_type="VLM",
                gpus_per_node=2,
                num_nodes=1,
                vocab_size=32000,
                model_weights_parent_dir=Path("/path/to/models"),
            ),
        ]

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init(self, mock_load_config, mock_configs):
        """Test ModelRegistry init."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()

        assert registry.model_configs[0].model_name == "model1"
        assert registry.model_configs[1].model_name == "model2"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_all_models(self, mock_load_config, mock_configs):
        """Test get_all_models returns correct model info."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()
        models = registry.get_all_models()

        assert len(models) == 2
        assert models[0].name == "model1"
        assert models[0].family == "family1"
        assert models[0].variant == "variant1"
        assert models[0].model_type == ModelType.LLM

        assert models[1].name == "model2"
        assert models[1].family == "family2"
        assert models[1].variant == "variant2"
        assert models[1].model_type == ModelType.VLM

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_single_model_config(self, mock_load_config, mock_configs):
        """Test retrieving the config for an existing model."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()
        config = registry.get_single_model_config("model1")

        assert config.model_name == "model1"
        assert config.model_family == "family1"
        assert config.model_variant == "variant1"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_single_model_config_not_found(self, mock_load_config, mock_configs):
        """Test retrieving a config for a non-existent model."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()

        with pytest.raises(ModelNotFoundError):
            registry.get_single_model_config("nonexistent_model")
