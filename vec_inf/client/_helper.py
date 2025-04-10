"""Helper classes for the model."""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse, urlunparse

import requests

import vec_inf.client._utils as utils
from vec_inf.client._config import ModelConfig
from vec_inf.client._exceptions import (
    MissingRequiredFieldsError,
    ModelConfigurationError,
    ModelNotFoundError,
    SlurmJobError,
)
from vec_inf.client._models import (
    LaunchResponse,
    ModelInfo,
    ModelStatus,
    ModelType,
    StatusResponse,
)
from vec_inf.client._slurm_script_generator import SlurmScriptGenerator
from vec_inf.client._vars import (
    BOOLEAN_FIELDS,
    LD_LIBRARY_PATH,
    REQUIRED_FIELDS,
    SINGULARITY_IMAGE,
    SRC_DIR,
    VLLM_NCCL_SO_PATH,
)


class ModelLauncher:
    """Helper class for handling inference server launch."""

    def __init__(self, model_name: str, kwargs: Optional[dict[str, Any]]):
        """Initialize the model launcher.

        Parameters
        ----------
        model_name: str
            Name of the model to launch
        kwargs: Optional[dict[str, Any]]
            Optional launch keyword arguments to override default configuration
        """
        self.model_name = model_name
        self.kwargs = kwargs or {}
        self.slurm_job_id = ""
        self.slurm_script_path = Path("")
        self.model_config = self._get_model_configuration()
        self.params = self._get_launch_params()

    def _warn(self, message: str) -> None:
        """Warn the user about a potential issue."""
        warnings.warn(message, UserWarning, stacklevel=2)

    def _get_model_configuration(self) -> ModelConfig:
        """Load and validate model configuration."""
        model_configs = utils.load_config()
        config = next(
            (m for m in model_configs if m.model_name == self.model_name), None
        )

        if config:
            return config

        # If model config not found, check for path from CLI kwargs or use fallback
        model_weights_parent_dir = self.kwargs.get(
            "model_weights_parent_dir",
            model_configs[0].model_weights_parent_dir if model_configs else None,
        )

        if not model_weights_parent_dir:
            raise ModelNotFoundError(
                "Could not determine model weights parent directory"
            )

        model_weights_path = Path(model_weights_parent_dir, self.model_name)

        # Only give a warning if weights exist but config missing
        if model_weights_path.exists():
            self._warn(
                f"Warning: '{self.model_name}' configuration not found in config, please ensure model configuration are properly set in command arguments",
            )
            # Return a dummy model config object with model name and weights parent dir
            return ModelConfig(
                model_name=self.model_name,
                model_family="model_family_placeholder",
                model_type="LLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=1000,
                max_model_len=8192,
                model_weights_parent_dir=Path(str(model_weights_parent_dir)),
            )

        raise ModelConfigurationError(
            f"'{self.model_name}' not found in configuration and model weights "
            f"not found at expected path '{model_weights_path}'"
        )

    def _get_launch_params(self) -> dict[str, Any]:
        """Merge config defaults with CLI overrides."""
        params = self.model_config.model_dump()

        # Process boolean fields
        for bool_field in BOOLEAN_FIELDS:
            if self.kwargs.get(bool_field) and self.kwargs[bool_field]:
                params[bool_field] = True

        # Merge other overrides
        for key, value in self.kwargs.items():
            if value is not None and key not in [
                "json_mode",
                *BOOLEAN_FIELDS,
            ]:
                params[key] = value

        # Validate required fields
        if not REQUIRED_FIELDS.issubset(set(params.keys())):
            raise MissingRequiredFieldsError(
                f"Missing required fields: {REQUIRED_FIELDS - set(params.keys())}"
            )

        # Create log directory
        params["log_dir"] = Path(params["log_dir"], params["model_family"]).expanduser()
        params["log_dir"].mkdir(parents=True, exist_ok=True)

        # Convert to string for JSON serialization
        for field in params:
            params[field] = str(params[field])

        return params

    def _set_env_vars(self) -> None:
        """Set environment variables for the launch command."""
        os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
        os.environ["VLLM_NCCL_SO_PATH"] = VLLM_NCCL_SO_PATH
        os.environ["SINGULARITY_IMAGE"] = SINGULARITY_IMAGE

    def _build_launch_command(self) -> str:
        """Construct the full launch command with parameters."""
        # Base command
        command_list = ["sbatch"]
        # Append options
        command_list.extend(["--job-name", f"{self.model_name}"])
        command_list.extend(["--partition", f"{self.params['partition']}"])
        command_list.extend(["--qos", f"{self.params['qos']}"])
        command_list.extend(["--time", f"{self.params['time']}"])
        command_list.extend(["--nodes", f"{self.params['num_nodes']}"])
        command_list.extend(["--gpus-per-node", f"{self.params['gpus_per_node']}"])
        command_list.extend(
            [
                "--output",
                f"{self.params['log_dir']}/{self.model_name}.%j/{self.model_name}.%j.out",
            ]
        )
        command_list.extend(
            [
                "--error",
                f"{self.params['log_dir']}/{self.model_name}.%j/{self.model_name}.%j.err",
            ]
        )
        # Add slurm script
        self.slurm_script_path = SlurmScriptGenerator(
            self.params, SRC_DIR
        ).write_to_log_dir()
        command_list.append(str(self.slurm_script_path))
        return " ".join(command_list)

    def launch(self) -> LaunchResponse:
        """Launch the model."""
        # Set environment variables
        self._set_env_vars()

        # Build and execute the launch command
        command_output, stderr = utils.run_bash_command(self._build_launch_command())
        if stderr:
            raise SlurmJobError(f"Error: {stderr}")

        # Extract slurm job id from command output
        self.slurm_job_id = command_output.split(" ")[-1].strip().strip("\n")
        self.params["slurm_job_id"] = self.slurm_job_id

        # Create log directory and job json file, move slurm script to job log directory
        job_log_dir = Path(
            self.params["log_dir"], f"{self.model_name}.{self.slurm_job_id}"
        )
        job_log_dir.mkdir(parents=True, exist_ok=True)

        job_json = Path(
            job_log_dir,
            f"{self.model_name}.{self.slurm_job_id}.json",
        )
        job_json.touch(exist_ok=True)

        self.slurm_script_path.rename(
            job_log_dir / f"{self.model_name}.{self.slurm_job_id}.slurm"
        )

        with job_json.open("w") as file:
            json.dump(self.params, file, indent=4)

        return LaunchResponse(
            slurm_job_id=int(self.slurm_job_id),
            model_name=self.model_name,
            config=self.params,
            raw_output=command_output,
        )


class ModelStatusMonitor:
    """Class for handling server status information and monitoring."""

    def __init__(self, slurm_job_id: int, log_dir: Optional[str] = None):
        self.slurm_job_id = slurm_job_id
        self.output = self._get_raw_status_output()
        self.log_dir = log_dir
        self.status_info = self._get_base_status_data()

    def _get_raw_status_output(self) -> str:
        """Get the raw server status output from slurm."""
        status_cmd = f"scontrol show job {self.slurm_job_id} --oneliner"
        output, stderr = utils.run_bash_command(status_cmd)
        if stderr:
            raise SlurmJobError(f"Error: {stderr}")
        return output

    def _get_base_status_data(self) -> StatusResponse:
        """Extract basic job status information from scontrol output."""
        try:
            job_name = self.output.split(" ")[1].split("=")[1]
            job_state = self.output.split(" ")[9].split("=")[1]
        except IndexError:
            job_name = "UNAVAILABLE"
            job_state = ModelStatus.UNAVAILABLE

        return StatusResponse(
            model_name=job_name,
            server_status=ModelStatus.UNAVAILABLE,
            job_state=job_state,
            raw_output=self.output,
            base_url="UNAVAILABLE",
            pending_reason=None,
            failed_reason=None,
        )

    def _check_model_health(self) -> None:
        """Check model health and update status accordingly."""
        status, status_code = utils.model_health_check(
            self.status_info.model_name, self.slurm_job_id, self.log_dir
        )
        if status == ModelStatus.READY:
            self.status_info.base_url = utils.get_base_url(
                self.status_info.model_name,
                self.slurm_job_id,
                self.log_dir,
            )
            self.status_info.server_status = status
        else:
            self.status_info.server_status = status
            self.status_info.failed_reason = cast(str, status_code)

    def _process_running_state(self) -> None:
        """Process RUNNING job state and check server status."""
        server_status = utils.is_server_running(
            self.status_info.model_name, self.slurm_job_id, self.log_dir
        )

        if isinstance(server_status, tuple):
            self.status_info.server_status, self.status_info.failed_reason = (
                server_status
            )
            return

        if server_status == "RUNNING":
            self._check_model_health()
        else:
            self.status_info.server_status = cast(ModelStatus, server_status)

    def _process_pending_state(self) -> None:
        """Process PENDING job state."""
        try:
            self.status_info.pending_reason = self.output.split(" ")[10].split("=")[1]
            self.status_info.server_status = ModelStatus.PENDING
        except IndexError:
            self.status_info.pending_reason = "Unknown pending reason"

    def process_model_status(self) -> StatusResponse:
        """Process different job states and update status information."""
        if self.status_info.job_state == ModelStatus.PENDING:
            self._process_pending_state()
        elif self.status_info.job_state == "RUNNING":
            self._process_running_state()

        return self.status_info


class PerformanceMetricsCollector:
    """Class for handling metrics collection and processing."""

    def __init__(self, slurm_job_id: int, log_dir: Optional[str] = None):
        self.slurm_job_id = slurm_job_id
        self.log_dir = log_dir
        self.status_info = self._get_status_info()
        self.metrics_url = self._build_metrics_url()
        self.enabled_prefix_caching = self._check_prefix_caching()

        self._prev_prompt_tokens: float = 0.0
        self._prev_generation_tokens: float = 0.0
        self._last_updated: Optional[float] = None
        self._last_throughputs = {"prompt": 0.0, "generation": 0.0}

    def _get_status_info(self) -> StatusResponse:
        """Retrieve status info using existing StatusHelper."""
        status_helper = ModelStatusMonitor(self.slurm_job_id, self.log_dir)
        return status_helper.process_model_status()

    def _build_metrics_url(self) -> str:
        """Construct metrics endpoint URL from base URL with version stripping."""
        if self.status_info.job_state == ModelStatus.PENDING:
            return "Pending resources for server initialization"

        base_url = utils.get_base_url(
            self.status_info.model_name,
            self.slurm_job_id,
            self.log_dir,
        )
        if not base_url.startswith("http"):
            return "Server not ready"

        parsed = urlparse(base_url)
        clean_path = parsed.path.replace("/v1", "", 1).rstrip("/")
        return urlunparse(
            (parsed.scheme, parsed.netloc, f"{clean_path}/metrics", "", "", "")
        )

    def _check_prefix_caching(self) -> bool:
        """Check if prefix caching is enabled."""
        job_json = utils.read_slurm_log(
            self.status_info.model_name,
            self.slurm_job_id,
            "json",
            self.log_dir,
        )
        if isinstance(job_json, str):
            return False
        return bool(cast(dict[str, str], job_json).get("enable_prefix_caching", False))

    def _parse_metrics(self, metrics_text: str) -> dict[str, float]:
        """Parse metrics with latency count and sum."""
        key_metrics = {
            "vllm:prompt_tokens_total": "total_prompt_tokens",
            "vllm:generation_tokens_total": "total_generation_tokens",
            "vllm:e2e_request_latency_seconds_sum": "request_latency_sum",
            "vllm:e2e_request_latency_seconds_count": "request_latency_count",
            "vllm:request_queue_time_seconds_sum": "queue_time_sum",
            "vllm:request_success_total": "successful_requests_total",
            "vllm:num_requests_running": "requests_running",
            "vllm:num_requests_waiting": "requests_waiting",
            "vllm:num_requests_swapped": "requests_swapped",
            "vllm:gpu_cache_usage_perc": "gpu_cache_usage",
            "vllm:cpu_cache_usage_perc": "cpu_cache_usage",
        }

        if self.enabled_prefix_caching:
            key_metrics["vllm:gpu_prefix_cache_hit_rate"] = "gpu_prefix_cache_hit_rate"
            key_metrics["vllm:cpu_prefix_cache_hit_rate"] = "cpu_prefix_cache_hit_rate"

        parsed: dict[str, float] = {}
        for line in metrics_text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            metric_name = parts[0].split("{")[0]
            if metric_name in key_metrics:
                try:
                    parsed[key_metrics[metric_name]] = float(parts[1])
                except (ValueError, IndexError):
                    continue
        return parsed

    def fetch_metrics(self) -> Union[dict[str, float], str]:
        """Fetch metrics from the endpoint."""
        try:
            response = requests.get(self.metrics_url, timeout=3)
            response.raise_for_status()
            current_metrics = self._parse_metrics(response.text)
            current_time = time.time()

            # Set defaults using last known throughputs
            current_metrics.setdefault(
                "prompt_tokens_per_sec", self._last_throughputs["prompt"]
            )
            current_metrics.setdefault(
                "generation_tokens_per_sec", self._last_throughputs["generation"]
            )

            if self._last_updated is None:
                self._prev_prompt_tokens = current_metrics.get(
                    "total_prompt_tokens", 0.0
                )
                self._prev_generation_tokens = current_metrics.get(
                    "total_generation_tokens", 0.0
                )
                self._last_updated = current_time
                return current_metrics

            time_diff = current_time - self._last_updated
            if time_diff > 0:
                current_prompt = current_metrics.get("total_prompt_tokens", 0.0)
                current_gen = current_metrics.get("total_generation_tokens", 0.0)

                delta_prompt = current_prompt - self._prev_prompt_tokens
                delta_gen = current_gen - self._prev_generation_tokens

                # Only update throughputs when we have new tokens
                prompt_tps = (
                    delta_prompt / time_diff
                    if delta_prompt > 0
                    else self._last_throughputs["prompt"]
                )
                gen_tps = (
                    delta_gen / time_diff
                    if delta_gen > 0
                    else self._last_throughputs["generation"]
                )

                current_metrics["prompt_tokens_per_sec"] = prompt_tps
                current_metrics["generation_tokens_per_sec"] = gen_tps

                # Persist calculated values regardless of activity
                self._last_throughputs["prompt"] = prompt_tps
                self._last_throughputs["generation"] = gen_tps

                # Update tracking state
                self._prev_prompt_tokens = current_prompt
                self._prev_generation_tokens = current_gen
                self._last_updated = current_time

            # Calculate average latency if data is available
            if (
                "request_latency_sum" in current_metrics
                and "request_latency_count" in current_metrics
            ):
                latency_sum = current_metrics["request_latency_sum"]
                latency_count = current_metrics["request_latency_count"]
                current_metrics["avg_request_latency"] = (
                    latency_sum / latency_count if latency_count > 0 else 0.0
                )

            return current_metrics

        except requests.RequestException as e:
            return f"Metrics request failed, `metrics` endpoint might not be ready yet: {str(e)}"


class ModelRegistry:
    """Class for handling model listing and configuration management."""

    def __init__(self) -> None:
        """Initialize the model lister."""
        self.model_configs = utils.load_config()

    def get_all_models(self) -> list[ModelInfo]:
        """Get all available models."""
        available_models = []
        for config in self.model_configs:
            info = ModelInfo(
                name=config.model_name,
                family=config.model_family,
                variant=config.model_variant,
                type=ModelType(config.model_type),
                config=config.model_dump(exclude={"model_name", "venv", "log_dir"}),
            )
            available_models.append(info)
        return available_models

    def get_single_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        config = next(
            (c for c in self.model_configs if c.model_name == model_name), None
        )
        if not config:
            raise ModelNotFoundError(f"Model '{model_name}' not found in configuration")
        return config
