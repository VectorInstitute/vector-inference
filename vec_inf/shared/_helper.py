"""Helper class for the model launch."""

import os
import time
from pathlib import Path
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse, urlunparse

import click
import requests

import vec_inf.shared._utils as utils
from vec_inf.shared._config import ModelConfig
from vec_inf.shared._models import ModelStatus
from vec_inf.shared._utils import (
    BOOLEAN_FIELDS,
    LD_LIBRARY_PATH,
    REQUIRED_FIELDS,
    SRC_DIR,
    VLLM_TASK_MAP,
)


class LaunchHelper:
    """Helper class for handling inference server launch."""

    def __init__(self, model_name: str, cli_kwargs: Optional[dict[str, Any]]):
        """Initialize the model launcher.

        Parameters
        ----------
        model_name: str
            Name of the model to launch
        cli_kwargs: Optional[dict[str, Any]]
            Optional launch keyword arguments to override default configuration
        """
        self.model_name = model_name
        self.cli_kwargs = cli_kwargs or {}
        self.model_config = self._get_model_configuration()
        self.params = self._get_launch_params()

    def _get_model_configuration(self) -> ModelConfig:
        """Load and validate model configuration."""
        model_configs = utils.load_config()
        config = next(
            (m for m in model_configs if m.model_name == self.model_name), None
        )

        if config:
            return config

        # If model config not found, check for path from CLI kwargs or use fallback
        model_weights_parent_dir = self.cli_kwargs.get(
            "model_weights_parent_dir",
            model_configs[0].model_weights_parent_dir if model_configs else None,
        )

        if not model_weights_parent_dir:
            raise ValueError(
                f"Could not determine model_weights_parent_dir and '{self.model_name}' not found in configuration"
            )

        model_weights_path = Path(model_weights_parent_dir, self.model_name)

        # Only give a warning if weights exist but config missing
        if model_weights_path.exists():
            click.echo(
                click.style(
                    f"Warning: '{self.model_name}' configuration not found in config, please ensure model configuration are properly set in command arguments",
                    fg="yellow",
                )
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

        raise click.ClickException(
            f"'{self.model_name}' not found in configuration and model weights "
            f"not found at expected path '{model_weights_path}'"
        )

    def _get_launch_params(self) -> dict[str, Any]:
        """Merge config defaults with CLI overrides."""
        params = self.model_config.model_dump()

        # Process boolean fields
        for bool_field in BOOLEAN_FIELDS:
            if self.cli_kwargs[bool_field]:
                params[bool_field] = True

        # Merge other overrides
        for key, value in self.cli_kwargs.items():
            if value is not None and key not in [
                "json_mode",
                *BOOLEAN_FIELDS,
            ]:
                params[key] = value

        # Validate required fields
        if not REQUIRED_FIELDS.issubset(set(params.keys())):
            raise click.ClickException(
                f"Missing required fields: {REQUIRED_FIELDS - set(params.keys())}"
            )

        # Create log directory
        params["log_dir"] = Path(params["log_dir"], params["model_family"]).expanduser()
        params["log_dir"].mkdir(parents=True, exist_ok=True)

        # Convert to string for JSON serialization
        for field in params:
            params[field] = str(params[field])

        return params

    def set_env_vars(self) -> None:
        """Set environment variables for the launch command."""
        os.environ["MODEL_NAME"] = self.model_name
        os.environ["MAX_MODEL_LEN"] = self.params["max_model_len"]
        os.environ["MAX_LOGPROBS"] = self.params["vocab_size"]
        os.environ["DATA_TYPE"] = self.params["data_type"]
        os.environ["MAX_NUM_SEQS"] = self.params["max_num_seqs"]
        os.environ["GPU_MEMORY_UTILIZATION"] = self.params["gpu_memory_utilization"]
        os.environ["TASK"] = VLLM_TASK_MAP[self.params["model_type"]]
        os.environ["PIPELINE_PARALLELISM"] = self.params["pipeline_parallelism"]
        os.environ["COMPILATION_CONFIG"] = self.params["compilation_config"]
        os.environ["SRC_DIR"] = SRC_DIR
        os.environ["MODEL_WEIGHTS"] = str(
            Path(self.params["model_weights_parent_dir"], self.model_name)
        )
        os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
        os.environ["VENV_BASE"] = self.params["venv"]
        os.environ["LOG_DIR"] = self.params["log_dir"]

        if self.params.get("enable_prefix_caching"):
            os.environ["ENABLE_PREFIX_CACHING"] = self.params["enable_prefix_caching"]
        if self.params.get("enable_chunked_prefill"):
            os.environ["ENABLE_CHUNKED_PREFILL"] = self.params["enable_chunked_prefill"]
        if self.params.get("max_num_batched_tokens"):
            os.environ["MAX_NUM_BATCHED_TOKENS"] = self.params["max_num_batched_tokens"]
        if self.params.get("enforce_eager"):
            os.environ["ENFORCE_EAGER"] = self.params["enforce_eager"]

    def build_launch_command(self) -> str:
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
        slurm_script = "vllm.slurm"
        if int(self.params["num_nodes"]) > 1:
            slurm_script = "multinode_vllm.slurm"
        command_list.append(f"{SRC_DIR}/{slurm_script}")
        return " ".join(command_list)


class StatusHelper:
    """Helper class for handling server status information."""

    def __init__(self, slurm_job_id: int, output: str, log_dir: Optional[str] = None):
        self.slurm_job_id = slurm_job_id
        self.output = output
        self.log_dir = log_dir
        self.status_info = self._get_base_status_data()

    def _get_base_status_data(self) -> dict[str, Union[str, None]]:
        """Extract basic job status information from scontrol output."""
        try:
            job_name = self.output.split(" ")[1].split("=")[1]
            job_state = self.output.split(" ")[9].split("=")[1]
        except IndexError:
            job_name = ModelStatus.UNAVAILABLE
            job_state = ModelStatus.UNAVAILABLE

        return {
            "model_name": job_name,
            "status": ModelStatus.UNAVAILABLE,
            "base_url": ModelStatus.UNAVAILABLE,
            "state": job_state,
            "pending_reason": None,
            "failed_reason": None,
        }

    def process_job_state(self) -> None:
        """Process different job states and update status information."""
        if self.status_info["state"] == ModelStatus.PENDING:
            self.process_pending_state()
        elif self.status_info["state"] == "RUNNING":
            self.process_running_state()

    def check_model_health(self) -> None:
        """Check model health and update status accordingly."""
        status, status_code = utils.model_health_check(
            cast(str, self.status_info["model_name"]), self.slurm_job_id, self.log_dir
        )
        if status == ModelStatus.READY:
            self.status_info["base_url"] = utils.get_base_url(
                cast(str, self.status_info["model_name"]),
                self.slurm_job_id,
                self.log_dir,
            )
            self.status_info["status"] = status
        else:
            self.status_info["status"], self.status_info["failed_reason"] = (
                status,
                cast(str, status_code),
            )

    def process_running_state(self) -> None:
        """Process RUNNING job state and check server status."""
        server_status = utils.is_server_running(
            cast(str, self.status_info["model_name"]), self.slurm_job_id, self.log_dir
        )

        if isinstance(server_status, tuple):
            self.status_info["status"], self.status_info["failed_reason"] = (
                server_status
            )
            return

        if server_status == "RUNNING":
            self.check_model_health()
        else:
            self.status_info["status"] = server_status

    def process_pending_state(self) -> None:
        """Process PENDING job state."""
        try:
            self.status_info["pending_reason"] = self.output.split(" ")[10].split("=")[
                1
            ]
            self.status_info["status"] = ModelStatus.PENDING
        except IndexError:
            self.status_info["pending_reason"] = "Unknown pending reason"


class MetricsHelper:
    """Helper class for handling metrics information."""

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

    def _get_status_info(self) -> dict[str, Union[str, None]]:
        """Retrieve status info using existing StatusHelper."""
        status_cmd = f"scontrol show job {self.slurm_job_id} --oneliner"
        output, stderr = utils.run_bash_command(status_cmd)
        if stderr:
            raise click.ClickException(f"Error: {stderr}")
        status_helper = StatusHelper(self.slurm_job_id, output, self.log_dir)
        return status_helper.status_info

    def _build_metrics_url(self) -> str:
        """Construct metrics endpoint URL from base URL with version stripping."""
        if self.status_info.get("state") == "PENDING":
            return "Pending resources for server initialization"

        base_url = utils.get_base_url(
            cast(str, self.status_info["model_name"]),
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
            cast(str, self.status_info["model_name"]),
            self.slurm_job_id,
            "json",
            self.log_dir,
        )
        if isinstance(job_json, str):
            return False
        return bool(cast(dict[str, str], job_json).get("enable_prefix_caching", False))

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


class ListHelper:
    """Helper class for handling model listing functionality."""

    def __init__(self, model_name: Optional[str] = None, json_mode: bool = False):
        self.model_name = model_name
        self.json_mode = json_mode
        self.model_configs = utils.load_config()

    def get_single_model_config(self) -> ModelConfig:
        """Get configuration for a specific model."""
        config = next(
            (c for c in self.model_configs if c.model_name == self.model_name), None
        )
        if not config:
            raise click.ClickException(
                f"Model '{self.model_name}' not found in configuration"
            )
        return config
