"""Command line interface for Vector Inference."""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse, urlunparse

import click
import requests
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import vec_inf.cli._utils as utils
from vec_inf.cli._config import ModelConfig


VLLM_TASK_MAP = {
    "LLM": "generate",
    "VLM": "generate",
    "Text_Embedding": "embed",
    "Reward_Modeling": "reward",
}

REQUIRED_FIELDS = {
    "model_family",
    "model_type",
    "gpus_per_node",
    "num_nodes",
    "vocab_size",
    "max_model_len",
}

BOOLEAN_FIELDS = {
    "pipeline_parallelism",
    "enforce_eager",
    "enable_prefix_caching",
    "enable_chunked_prefill",
}

LD_LIBRARY_PATH = "/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"
SRC_DIR = str(Path(__file__).parent.parent)


class LaunchHelper:
    def __init__(
        self, model_name: str, cli_kwargs: dict[str, Optional[Union[str, int, bool]]]
    ):
        self.model_name = model_name
        self.cli_kwargs = cli_kwargs
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
        # If model config not found, load path from CLI args or fallback to default
        model_weights_parent_dir = self.cli_kwargs.get(
            "model_weights_parent_dir", model_configs[0].model_weights_parent_dir
        )
        model_weights_path = Path(cast(str, model_weights_parent_dir), self.model_name)
        # Only give a warning msg if weights exist but config missing
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
                model_weights_parent_dir=Path(cast(str, model_weights_parent_dir)),
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

    def format_table_output(self, job_id: str) -> Table:
        """Format output as rich Table."""
        table = utils.create_table(key_title="Job Config", value_title="Value")
        # Add rows
        table.add_row("Slurm Job ID", job_id, style="blue")
        table.add_row("Job Name", self.model_name)
        table.add_row("Model Type", self.params["model_type"])
        table.add_row("Partition", self.params["partition"])
        table.add_row("QoS", self.params["qos"])
        table.add_row("Time Limit", self.params["time"])
        table.add_row("Num Nodes", self.params["num_nodes"])
        table.add_row("GPUs/Node", self.params["gpus_per_node"])
        table.add_row("Data Type", self.params["data_type"])
        table.add_row("Vocabulary Size", self.params["vocab_size"])
        table.add_row("Max Model Length", self.params["max_model_len"])
        table.add_row("Max Num Seqs", self.params["max_num_seqs"])
        table.add_row("GPU Memory Utilization", self.params["gpu_memory_utilization"])
        table.add_row("Compilation Config", self.params["compilation_config"])
        table.add_row("Pipeline Parallelism", self.params["pipeline_parallelism"])
        if self.params.get("enable_prefix_caching"):
            table.add_row("Enable Prefix Caching", self.params["enable_prefix_caching"])
        if self.params.get("enable_chunked_prefill"):
            table.add_row(
                "Enable Chunked Prefill", self.params["enable_chunked_prefill"]
            )
        if self.params.get("max_num_batched_tokens"):
            table.add_row(
                "Max Num Batched Tokens", self.params["max_num_batched_tokens"]
            )
        if self.params.get("enforce_eager"):
            table.add_row("Enforce Eager", self.params["enforce_eager"])
        table.add_row("Model Weights Directory", os.environ.get("MODEL_WEIGHTS"))
        table.add_row("Log Directory", self.params["log_dir"])

        return table

    def post_launch_processing(self, output: str, console: Console) -> None:
        """Process and display launch output."""
        json_mode = bool(self.cli_kwargs.get("json_mode", False))
        slurm_job_id = output.split(" ")[-1].strip().strip("\n")
        self.params["slurm_job_id"] = slurm_job_id
        job_json = Path(
            self.params["log_dir"],
            f"{self.model_name}.{slurm_job_id}",
            f"{self.model_name}.{slurm_job_id}.json",
        )
        job_json.parent.mkdir(parents=True, exist_ok=True)
        job_json.touch(exist_ok=True)

        with job_json.open("w") as file:
            json.dump(self.params, file, indent=4)
        if json_mode:
            click.echo(self.params)
        else:
            table = self.format_table_output(slurm_job_id)
            console.print(table)


class StatusHelper:
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
            job_name = "UNAVAILABLE"
            job_state = "UNAVAILABLE"

        return {
            "model_name": job_name,
            "status": "UNAVAILABLE",
            "base_url": "UNAVAILABLE",
            "state": job_state,
            "pending_reason": None,
            "failed_reason": None,
        }

    def process_job_state(self) -> None:
        """Process different job states and update status information."""
        if self.status_info["state"] == "PENDING":
            self.process_pending_state()
        elif self.status_info["state"] == "RUNNING":
            self.process_running_state()

    def check_model_health(self) -> None:
        """Check model health and update status accordingly."""
        status, status_code = utils.model_health_check(
            cast(str, self.status_info["model_name"]), self.slurm_job_id, self.log_dir
        )
        if status == "READY":
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
            self.status_info["status"] = "PENDING"
        except IndexError:
            self.status_info["pending_reason"] = "Unknown pending reason"

    def output_json(self) -> None:
        """Format and output JSON data."""
        json_data = {
            "model_name": self.status_info["model_name"],
            "model_status": self.status_info["status"],
            "base_url": self.status_info["base_url"],
        }
        if self.status_info["pending_reason"]:
            json_data["pending_reason"] = self.status_info["pending_reason"]
        if self.status_info["failed_reason"]:
            json_data["failed_reason"] = self.status_info["failed_reason"]
        click.echo(json_data)

    def output_table(self, console: Console) -> None:
        """Create and display rich table."""
        table = utils.create_table(key_title="Job Status", value_title="Value")
        table.add_row("Model Name", self.status_info["model_name"])
        table.add_row("Model Status", self.status_info["status"], style="blue")

        if self.status_info["pending_reason"]:
            table.add_row("Pending Reason", self.status_info["pending_reason"])
        if self.status_info["failed_reason"]:
            table.add_row("Failed Reason", self.status_info["failed_reason"])

        table.add_row("Base URL", self.status_info["base_url"])
        console.print(table)


class MetricsHelper:
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

    def display_failed_metrics(self, table: Table, metrics: str) -> None:
        table.add_row("Server State", self.status_info["state"], style="yellow")
        table.add_row("Message", metrics)

    def display_metrics(self, table: Table, metrics: dict[str, float]) -> None:
        # Throughput metrics
        table.add_row(
            "Prompt Throughput",
            f"{metrics.get('prompt_tokens_per_sec', 0):.1f} tokens/s",
        )
        table.add_row(
            "Generation Throughput",
            f"{metrics.get('generation_tokens_per_sec', 0):.1f} tokens/s",
        )

        # Request queue metrics
        table.add_row(
            "Requests Running",
            f"{metrics.get('requests_running', 0):.0f} reqs",
        )
        table.add_row(
            "Requests Waiting",
            f"{metrics.get('requests_waiting', 0):.0f} reqs",
        )
        table.add_row(
            "Requests Swapped",
            f"{metrics.get('requests_swapped', 0):.0f} reqs",
        )

        # Cache usage metrics
        table.add_row(
            "GPU Cache Usage",
            f"{metrics.get('gpu_cache_usage', 0) * 100:.1f}%",
        )
        table.add_row(
            "CPU Cache Usage",
            f"{metrics.get('cpu_cache_usage', 0) * 100:.1f}%",
        )

        if self.enabled_prefix_caching:
            table.add_row(
                "GPU Prefix Cache Hit Rate",
                f"{metrics.get('gpu_prefix_cache_hit_rate', 0) * 100:.1f}%",
            )
            table.add_row(
                "CPU Prefix Cache Hit Rate",
                f"{metrics.get('cpu_prefix_cache_hit_rate', 0) * 100:.1f}%",
            )

        # Show average latency if available
        if "avg_request_latency" in metrics:
            table.add_row(
                "Avg Request Latency",
                f"{metrics['avg_request_latency']:.1f} s",
            )

        # Token counts
        table.add_row(
            "Total Prompt Tokens",
            f"{metrics.get('total_prompt_tokens', 0):.0f} tokens",
        )
        table.add_row(
            "Total Generation Tokens",
            f"{metrics.get('total_generation_tokens', 0):.0f} tokens",
        )
        table.add_row(
            "Successful Requests",
            f"{metrics.get('successful_requests_total', 0):.0f} reqs",
        )


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

    def format_single_model_output(
        self, config: ModelConfig
    ) -> Union[dict[str, Any], Table]:
        """Format output for a single model."""
        if self.json_mode:
            # Exclude non-essential fields from JSON output
            excluded = {"venv", "log_dir"}
            config_dict = config.model_dump(exclude=excluded)
            # Convert Path objects to strings
            config_dict["model_weights_parent_dir"] = str(
                config_dict["model_weights_parent_dir"]
            )
            return config_dict

        table = utils.create_table(key_title="Model Config", value_title="Value")
        for field, value in config.model_dump().items():
            if field not in {"venv", "log_dir"}:
                table.add_row(field, str(value))
        return table

    def format_all_models_output(self) -> Union[list[str], list[Panel]]:
        """Format output for all models."""
        if self.json_mode:
            return [config.model_name for config in self.model_configs]

        # Sort by model type priority
        type_priority = {"LLM": 0, "VLM": 1, "Text_Embedding": 2, "Reward_Modeling": 3}
        sorted_configs = sorted(
            self.model_configs, key=lambda x: type_priority.get(x.model_type, 4)
        )

        # Create panels with color coding
        model_type_colors = {
            "LLM": "cyan",
            "VLM": "bright_blue",
            "Text_Embedding": "purple",
            "Reward_Modeling": "bright_magenta",
        }

        panels = []
        for config in sorted_configs:
            color = model_type_colors.get(config.model_type, "white")
            variant = config.model_variant or ""
            display_text = f"[magenta]{config.model_family}[/magenta]"
            if variant:
                display_text += f"-{variant}"
            panels.append(Panel(display_text, expand=True, border_style=color))

        return panels

    def process_list_command(self, console: Console) -> None:
        """Process the list command and display output."""
        try:
            if self.model_name:
                # Handle single model case
                config = self.get_single_model_config()
                output = self.format_single_model_output(config)
                if self.json_mode:
                    click.echo(output)
                else:
                    console.print(output)
            # Handle all models case
            elif self.json_mode:
                # JSON output for all models is just a list of names
                model_names = [config.model_name for config in self.model_configs]
                click.echo(model_names)
            else:
                # Rich output for all models is a list of panels
                panels = self.format_all_models_output()
                if isinstance(panels, list):  # This helps mypy understand the type
                    console.print(Columns(panels, equal=True))
        except Exception as e:
            raise click.ClickException(str(e)) from e
