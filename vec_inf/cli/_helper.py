"""Command line interface for Vector Inference."""

import time
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse, urlunparse

import click
import requests
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vec_inf.shared import utils
from vec_inf.shared.config import ModelConfig
from vec_inf.shared.models import ModelStatus


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
        type_priority = {
            "LLM": 0,
            "VLM": 1,
            "Text_Embedding": 2,
            "Reward_Modeling": 3,
        }
        sorted_configs = sorted(
            self.model_configs,
            key=lambda x: type_priority.get(x.model_type, 4),
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
