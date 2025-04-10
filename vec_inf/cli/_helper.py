"""Helper classes for the CLI."""

import os
from typing import Any, Union

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vec_inf.cli._models import MODEL_TYPE_COLORS, MODEL_TYPE_PRIORITY
from vec_inf.cli._utils import create_table
from vec_inf.client import ModelConfig, ModelInfo, StatusResponse


class LaunchResponseFormatter:
    """CLI Helper class for formatting LaunchResponse."""

    def __init__(self, model_name: str, params: dict[str, Any]):
        self.model_name = model_name
        self.params = params

    def format_table_output(self) -> Table:
        """Format output as rich Table."""
        table = create_table(key_title="Job Config", value_title="Value")

        # Add key information with consistent styling
        table.add_row("Slurm Job ID", self.params["slurm_job_id"], style="blue")
        table.add_row("Job Name", self.model_name)

        # Add model details
        table.add_row("Model Type", self.params["model_type"])

        # Add resource allocation details
        table.add_row("Partition", self.params["partition"])
        table.add_row("QoS", self.params["qos"])
        table.add_row("Time Limit", self.params["time"])
        table.add_row("Num Nodes", self.params["num_nodes"])
        table.add_row("GPUs/Node", self.params["gpus_per_node"])

        # Add model configuration details
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
        table.add_row(
            "Model Weights Directory",
            str(Path(self.params["model_weights_parent_dir"], self.model_name)),
        )
        table.add_row("Log Directory", self.params["log_dir"])

        return table


class StatusResponseFormatter:
    """CLI Helper class for formatting StatusResponse."""

    def __init__(self, status_info: StatusResponse):
        self.status_info = status_info

    def output_json(self) -> None:
        """Format and output JSON data."""
        json_data = {
            "model_name": self.status_info.model_name,
            "model_status": self.status_info.server_status,
            "base_url": self.status_info.base_url,
        }
        if self.status_info.pending_reason:
            json_data["pending_reason"] = self.status_info.pending_reason
        if self.status_info.failed_reason:
            json_data["failed_reason"] = self.status_info.failed_reason
        click.echo(json_data)

    def output_table(self) -> Table:
        """Create and display rich table."""
        table = create_table(key_title="Job Status", value_title="Value")
        table.add_row("Model Name", self.status_info.model_name)
        table.add_row("Model Status", self.status_info.server_status, style="blue")

        if self.status_info.pending_reason:
            table.add_row("Pending Reason", self.status_info.pending_reason)
        if self.status_info.failed_reason:
            table.add_row("Failed Reason", self.status_info.failed_reason)

        table.add_row("Base URL", self.status_info.base_url)
        return table


class MetricsResponseFormatter:
    """CLI Helper class for formatting MetricsResponse."""

    def __init__(self, metrics: Union[dict[str, float], str]):
        self.metrics = self._set_metrics(metrics)
        self.table = create_table("Metric", "Value")
        self.enabled_prefix_caching = self._check_prefix_caching()

    def _set_metrics(self, metrics: Union[dict[str, float], str]) -> dict[str, float]:
        """Set the metrics attribute."""
        return metrics if isinstance(metrics, dict) else {}

    def _check_prefix_caching(self) -> bool:
        """Check if prefix caching is enabled by looking for prefix cache metrics."""
        return self.metrics.get("gpu_prefix_cache_hit_rate") is not None

    def format_failed_metrics(self, message: str) -> None:
        self.table.add_row("ERROR", message)

    def format_metrics(self) -> None:
        # Throughput metrics
        self.table.add_row(
            "Prompt Throughput",
            f"{self.metrics.get('prompt_tokens_per_sec', 0):.1f} tokens/s",
        )
        self.table.add_row(
            "Generation Throughput",
            f"{self.metrics.get('generation_tokens_per_sec', 0):.1f} tokens/s",
        )

        # Request queue metrics
        self.table.add_row(
            "Requests Running",
            f"{self.metrics.get('requests_running', 0):.0f} reqs",
        )
        self.table.add_row(
            "Requests Waiting",
            f"{self.metrics.get('requests_waiting', 0):.0f} reqs",
        )
        self.table.add_row(
            "Requests Swapped",
            f"{self.metrics.get('requests_swapped', 0):.0f} reqs",
        )

        # Cache usage metrics
        self.table.add_row(
            "GPU Cache Usage",
            f"{self.metrics.get('gpu_cache_usage', 0) * 100:.1f}%",
        )
        self.table.add_row(
            "CPU Cache Usage",
            f"{self.metrics.get('cpu_cache_usage', 0) * 100:.1f}%",
        )

        if self.enabled_prefix_caching:
            self.table.add_row(
                "GPU Prefix Cache Hit Rate",
                f"{self.metrics.get('gpu_prefix_cache_hit_rate', 0) * 100:.1f}%",
            )
            self.table.add_row(
                "CPU Prefix Cache Hit Rate",
                f"{self.metrics.get('cpu_prefix_cache_hit_rate', 0) * 100:.1f}%",
            )

        # Show average latency if available
        if "avg_request_latency" in self.metrics:
            self.table.add_row(
                "Avg Request Latency",
                f"{self.metrics['avg_request_latency']:.1f} s",
            )

        # Token counts
        self.table.add_row(
            "Total Prompt Tokens",
            f"{self.metrics.get('total_prompt_tokens', 0):.0f} tokens",
        )
        self.table.add_row(
            "Total Generation Tokens",
            f"{self.metrics.get('total_generation_tokens', 0):.0f} tokens",
        )
        self.table.add_row(
            "Successful Requests",
            f"{self.metrics.get('successful_requests_total', 0):.0f} reqs",
        )


class ListCmdDisplay:
    """CLI Helper class for displaying model listing functionality."""

    def __init__(self, console: Console, json_mode: bool = False):
        self.console = console
        self.json_mode = json_mode
        self.model_config = None
        self.model_names: list[str] = []

    def _format_single_model_output(
        self, config: ModelConfig
    ) -> Union[dict[str, Any], Table]:
        """Format output table for a single model."""
        if self.json_mode:
            # Exclude non-essential fields from JSON output
            excluded = {"venv", "log_dir"}
            config_dict = config.model_dump(exclude=excluded)
            # Convert Path objects to strings
            config_dict["model_weights_parent_dir"] = str(
                config_dict["model_weights_parent_dir"]
            )
            return config_dict

        table = create_table(key_title="Model Config", value_title="Value")
        for field, value in config.model_dump().items():
            if field not in {"venv", "log_dir"}:
                table.add_row(field, str(value))
        return table

    def _format_all_models_output(
        self, model_infos: list[ModelInfo]
    ) -> Union[list[str], list[Panel]]:
        """Format output table for all models."""
        # Sort by model type priority
        sorted_model_infos = sorted(
            model_infos,
            key=lambda x: MODEL_TYPE_PRIORITY.get(x.type, 4),
        )

        # Create panels with color coding
        panels = []
        for model_info in sorted_model_infos:
            color = MODEL_TYPE_COLORS.get(model_info.type, "white")
            variant = model_info.variant or ""
            display_text = f"[magenta]{model_info.family}[/magenta]"
            if variant:
                display_text += f"-{variant}"
            panels.append(Panel(display_text, expand=True, border_style=color))

        return panels

    def display_single_model_output(self, config: ModelConfig) -> None:
        """Display the output for a single model."""
        output = self._format_single_model_output(config)
        if self.json_mode:
            click.echo(output)
        else:
            self.console.print(output)

    def display_all_models_output(self, model_infos: list[ModelInfo]) -> None:
        """Display the output for all models."""
        if self.json_mode:
            model_names = [info.name for info in model_infos]
            click.echo(model_names)
        else:
            panels = self._format_all_models_output(model_infos)
            self.console.print(Columns(panels, equal=True))
