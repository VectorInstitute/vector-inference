"""Helper classes for the CLI."""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import vec_inf.client._utils as utils
from vec_inf.cli._models import MODEL_TYPE_COLORS, MODEL_TYPE_PRIORITY
from vec_inf.client._config import ModelConfig
from vec_inf.client._helper import (
    ModelLauncher,
    ModelRegistry,
    ModelStatusMonitor,
    PerformanceMetricsCollector,
)


class CLIModelLauncher(ModelLauncher):
    """CLI Helper class for handling inference server launch."""

    def __init__(self, model_name: str, kwargs: Optional[dict[str, Any]]):
        super().__init__(model_name, kwargs)

    def _warn(self, message: str) -> None:
        """Warn the user about a potential issue."""
        click.echo(click.style(f"Warning: {message}", fg="yellow"), err=True)

    def format_table_output(self) -> Table:
        """Format output as rich Table."""
        table = utils.create_table(key_title="Job Config", value_title="Value")

        # Add key information with consistent styling
        table.add_row("Slurm Job ID", self.slurm_job_id, style="blue")
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

        # Add path details
        table.add_row("Model Weights Directory", os.environ.get("MODEL_WEIGHTS"))
        table.add_row("Log Directory", self.params["log_dir"])

        return table


class CLIModelStatusMonitor(ModelStatusMonitor):
    """CLI Helper class for handling server status information and monitoring."""

    def __init__(self, slurm_job_id: int, log_dir: Optional[str] = None):
        super().__init__(slurm_job_id, log_dir)

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


class CLIMetricsCollector(PerformanceMetricsCollector):
    """CLI Helper class for streaming metrics information."""

    def __init__(self, slurm_job_id: int, log_dir: Optional[str] = None):
        super().__init__(slurm_job_id, log_dir)

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


class CLIModelRegistry(ModelRegistry):
    """CLI Helper class for handling model listing functionality."""

    def __init__(self, json_mode: bool = False):
        super().__init__()
        self.json_mode = json_mode

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
        sorted_configs = sorted(
            self.model_configs,
            key=lambda x: MODEL_TYPE_PRIORITY.get(x.model_type, 4),
        )

        # Create panels with color coding
        panels = []
        for config in sorted_configs:
            color = MODEL_TYPE_COLORS.get(config.model_type, "white")
            variant = config.model_variant or ""
            display_text = f"[magenta]{config.model_family}[/magenta]"
            if variant:
                display_text += f"-{variant}"
            panels.append(Panel(display_text, expand=True, border_style=color))

        return panels

    def process_list_command(
        self, console: Console, model_name: Optional[str] = None
    ) -> None:
        """Process the list command and display output."""
        try:
            if model_name:
                # Handle single model case
                config = self.get_single_model_config(model_name)
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
