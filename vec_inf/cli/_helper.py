"""Helper classes for the CLI.

This module provides formatting and display classes for the command-line interface,
handling the presentation of model information, status updates, and metrics.
"""

from pathlib import Path
from typing import Any, Union

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vec_inf.cli._utils import create_table
from vec_inf.cli._vars import MODEL_TYPE_COLORS, MODEL_TYPE_PRIORITY
from vec_inf.client import ModelConfig, ModelInfo, StatusResponse


class LaunchResponseFormatter:
    """CLI Helper class for formatting LaunchResponse.

    A formatter class that handles the presentation of model launch information
    in both table and JSON formats.

    Parameters
    ----------
    model_name : str
        Name of the launched model
    params : dict[str, Any]
        Launch parameters and configuration
    """

    def __init__(self, model_name: str, params: dict[str, Any]):
        self.model_name = model_name
        self.params = params

    def format_table_output(self) -> Table:
        """Format output as rich Table.

        Returns
        -------
        Table
            Rich table containing formatted launch information including:
            - Job configuration
            - Model details
            - Resource allocation
            - vLLM configuration
        """
        table = create_table(key_title="Job Config", value_title="Value")

        # Add key information with consistent styling
        table.add_row("Slurm Job ID", self.params["slurm_job_id"], style="blue")
        table.add_row("Job Name", self.model_name)

        # Add model details
        table.add_row("Model Type", self.params["model_type"])
        table.add_row("Vocabulary Size", self.params["vocab_size"])

        # Add resource allocation details
        table.add_row("Partition", self.params["partition"])
        table.add_row("QoS", self.params["qos"])
        table.add_row("Time Limit", self.params["time"])
        table.add_row("Num Nodes", self.params["num_nodes"])
        table.add_row("GPUs/Node", self.params["gpus_per_node"])
        table.add_row("CPUs/Task", self.params["cpus_per_task"])
        table.add_row("Memory/Node", self.params["mem_per_node"])

        # Add job config details
        table.add_row(
            "Model Weights Directory",
            str(Path(self.params["model_weights_parent_dir"], self.model_name)),
        )
        table.add_row("Log Directory", self.params["log_dir"])

        # Add vLLM configuration details
        table.add_row("vLLM Arguments:", style="magenta")
        for arg, value in self.params["vllm_args"].items():
            table.add_row(f"  {arg}:", str(value))

        return table


class StatusResponseFormatter:
    """CLI Helper class for formatting StatusResponse.

    A formatter class that handles the presentation of model status information
    in both table and JSON formats.

    Parameters
    ----------
    status_info : StatusResponse
        Status information to format
    """

    def __init__(self, status_info: StatusResponse):
        self.status_info = status_info

    def output_json(self) -> None:
        """Format and output JSON data.

        Outputs a JSON object containing:
        - model_name
        - model_status
        - base_url
        - pending_reason (if applicable)
        - failed_reason (if applicable)
        """
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
        """Create and display rich table.

        Returns
        -------
        Table
            Rich table containing formatted status information including:
            - Model name
            - Status
            - Base URL
            - Error information (if applicable)
        """
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
    """CLI Helper class for formatting MetricsResponse.

    A formatter class that handles the presentation of model metrics
    in a table format.

    Parameters
    ----------
    metrics : Union[dict[str, float], str]
        Dictionary of metrics or error message
    """

    def __init__(self, metrics: Union[dict[str, float], str]):
        self.metrics = self._set_metrics(metrics)
        self.table = create_table("Metric", "Value")
        self.enabled_prefix_caching = self._check_prefix_caching()

    def _set_metrics(self, metrics: Union[dict[str, float], str]) -> dict[str, float]:
        """Set the metrics attribute.

        Parameters
        ----------
        metrics : Union[dict[str, float], str]
            Raw metrics data

        Returns
        -------
        dict[str, float]
            Processed metrics dictionary
        """
        return metrics if isinstance(metrics, dict) else {}

    def _check_prefix_caching(self) -> bool:
        """Check if prefix caching is enabled.

        Returns
        -------
        bool
            True if prefix caching metrics are present
        """
        return self.metrics.get("gpu_prefix_cache_hit_rate") is not None

    def format_failed_metrics(self, message: str) -> None:
        """Format error message for failed metrics collection.

        Parameters
        ----------
        message : str
            Error message to display
        """
        self.table.add_row("ERROR", message)

    def format_metrics(self) -> None:
        """Format and display all available metrics.

        Formats and adds to the table:
        - Throughput metrics
        - Request queue metrics
        - Cache usage metrics
        - Prefix cache metrics (if enabled)
        - Latency metrics
        - Token counts
        """
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
    """CLI Helper class for displaying model listing functionality.

    A display class that handles the presentation of model listings
    in both table and JSON formats.

    Parameters
    ----------
    console : Console
        Rich console instance for output
    json_mode : bool, default=False
        Whether to output in JSON format
    """

    def __init__(self, console: Console, json_mode: bool = False):
        self.console = console
        self.json_mode = json_mode
        self.model_config = None
        self.model_names: list[str] = []

    def _format_single_model_output(
        self, config: ModelConfig
    ) -> Union[dict[str, Any], Table]:
        """Format output table for a single model.

        Parameters
        ----------
        config : ModelConfig
            Model configuration to format

        Returns
        -------
        Union[dict[str, Any], Table]
            Either a dictionary for JSON output or a Rich table
        """
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
            if field not in {"venv", "log_dir", "vllm_args"}:
                table.add_row(field, str(value))
            if field == "vllm_args":
                table.add_row("vLLM Arguments:", style="magenta")
                for vllm_arg, vllm_value in value.items():
                    table.add_row(f"  {vllm_arg}:", str(vllm_value))
        return table

    def _format_all_models_output(
        self, model_infos: list[ModelInfo]
    ) -> Union[list[str], list[Panel]]:
        """Format output table for all models.

        Parameters
        ----------
        model_infos : list[ModelInfo]
            List of model information to format

        Returns
        -------
        Union[list[str], list[Panel]]
            Either a list of model names or a list of formatted panels

        Notes
        -----
        Models are sorted by type priority and color-coded based on their type.
        """
        # Sort by model type priority
        sorted_model_infos = sorted(
            model_infos,
            key=lambda x: MODEL_TYPE_PRIORITY.get(x.model_type, 4),
        )

        # Create panels with color coding
        panels = []
        for model_info in sorted_model_infos:
            color = MODEL_TYPE_COLORS.get(model_info.model_type, "white")
            variant = model_info.variant or ""
            display_text = f"[magenta]{model_info.family}[/magenta]"
            if variant:
                display_text += f"-{variant}"
            panels.append(Panel(display_text, expand=True, border_style=color))

        return panels

    def display_single_model_output(self, config: ModelConfig) -> None:
        """Display the output for a single model.

        Parameters
        ----------
        config : ModelConfig
            Model configuration to display
        """
        output = self._format_single_model_output(config)
        if self.json_mode:
            click.echo(output)
        else:
            self.console.print(output)

    def display_all_models_output(self, model_infos: list[ModelInfo]) -> None:
        """Display the output for all models.

        Parameters
        ----------
        model_infos : list[ModelInfo]
            List of model information to display

        Notes
        -----
        Output format depends on json_mode:
        - JSON: List of model names
        - Table: Color-coded panels with model information
        """
        if self.json_mode:
            model_names = [info.name for info in model_infos]
            click.echo(model_names)
        else:
            panels = self._format_all_models_output(model_infos)
            self.console.print(Columns(panels, equal=True))
