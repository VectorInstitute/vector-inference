"""Command line interface for Vector Inference."""

import os
from typing import Any, Dict, Optional, Union, cast

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vec_inf.shared.config import ModelConfig
from vec_inf.shared.models import ModelStatus
from vec_inf.shared.utils import (
    create_table,
    get_base_url,
    is_server_running,
    load_config,
    model_health_check,
)


# Required fields for model configuration
REQUIRED_FIELDS = {
    "model_family",
    "model_type",
    "gpus_per_node",
    "num_nodes",
    "vocab_size",
    "max_model_len",
}


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
        status, status_code = model_health_check(
            cast(str, self.status_info["model_name"]), self.slurm_job_id, self.log_dir
        )
        if status == ModelStatus.READY:
            self.status_info["base_url"] = get_base_url(
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
        server_status = is_server_running(
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
        table = create_table(key_title="Job Status", value_title="Value")
        table.add_row("Model Name", self.status_info["model_name"])
        table.add_row("Model Status", self.status_info["status"], style="blue")

        if self.status_info["pending_reason"]:
            table.add_row("Pending Reason", self.status_info["pending_reason"])
        if self.status_info["failed_reason"]:
            table.add_row("Failed Reason", self.status_info["failed_reason"])

        table.add_row("Base URL", self.status_info["base_url"])
        console.print(table)


class LaunchHelper:
    """Helper class for handling model launch output formatting."""

    def __init__(
        self,
        job_id: str,
        model_name: str,
        params: Dict[str, Any],
        json_mode: bool = False,
    ):
        """Initialize LaunchHelper with launch results.

        Parameters
        ----------
        job_id : str
            The Slurm job ID assigned to the launched model
        model_name : str
            The name of the launched model
        params : Dict[str, Any]
            Dictionary containing all model parameters
        json_mode : bool, optional
            Whether to output in JSON format, by default False
        """
        self.job_id = job_id
        self.model_name = model_name
        self.params = params
        self.json_mode = json_mode

    def output_json(self) -> None:
        """Format and output launch information as JSON."""
        # Convert params for JSON output
        serializable_params = {k: str(v) for k, v in self.params.items()}
        serializable_params["slurm_job_id"] = self.job_id
        click.echo(serializable_params)

    def output_table(self, console: Console) -> None:
        """Create and display a formatted table with launch information."""
        table = create_table(key_title="Job Config", value_title="Value")

        # Add key information with consistent styling
        table.add_row("Slurm Job ID", self.job_id, style="blue")
        table.add_row("Job Name", self.model_name)

        # Add model details
        table.add_row("Model Type", str(self.params["model_type"]))

        # Add resource allocation details
        table.add_row("Partition", str(self.params["partition"]))
        table.add_row("QoS", str(self.params["qos"]))
        table.add_row("Time Limit", str(self.params["time"]))
        table.add_row("Num Nodes", str(self.params["num_nodes"]))
        table.add_row("GPUs/Node", str(self.params["gpus_per_node"]))

        # Add model configuration details
        table.add_row("Data Type", str(self.params["data_type"]))
        table.add_row("Vocabulary Size", str(self.params["vocab_size"]))
        table.add_row("Max Model Length", str(self.params["max_model_len"]))
        table.add_row("Max Num Seqs", str(self.params["max_num_seqs"]))
        table.add_row(
            "GPU Memory Utilization", str(self.params["gpu_memory_utilization"])
        )
        table.add_row("Pipeline Parallelism", str(self.params["pipeline_parallelism"]))
        table.add_row("Enforce Eager", str(self.params["enforce_eager"]))

        # Add path details
        table.add_row("Model Weights Directory", os.environ.get("MODEL_WEIGHTS", ""))
        table.add_row("Log Directory", str(self.params["log_dir"]))

        console.print(table)


class ListHelper:
    """Helper class for handling model listing functionality."""

    def __init__(self, model_name: Optional[str] = None, json_mode: bool = False):
        self.model_name = model_name
        self.json_mode = json_mode
        self.model_configs = load_config()

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

        table = create_table(key_title="Model Config", value_title="Value")
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
