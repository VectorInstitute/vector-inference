"""Command line interface for Vector Inference."""

import json
import os
from typing import Any, Optional, Union, cast

import click
from rich.console import Console
from rich.table import Table

import vec_inf.cli._utils as utils
from vec_inf.cli._config import ModelConfig


class LaunchHelper:
    def __init__(
        self, model_name: str, cli_kwargs: dict[str, Optional[Union[str, int, bool]]]
    ):
        self.model_name = model_name
        self.cli_kwargs = cli_kwargs
        self.model_config = self.get_model_configuration()

    def get_model_configuration(self) -> ModelConfig:
        """Load and validate model configuration."""
        model_configs = utils.load_config()
        if config := next(
            (m for m in model_configs if m.model_name == self.model_name), None
        ):
            return config
        raise click.ClickException(
            f"Model '{self.model_name}' not found in configuration"
        )

    def get_base_launch_command(self) -> str:
        """Construct base launch command."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "launch_server.sh",
        )
        return f"bash {script_path}"

    def process_configuration(self) -> dict[str, Any]:
        """Merge config defaults with CLI overrides."""
        params = self.model_config.model_dump(exclude={"model_name"})

        # Process boolean fields
        for bool_field in ["pipeline_parallelism", "enforce_eager"]:
            if (value := self.cli_kwargs.get(bool_field)) is not None:
                params[bool_field] = self.convert_boolean_value(value)

        # Merge other overrides
        for key, value in self.cli_kwargs.items():
            if value is not None and key not in [
                "json_mode",
                "pipeline_parallelism",
                "enforce_eager",
            ]:
                params[key] = value
        return params

    def convert_boolean_value(self, value: Union[str, int, bool]) -> str:
        """Convert various input types to boolean strings."""
        if isinstance(value, str):
            return "True" if value.lower() == "true" else "False"
        return "True" if bool(value) else "False"

    def build_launch_command(self, base_command: str, params: dict[str, Any]) -> str:
        """Construct the full launch command with parameters."""
        print(f"Building launch command with params: {params}")
        command = base_command
        for param_name, param_value in params.items():
            if param_value is None:
                continue

            formatted_value = param_value
            if isinstance(formatted_value, bool):
                formatted_value = "True" if formatted_value else "False"

            arg_name = param_name.replace("_", "-")
            command += f" --{arg_name} {formatted_value}"

        return command

    def parse_launch_output(self, output: str) -> tuple[str, list[str]]:
        """Extract job ID and output lines from command output."""
        slurm_job_id = output.split(" ")[-1].strip().strip("\n")
        output_lines = output.split("\n")[:-2]
        return slurm_job_id, output_lines

    def format_json_output(self, job_id: str, lines: list[str]) -> str:
        """Format output as JSON string with proper double quotes."""
        output_data = {"slurm_job_id": job_id}
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                output_data[key.lower().replace(" ", "_")] = value
        return json.dumps(output_data)

    def format_table_output(self, job_id: str, lines: list[str]) -> Table:
        """Format output as rich Table."""
        table = utils.create_table(key_title="Job Config", value_title="Value")
        table.add_row("Slurm Job ID", job_id, style="blue")
        for line in lines:
            key, value = line.split(": ")
            table.add_row(key, value)
        return table

    def handle_launch_output(self, output: str, console: Console) -> None:
        """Process and display launch output."""
        json_mode = bool(self.cli_kwargs.get("json_mode", False))
        slurm_job_id, output_lines = self.parse_launch_output(output)
        console.print(f"SLURM Job ID: {slurm_job_id}")
        console.print(f"Output lines: {output_lines}")

        if json_mode:
            output_data = self.format_json_output(slurm_job_id, output_lines)
            click.echo(output_data)
        else:
            table = self.format_table_output(slurm_job_id, output_lines)
            console.print(table)


class StatusHelper:
    def __init__(self, slurm_job_id: int, output: str, log_dir: Optional[str] = None):
        self.slurm_job_id = slurm_job_id
        self.output = output
        self.log_dir = log_dir
        self.status_info = self.get_base_status_data()

    def get_base_status_data(self) -> dict[str, Union[str, None]]:
        """Extract basic job status information from scontrol output."""
        try:
            print(f"get_base_status_data Output: {self.output}")
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
