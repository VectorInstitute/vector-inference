"""Command line interface for Vector Inference."""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union, cast

import click
from rich.console import Console
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
            f"Model '{self.model_name}' not found in configuration and model weights "
            f"not found at expected path '{model_weights_path}'"
        )

    def _get_launch_params(self) -> dict[str, Any]:
        """Merge config defaults with CLI overrides."""
        params = self.model_config.model_dump(exclude={"model_name"})

        # Process boolean fields
        for bool_field in ["pipeline_parallelism", "enforce_eager"]:
            if (value := self.cli_kwargs.get(bool_field)) is not None:
                params[bool_field] = utils.convert_boolean_value(value)

        # Merge other overrides
        for key, value in self.cli_kwargs.items():
            if value is not None and key not in [
                "json_mode",
                "pipeline_parallelism",
                "enforce_eager",
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
        os.environ["VLLM_MAX_MODEL_LEN"] = self.params["max_model_len"]
        os.environ["VLLM_MAX_LOGPROBS"] = self.params["vocab_size"]
        os.environ["VLLM_DATA_TYPE"] = self.params["data_type"]
        os.environ["VLLM_MAX_NUM_SEQS"] = self.params["max_num_seqs"]
        os.environ["VLLM_TASK"] = VLLM_TASK_MAP[self.params["model_type"]]
        os.environ["PIPELINE_PARALLELISM"] = self.params["pipeline_parallelism"]
        os.environ["ENFORCE_EAGER"] = self.params["enforce_eager"]
        os.environ["SRC_DIR"] = SRC_DIR
        os.environ["VLLM_MODEL_WEIGHTS"] = str(
            Path(self.params["model_weights_parent_dir"], self.model_name)
        )
        os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
        os.environ["VENV_BASE"] = self.params["venv"]
        os.environ["LOG_DIR"] = self.params["log_dir"]

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
        table.add_row("Pipeline Parallelism", self.params["pipeline_parallelism"])
        table.add_row("Enforce Eager", self.params["enforce_eager"])
        table.add_row("Model Weights Directory", os.environ.get("VLLM_MODEL_WEIGHTS"))
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
        self.status_info = self.get_base_status_data()

    def get_base_status_data(self) -> dict[str, Union[str, None]]:
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
