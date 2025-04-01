"""Helper class for the model launch."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.table import Table

from vec_inf.shared import utils
from vec_inf.shared.config import ModelConfig
from vec_inf.shared.utils import (
    BOOLEAN_FIELDS,
    LD_LIBRARY_PATH,
    REQUIRED_FIELDS,
    SRC_DIR,
    VLLM_TASK_MAP,
)


class LaunchHelper:
    """Shared launch helper for both CLI and API."""

    def __init__(
            self, model_name: str, cli_kwargs: Optional[dict[str, Any]]
        ):
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

        # If model config not found, check for path from keyword arguments or use fallback
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

    def launch(self) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        """Launch the model and return job information.

        Returns
        -------
        tuple[str, Dict[str, str], Dict[str, Any]]
            Slurm job ID, config dictionary, and parameters dictionary
        """
        # Set environment variables
        self.set_env_vars()

        # Build and execute the command
        command = self.build_launch_command()
        output, _ = utils.run_bash_command(command)

        # Parse the output
        job_id, config_dict = utils.parse_launch_output(output)

        # Save job configuration to JSON
        job_json_dir = Path(self.params["log_dir"], f"{self.model_name}.{job_id}")
        job_json_dir.mkdir(parents=True, exist_ok=True)

        job_json_path = job_json_dir / f"{self.model_name}.{job_id}.json"

        # Convert params for serialization
        serializable_params = {k: str(v) for k, v in self.params.items()}
        serializable_params["slurm_job_id"] = job_id

        with job_json_path.open("w") as f:
            json.dump(serializable_params, f, indent=4)

        return job_id, config_dict, self.params

    def format_table_output(self, job_id: str) -> Table:
        """Format output as rich Table."""
        table = utils.create_table(key_title="Job Config", value_title="Value")

        # Add key information with consistent styling
        table.add_row("Slurm Job ID", job_id, style="blue")
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

