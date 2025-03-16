"""Utility functions shared between CLI and API."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import requests
import yaml
from rich.table import Table

from vec_inf.shared.config import ModelConfig


MODEL_READY_SIGNATURE = "INFO:     Application startup complete."
CACHED_CONFIG = Path("/", "model-weights", "vec-inf-shared", "models.yaml")
SRC_DIR = str(Path(__file__).parent.parent)
LD_LIBRARY_PATH = "/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"

# Maps model types to vLLM tasks
VLLM_TASK_MAP = {
    "LLM": "generate",
    "VLM": "generate",
    "TEXT_EMBEDDING": "embed",
    "REWARD_MODELING": "reward",
}

# Required fields for model configuration
REQUIRED_FIELDS = {
    "model_family",
    "model_type",
    "gpus_per_node",
    "num_nodes",
    "vocab_size",
    "max_model_len",
}


def run_bash_command(command: str) -> tuple[str, str]:
    """Run a bash command and return the output."""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return process.communicate()


def read_slurm_log(
    slurm_job_name: str,
    slurm_job_id: int,
    slurm_log_type: str,
    log_dir: Optional[Union[str, Path]],
) -> Union[list[str], str, Dict[str, str]]:
    """Read the slurm log file."""
    if not log_dir:
        # Default log directory
        models_dir = Path.home() / ".vec-inf-logs"
        # Iterate over all dirs in models_dir, sorted by dir name length in desc order
        for directory in sorted(
            [d for d in models_dir.iterdir() if d.is_dir()],
            key=lambda d: len(d.name),
            reverse=True,
        ):
            if directory.name in slurm_job_name:
                log_dir = directory
                break
    else:
        log_dir = Path(log_dir)

    # If log_dir is still not set, then didn't find the log dir at default location
    if not log_dir:
        return "LOG DIR NOT FOUND"

    try:
        file_path = (
            log_dir
            / Path(f"{slurm_job_name}.{slurm_job_id}")
            / f"{slurm_job_name}.{slurm_job_id}.{slurm_log_type}"
        )
        if slurm_log_type == "json":
            with file_path.open("r") as file:
                json_content: Dict[str, str] = json.load(file)
                return json_content
        else:
            with file_path.open("r") as file:
                return file.readlines()
    except FileNotFoundError:
        return f"LOG FILE NOT FOUND: {file_path}"


def is_server_running(
    slurm_job_name: str, slurm_job_id: int, log_dir: Optional[str]
) -> Union[str, tuple[str, str]]:
    """Check if a model is ready to serve requests."""
    log_content = read_slurm_log(slurm_job_name, slurm_job_id, "err", log_dir)
    if isinstance(log_content, str):
        return log_content

    status: Union[str, tuple[str, str]] = "LAUNCHING"

    for line in log_content:
        if "error" in line.lower():
            status = ("FAILED", line.strip("\n"))
        if MODEL_READY_SIGNATURE in line:
            status = "RUNNING"

    return status


def get_base_url(slurm_job_name: str, slurm_job_id: int, log_dir: Optional[str]) -> str:
    """Get the base URL of a model."""
    log_content = read_slurm_log(slurm_job_name, slurm_job_id, "json", log_dir)
    if isinstance(log_content, str):
        return log_content

    server_addr = cast(Dict[str, str], log_content).get("server_address")
    return server_addr if server_addr else "URL NOT FOUND"


def model_health_check(
    slurm_job_name: str, slurm_job_id: int, log_dir: Optional[str]
) -> tuple[str, Union[str, int]]:
    """Check the health of a running model on the cluster."""
    base_url = get_base_url(slurm_job_name, slurm_job_id, log_dir)
    if not base_url.startswith("http"):
        return ("FAILED", base_url)
    health_check_url = base_url.replace("v1", "health")

    try:
        response = requests.get(health_check_url)
        # Check if the request was successful
        if response.status_code == 200:
            return ("READY", response.status_code)
        return ("FAILED", response.status_code)
    except requests.exceptions.RequestException as e:
        return ("FAILED", str(e))


def create_table(
    key_title: str = "", value_title: str = "", show_header: bool = True
) -> Table:
    """Create a table for displaying model status."""
    table = Table(show_header=show_header, header_style="bold magenta")
    table.add_column(key_title, style="dim")
    table.add_column(value_title)
    return table


def load_config() -> list[ModelConfig]:
    """Load the model configuration."""
    default_path = (
        CACHED_CONFIG
        if CACHED_CONFIG.exists()
        else Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    )

    config: Dict[str, Any] = {}
    with open(default_path) as f:
        config = yaml.safe_load(f) or {}

    user_path = os.getenv("VEC_INF_CONFIG")
    if user_path:
        user_path_obj = Path(user_path)
        if user_path_obj.exists():
            with open(user_path_obj) as f:
                user_config = yaml.safe_load(f) or {}
                for name, data in user_config.get("models", {}).items():
                    if name in config.get("models", {}):
                        config["models"][name].update(data)
                    else:
                        config.setdefault("models", {})[name] = data
        else:
            print(
                f"WARNING: Could not find user config: {user_path}, revert to default config located at {default_path}"
            )

    return [
        ModelConfig(model_name=name, **model_data)
        for name, model_data in config.get("models", {}).items()
    ]


def get_latest_metric(log_lines: list[str]) -> Union[str, Dict[str, str]]:
    """Read the latest metric entry from the log file."""
    latest_metric = {}

    try:
        for line in reversed(log_lines):
            if "Avg prompt throughput" in line:
                # Parse the metric values from the line
                metrics_str = line.split("] ")[1].strip().strip(".")
                metrics_list = metrics_str.split(", ")
                for metric in metrics_list:
                    key, value = metric.split(": ")
                    latest_metric[key] = value
                break
    except Exception as e:
        return f"[red]Error reading log file: {e}[/red]"

    return latest_metric


def convert_boolean_value(value: Union[str, int, bool]) -> bool:
    """Convert various input types to boolean strings."""
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def parse_launch_output(output: str) -> tuple[str, Dict[str, str]]:
    """Parse output from model launch command.

    Parameters
    ----------
    output: str
        Output from the launch command

    Returns
    -------
    tuple[str, Dict[str, str]]
        Slurm job ID and dictionary of config parameters

    """
    slurm_job_id = output.split(" ")[-1].strip().strip("\n")

    # Extract config parameters
    config_dict = {}
    output_lines = output.split("\n")[:-2]
    for line in output_lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            config_dict[key.lower().replace(" ", "_")] = value

    return slurm_job_id, config_dict


class ModelLauncher:
    """Shared model launcher for both CLI and API."""

    def __init__(self, model_name: str, options: Optional[Dict[str, Any]] = None):
        """Initialize the model launcher.

        Parameters
        ----------
        model_name: str
            Name of the model to launch
        options: Optional[Dict[str, Any]]
            Optional launch options to override default configuration
        """
        self.model_name = model_name
        self.options = options or {}
        self.model_config = self._get_model_configuration()
        self.params = self._get_launch_params()

    def _get_model_configuration(self) -> ModelConfig:
        """Load and validate model configuration."""
        model_configs = load_config()
        config = next(
            (m for m in model_configs if m.model_name == self.model_name), None
        )

        if config:
            return config

        # If model config not found, check for path from options or use fallback
        model_weights_parent_dir = self.options.get(
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
            print(
                f"Warning: '{self.model_name}' configuration not found in config, please ensure model configuration are properly set in options"
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

        raise ValueError(
            f"'{self.model_name}' not found in configuration and model weights "
            f"not found at expected path '{model_weights_path}'"
        )

    def _get_launch_params(self) -> dict[str, Any]:
        """Merge config defaults with overrides."""
        params = self.model_config.model_dump()

        # Process boolean fields
        for bool_field in ["pipeline_parallelism", "enforce_eager"]:
            if (value := self.options.get(bool_field)) is not None:
                params[bool_field] = convert_boolean_value(value)

        # Merge other overrides
        for key, value in self.options.items():
            if value is not None and key not in [
                "json_mode",
                "pipeline_parallelism",
                "enforce_eager",
            ]:
                params[key] = value

        # Validate required fields
        if not REQUIRED_FIELDS.issubset(set(params.keys())):
            missing_fields = REQUIRED_FIELDS - set(params.keys())
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Create log directory
        params["log_dir"] = Path(params["log_dir"], params["model_family"]).expanduser()
        params["log_dir"].mkdir(parents=True, exist_ok=True)

        return params

    def set_env_vars(self) -> None:
        """Set environment variables for the launch command."""
        os.environ["MODEL_NAME"] = self.model_name
        os.environ["MAX_MODEL_LEN"] = str(self.params["max_model_len"])
        os.environ["MAX_LOGPROBS"] = str(self.params["vocab_size"])
        os.environ["DATA_TYPE"] = str(self.params["data_type"])
        os.environ["MAX_NUM_SEQS"] = str(self.params["max_num_seqs"])
        os.environ["GPU_MEMORY_UTILIZATION"] = str(
            self.params["gpu_memory_utilization"]
        )
        os.environ["TASK"] = VLLM_TASK_MAP[self.params["model_type"]]
        os.environ["PIPELINE_PARALLELISM"] = str(self.params["pipeline_parallelism"])
        os.environ["ENFORCE_EAGER"] = str(self.params["enforce_eager"])
        os.environ["SRC_DIR"] = SRC_DIR
        os.environ["MODEL_WEIGHTS"] = str(
            Path(self.params["model_weights_parent_dir"], self.model_name)
        )
        os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
        os.environ["VENV_BASE"] = str(self.params["venv"])
        os.environ["LOG_DIR"] = str(self.params["log_dir"])

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
        output, _ = run_bash_command(command)

        # Parse the output
        job_id, config_dict = parse_launch_output(output)

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
