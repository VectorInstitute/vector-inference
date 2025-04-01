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

# Boolean fields for model configuration
BOOLEAN_FIELDS = {
    "pipeline_parallelism",
    "enforce_eager",
    "enable_prefix_caching",
    "enable_chunked_prefill",
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