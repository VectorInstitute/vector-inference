"""Utility functions for the CLI."""

import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Union, cast

import requests
import yaml
from rich.table import Table

from vec_inf.cli._config import ModelConfig


MODEL_READY_SIGNATURE = "INFO:     Application startup complete."
SERVER_ADDRESS_SIGNATURE = "Server address: "
CACHED_CONFIG = os.path.join("/", "model-weights", "vec-inf-config.yaml")


def run_bash_command(command: str) -> str:
    """Run a bash command and return the output."""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr}")
    return stdout


def read_slurm_log(
    slurm_job_name: str, slurm_job_id: int, slurm_log_type: str, log_dir: Optional[str]
) -> Union[list[str], str]:
    """Read the slurm log file."""
    if not log_dir:
        models_dir = os.path.join(os.path.expanduser("~"), ".vec-inf-logs")

        for directory in sorted(os.listdir(models_dir), key=len, reverse=True):
            if directory in slurm_job_name:
                log_dir = os.path.join(models_dir, directory)
                break

    log_dir = cast(str, log_dir)

    try:
        file_path = os.path.join(
            log_dir,
            f"{slurm_job_name}.{slurm_job_id}.{slurm_log_type}",
        )
        with open(file_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Could not find file: {file_path}")
        return "LOG_FILE_NOT_FOUND"
    return lines


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
    log_content = read_slurm_log(slurm_job_name, slurm_job_id, "out", log_dir)
    if isinstance(log_content, str):
        return log_content

    for line in log_content:
        if SERVER_ADDRESS_SIGNATURE in line:
            return line.split(SERVER_ADDRESS_SIGNATURE)[1].strip("\n")
    return "URL_NOT_FOUND"


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
        if os.path.exists(CACHED_CONFIG)
        else os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "config",
            "models.yaml",
        )
    )

    config: dict[str, Any] = {}
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


def get_latest_metric(log_lines: list[str]) -> Union[str, dict[str, str]]:
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
