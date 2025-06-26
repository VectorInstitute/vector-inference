"""Utility functions shared between CLI and API.

This module provides utility functions for managing SLURM jobs, server status checks,
and configuration handling for the vector inference package.
"""

import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional, Union, cast

import requests
import yaml

from vec_inf.client._client_vars import MODEL_READY_SIGNATURE
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import ModelStatus
from vec_inf.client.slurm_vars import CACHED_CONFIG


def run_bash_command(command: str) -> tuple[str, str]:
    """Run a bash command and return the output.

    Parameters
    ----------
    command : str
        The bash command to execute

    Returns
    -------
    tuple[str, str]
        A tuple containing (stdout, stderr) from the command execution
    """
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return process.communicate()


def read_slurm_log(
    slurm_job_name: str,
    slurm_job_id: int,
    slurm_log_type: str,
    log_dir: Optional[Union[str, Path]],
) -> Union[list[str], str, dict[str, str]]:
    """Read the slurm log file.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : int
        ID of the SLURM job
    slurm_log_type : str
        Type of log file to read ('out', 'err', or 'json')
    log_dir : Optional[Union[str, Path]]
        Directory containing log files, if None uses default location

    Returns
    -------
    Union[list[str], str, dict[str, str]]
        Contents of the log file:
        - list[str] for 'out' and 'err' logs
        - dict[str, str] for 'json' logs
        - str for error messages if file not found
    """
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
                json_content: dict[str, str] = json.load(file)
                return json_content
        else:
            with file_path.open("r") as file:
                return file.readlines()
    except FileNotFoundError:
        return f"LOG FILE NOT FOUND: {file_path}"


def is_server_running(
    slurm_job_name: str, slurm_job_id: int, log_dir: Optional[str]
) -> Union[str, ModelStatus, tuple[ModelStatus, str]]:
    """Check if a model is ready to serve requests.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : int
        ID of the SLURM job
    log_dir : Optional[str]
        Directory containing log files

    Returns
    -------
    Union[str, ModelStatus, tuple[ModelStatus, str]]
        - str: Error message if logs cannot be read
        - ModelStatus: Current status of the server
        - tuple[ModelStatus, str]: Status and error message if server failed
    """
    log_content = read_slurm_log(slurm_job_name, slurm_job_id, "err", log_dir)
    if isinstance(log_content, str):
        return log_content

    status: Union[str, tuple[ModelStatus, str]] = ModelStatus.LAUNCHING

    for line in log_content:
        if "error" in line.lower():
            status = (ModelStatus.FAILED, line.strip("\n"))
        if MODEL_READY_SIGNATURE in line:
            status = "RUNNING"

    return status


def get_base_url(slurm_job_name: str, slurm_job_id: int, log_dir: Optional[str]) -> str:
    """Get the base URL of a model.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : int
        ID of the SLURM job
    log_dir : Optional[str]
        Directory containing log files

    Returns
    -------
    str
        Base URL of the model server or error message if not found
    """
    log_content = read_slurm_log(slurm_job_name, slurm_job_id, "json", log_dir)
    if isinstance(log_content, str):
        return log_content

    server_addr = cast(dict[str, str], log_content).get("server_address")
    return server_addr if server_addr else "URL NOT FOUND"


def model_health_check(
    slurm_job_name: str, slurm_job_id: int, log_dir: Optional[str]
) -> tuple[ModelStatus, Union[str, int]]:
    """Check the health of a running model on the cluster.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : int
        ID of the SLURM job
    log_dir : Optional[str]
        Directory containing log files

    Returns
    -------
    tuple[ModelStatus, Union[str, int]]
        Tuple containing:
        - ModelStatus: Current status of the model
        - Union[str, int]: Either HTTP status code or error message
    """
    base_url = get_base_url(slurm_job_name, slurm_job_id, log_dir)
    if not base_url.startswith("http"):
        return (ModelStatus.FAILED, base_url)
    health_check_url = base_url.replace("v1", "health")

    try:
        response = requests.get(health_check_url)
        # Check if the request was successful
        if response.status_code == 200:
            return (ModelStatus.READY, response.status_code)
        return (ModelStatus.FAILED, response.status_code)
    except requests.exceptions.RequestException as e:
        return (ModelStatus.FAILED, str(e))


def load_config(config_path: Optional[str] = None) -> list[ModelConfig]:
    """Load the model configuration.

    Loads configuration from default and user-specified paths, merging them
    if both exist. User configuration takes precedence over default values.

    Parameters
    ----------
    config_path : Optional[str]
        Path to the configuration file

    Returns
    -------
    list[ModelConfig]
        List of validated model configurations

    Notes
    -----
    Configuration is loaded from:
    1. User path: specified by config_path
    2. Default path: package's config/models.yaml or CACHED_CONFIG if it exists
    3. User path: specified by VEC_INF_CONFIG environment variable and merged with default config

    If user configuration exists, it will be merged with default configuration,
    with user values taking precedence for overlapping fields.
    """

    def load_yaml_config(path: Path) -> dict[str, Any]:
        """Helper to load YAML config with error handling."""
        try:
            with path.open() as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find config: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config at {path}: {e}")

    # 1. If config_path is given, use only that
    if config_path:
        config = load_yaml_config(Path(config_path))
        return [
            ModelConfig(model_name=name, **model_data)
            for name, model_data in config.get("models", {}).items()
        ]

    # 2. Otherwise, load default config
    default_path = (
        CACHED_CONFIG
        if CACHED_CONFIG.exists()
        else Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    )
    config = load_yaml_config(default_path)

    # 3. If user config exists, merge it
    user_path = os.getenv("VEC_INF_CONFIG")
    if user_path:
        user_path_obj = Path(user_path)
        if user_path_obj.exists():
            user_config = load_yaml_config(user_path_obj)
            for name, data in user_config.get("models", {}).items():
                if name in config.get("models", {}):
                    config["models"][name].update(data)
                else:
                    config.setdefault("models", {})[name] = data
        else:
            warnings.warn(
                f"WARNING: Could not find user config: {user_path}, revert to default config located at {default_path}",
                UserWarning,
                stacklevel=2,
            )

    return [
        ModelConfig(model_name=name, **model_data)
        for name, model_data in config.get("models", {}).items()
    ]


def parse_launch_output(output: str) -> tuple[str, dict[str, str]]:
    """Parse output from model launch command.

    Parameters
    ----------
    output : str
        Raw output from the launch command

    Returns
    -------
    tuple[str, dict[str, str]]
        Tuple containing:
        - str: SLURM job ID
        - dict[str, str]: Dictionary of parsed configuration parameters

    Notes
    -----
    Extracts the SLURM job ID and configuration parameters from the launch
    command output. Configuration parameters are parsed from key-value pairs
    in the output text.
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


def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two.

    Parameters
    ----------
    n : int
        The number to check
    """
    return n > 0 and (n & (n - 1)) == 0
