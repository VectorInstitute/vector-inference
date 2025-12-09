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
from vec_inf.client._exceptions import MissingRequiredFieldsError
from vec_inf.client._slurm_vars import CACHED_MODEL_CONFIG_PATH, REQUIRED_ARGS
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import ModelStatus


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
    slurm_job_id: str,
    slurm_log_type: str,
    log_dir: str,
) -> Union[list[str], str, dict[str, str]]:
    """Read the slurm log file.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : str
        ID of the SLURM job
    slurm_log_type : str
        Type of log file to read ('out', 'err', or 'json')
    log_dir : str
        Directory containing log files

    Returns
    -------
    Union[list[str], str, dict[str, str]]
        Contents of the log file:
        - list[str] for 'out' and 'err' logs
        - dict[str, str] for 'json' logs
        - str for error messages if file not found
    """
    try:
        if "+" in slurm_job_id:
            main_job_id, het_job_id = slurm_job_id.split("+")
            slurm_job_id = str(int(main_job_id) + int(het_job_id))
        file_path = Path(log_dir, f"{slurm_job_name}.{slurm_job_id}.{slurm_log_type}")
        if slurm_log_type == "json":
            with file_path.open("r") as file:
                json_content: dict[str, str] = json.load(file)
                return json_content
        else:
            with file_path.open("r", errors="replace") as file:
                return file.readlines()
    except FileNotFoundError:
        return f"LOG FILE NOT FOUND: {file_path}"


def is_server_running(
    slurm_job_name: str, slurm_job_id: str, log_dir: str
) -> Union[str, ModelStatus, tuple[ModelStatus, str]]:
    """Check if a model is ready to serve requests.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : str
        ID of the SLURM job
    log_dir : str
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

    # Patterns that indicate fatal errors (not just warnings)
    fatal_error_patterns = [
        "traceback",
        "exception",
        "fatal error",
        "critical error",
        "failed to",
        "could not",
        "unable to",
        "error:",
    ]

    # Patterns to ignore (non-fatal warnings/info messages)
    ignore_patterns = [
        "deprecated",
        "futurewarning",
        "userwarning",
        "deprecationwarning",
        "slurmstepd: error:",  # SLURM cancellation messages (often after server started)
    ]

    ready_signature_found = False
    fatal_error_line = None

    for line in log_content:
        line_lower = line.lower()

        # Check for ready signature first - if found, server is running
        if MODEL_READY_SIGNATURE in line:
            ready_signature_found = True
            # Continue checking to see if there are errors after startup

        # Check for fatal errors (only if we haven't seen ready signature yet)
        if not ready_signature_found:
            # Skip lines that match ignore patterns
            if any(ignore_pattern in line_lower for ignore_pattern in ignore_patterns):
                continue

            # Check for fatal error patterns
            for pattern in fatal_error_patterns:
                if pattern in line_lower:
                    # Additional check: skip if it's part of a warning message
                    # (warnings often contain "error:" but aren't fatal)
                    if "warning" in line_lower and "error:" in line_lower:
                        continue
                    fatal_error_line = line.strip("\n")
                    break

    # If we found a fatal error, mark as failed
    if fatal_error_line:
        return (ModelStatus.FAILED, fatal_error_line)

    # If ready signature was found and no fatal errors, server is running
    if ready_signature_found:
        return "RUNNING"

    # Otherwise, still launching
    return ModelStatus.LAUNCHING


def get_base_url(slurm_job_name: str, slurm_job_id: str, log_dir: str) -> str:
    """Get the base URL of a model.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : str
        ID of the SLURM job
    log_dir : str
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
    slurm_job_name: str, slurm_job_id: str, log_dir: str
) -> tuple[ModelStatus, Union[str, int]]:
    """Check the health of a running model on the cluster.

    Parameters
    ----------
    slurm_job_name : str
        Name of the SLURM job
    slurm_job_id : str
        ID of the SLURM job
    log_dir : str
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
    2. Default path: package's config/models.yaml or CACHED_MODEL_CONFIG_PATH if exists
    3. Environment variable: specified by VEC_INF_CONFIG environment variable
        and merged with default config

    If user configuration exists, it will be merged with default configuration,
    with user values taking precedence for overlapping fields.
    """

    def load_yaml_config(path: Path) -> dict[str, Any]:
        """Load YAML config with error handling."""
        try:
            with path.open() as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Could not find config: {path}") from err
        except yaml.YAMLError as err:
            raise ValueError(f"Error parsing YAML config at {path}: {err}") from err

    def process_config(config: dict[str, Any]) -> list[ModelConfig]:
        """Process the config based on the config type."""
        return [
            ModelConfig(model_name=name, **model_data)
            for name, model_data in config.get("models", {}).items()
        ]

    def resolve_config_path_from_env_var() -> Path | None:
        """Resolve the config path from the environment variable."""
        config_dir = os.getenv("VEC_INF_CONFIG_DIR")
        config_path = os.getenv("VEC_INF_MODEL_CONFIG")
        if config_path:
            return Path(config_path)
        if config_dir:
            return Path(config_dir, "models.yaml")
        return None

    def update_config(
        config: dict[str, Any], user_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the config with the user config."""
        for name, data in user_config.get("models", {}).items():
            if name in config.get("models", {}):
                config["models"][name].update(data)
            else:
                config.setdefault("models", {})[name] = data

        return config

    # 1. If config_path is given, use only that
    if config_path:
        config = load_yaml_config(Path(config_path))
        return process_config(config)

    # 2. Otherwise, load default config
    default_path = (
        CACHED_MODEL_CONFIG_PATH
        if CACHED_MODEL_CONFIG_PATH.exists()
        else Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    )
    config = load_yaml_config(default_path)

    # 3. If user config exists, merge it
    user_path = resolve_config_path_from_env_var()
    if user_path and user_path.exists():
        user_config = load_yaml_config(user_path)
        config = update_config(config, user_config)
    elif user_path:
        warnings.warn(
            f"WARNING: Could not find user config: {str(user_path)}, revert to default config located at {default_path}",
            UserWarning,
            stacklevel=2,
        )

    return process_config(config)


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


def find_matching_dirs(
    log_dir: Path,
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
    job_id: Optional[int] = None,
    before_job_id: Optional[int] = None,
) -> list[Path]:
    """
    Find log directories based on filtering criteria.

    Parameters
    ----------
    log_dir : Path
        The base directory containing model family directories.
    model_family : str, optional
        Filter to only search inside this family.
    model_name : str, optional
        Filter to only match model names.
    job_id : int, optional
        Filter to only match this exact SLURM job ID.
    before_job_id : int, optional
        Filter to only include job IDs less than this value.

    Returns
    -------
    list[Path]
        List of directories that match the criteria and can be deleted.
    """
    matched = []

    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    if not model_family and not model_name and not job_id and not before_job_id:
        return [log_dir]

    for family_dir in log_dir.iterdir():
        if not family_dir.is_dir():
            continue
        if model_family and family_dir.name != model_family:
            continue

        if model_family and not model_name and not job_id and not before_job_id:
            return [family_dir]

        for job_dir in family_dir.iterdir():
            if not job_dir.is_dir():
                continue

            try:
                name_part, id_part = job_dir.name.rsplit(".", 1)
                parsed_id = int(id_part)
            except ValueError:
                continue

            if model_name and name_part != model_name:
                continue
            if job_id is not None and parsed_id != job_id:
                continue
            if before_job_id is not None and parsed_id >= before_job_id:
                continue

            matched.append(job_dir)

    return matched


def check_required_fields(params: dict[str, Any]) -> dict[str, Any]:
    """Check for required fields without default vals and their corresponding env vars.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameters to check.
    """
    env_overrides: dict[str, str] = {}

    if not REQUIRED_ARGS:
        return env_overrides
    for arg in REQUIRED_ARGS:
        if not params.get(arg):
            default_value = os.getenv(str(REQUIRED_ARGS[arg]))
            if default_value:
                params[arg] = default_value
                env_overrides[arg] = default_value
            else:
                raise MissingRequiredFieldsError(
                    f"{arg} is required, please set it in the command arguments or environment variables"
                )
    return env_overrides
