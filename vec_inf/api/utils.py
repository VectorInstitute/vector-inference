"""Utility functions for the Vector Inference API."""

from typing import Any, Optional

import requests

from vec_inf.api.models import ModelStatus
from vec_inf.cli._config import ModelConfig
from vec_inf.cli._utils import (
    MODEL_READY_SIGNATURE,
    get_base_url,
    read_slurm_log,
    run_bash_command,
)
from vec_inf.cli._utils import (
    load_config as cli_load_config,
)


class APIError(Exception):
    """Base exception for API errors."""

    pass


class ModelNotFoundError(APIError):
    """Exception raised when a model is not found."""

    pass


class SlurmJobError(APIError):
    """Exception raised when there's an error with a Slurm job."""

    pass


class ServerError(APIError):
    """Exception raised when there's an error with the inference server."""

    pass


def load_models() -> list[ModelConfig]:
    """Load model configurations."""
    return cli_load_config()


def parse_launch_output(output: str) -> tuple[str, dict[str, str]]:
    """Parse output from model launch command.

    Parameters
    ----------
    output: str
        Output from the launch command

    Returns
    -------
    tuple[str, dict[str, str]]
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


def get_model_status(
    slurm_job_id: str, log_dir: Optional[str] = None
) -> tuple[ModelStatus, dict[str, Any]]:
    """Get the status of a model.

    Parameters
    ----------
    slurm_job_id: str
        The Slurm job ID
    log_dir: str, optional
        Optional path to Slurm log directory

    Returns
    -------
    tuple[ModelStatus, dict[str, Any]]
        Model status and status information

    """
    status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
    output, _ = run_bash_command(status_cmd)

    # Check if job exists
    if "Invalid job id specified" in output:
        raise SlurmJobError(f"Job {slurm_job_id} not found")

    # Extract job information
    try:
        job_name = output.split(" ")[1].split("=")[1]
        job_state = output.split(" ")[9].split("=")[1]
    except IndexError as err:
        raise SlurmJobError(f"Could not parse job status for {slurm_job_id}") from err

    status_info = {
        "model_name": job_name,
        "base_url": None,
        "pending_reason": None,
        "failed_reason": None,
    }

    # Process based on job state
    if job_state == "PENDING":
        try:
            status_info["pending_reason"] = output.split(" ")[10].split("=")[1]
        except IndexError:
            status_info["pending_reason"] = "Unknown pending reason"
        return ModelStatus.PENDING, status_info

    if job_state in ["CANCELLED", "FAILED", "TIMEOUT", "PREEMPTED"]:
        return ModelStatus.SHUTDOWN, status_info

    if job_state == "RUNNING":
        return check_server_status(job_name, slurm_job_id, log_dir, status_info)

    # Unknown state
    status_info["failed_reason"] = f"Unknown job state: {job_state}"
    return ModelStatus.FAILED, status_info


def check_server_status(
    job_name: str, job_id: str, log_dir: Optional[str], status_info: dict[str, Any]
) -> tuple[ModelStatus, dict[str, Any]]:
    """Check the status of a running inference server."""
    # Initialize default status
    final_status = ModelStatus.LAUNCHING
    log_content = read_slurm_log(job_name, int(job_id), "err", log_dir)

    # Handle initial log reading error
    if isinstance(log_content, str):
        status_info["failed_reason"] = log_content
        return ModelStatus.FAILED, status_info

    # Process log content
    for line in log_content:
        line_lower = line.lower()

        # Check for error indicators
        if "error" in line_lower:
            status_info["failed_reason"] = line.strip("\n")
            final_status = ModelStatus.FAILED
            break

        # Check for server ready signal
        if MODEL_READY_SIGNATURE in line:
            base_url = get_base_url(job_name, int(job_id), log_dir)

            # Validate base URL
            if not isinstance(base_url, str) or not base_url.startswith("http"):
                status_info["failed_reason"] = f"Invalid base URL: {base_url}"
                final_status = ModelStatus.FAILED
                break

            status_info["base_url"] = base_url
            final_status = _perform_health_check(base_url, status_info)
            break  # Stop processing after first ready signature

    return final_status, status_info


def _perform_health_check(base_url: str, status_info: dict[str, Any]) -> ModelStatus:
    """Execute health check and return appropriate status."""
    health_check_url = base_url.replace("v1", "health")

    try:
        response = requests.get(health_check_url)
        if response.status_code == 200:
            return ModelStatus.READY

        status_info["failed_reason"] = (
            f"Health check failed with status code {response.status_code}"
        )
    except requests.exceptions.RequestException as e:
        status_info["failed_reason"] = f"Health check request error: {str(e)}"

    return ModelStatus.FAILED


def get_metrics(job_name: str, job_id: int, log_dir: Optional[str]) -> dict[str, str]:
    """Get the latest metrics for a model.

    Parameters
    ----------
    job_name: str
        The name of the Slurm job
    job_id: int
        The Slurm job ID
    log_dir: str, optional
        Optional path to Slurm log directory

    Returns
    -------
    dict[str, str]
        Dictionary of metrics or empty dict if not found

    """
    log_content = read_slurm_log(job_name, job_id, "out", log_dir)
    if isinstance(log_content, str):
        return {}

    # Find the latest metrics entry
    metrics = {}
    for line in reversed(log_content):
        if "Avg prompt throughput" in line:
            # Parse metrics from the line
            metrics_str = line.split("] ")[1].strip().strip(".")
            metrics_list = metrics_str.split(", ")
            for metric in metrics_list:
                key, value = metric.split(": ")
                metrics[key] = value
            break

    return metrics
