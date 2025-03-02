"""Utility functions for the Vector Inference API."""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import requests

from vec_inf.api.models import ModelStatus
from vec_inf.cli._utils import (
    MODEL_READY_SIGNATURE,
    SERVER_ADDRESS_SIGNATURE,
    load_config as cli_load_config,
    read_slurm_log,
    run_bash_command,
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


def load_models():
    """Load model configurations."""
    return cli_load_config()


def parse_launch_output(output: str) -> Tuple[str, Dict[str, str]]:
    """Parse output from model launch command.
    
    Args:
        output: Output from the launch command
        
    Returns:
        Tuple of (slurm_job_id, dict of config key-value pairs)
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
    slurm_job_id: str, 
    log_dir: Optional[str] = None
) -> Tuple[ModelStatus, Dict[str, Any]]:
    """Get the status of a model.
    
    Args:
        slurm_job_id: The Slurm job ID
        log_dir: Optional path to Slurm log directory
        
    Returns:
        Tuple of (ModelStatus, dict with additional status info)
    """
    status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
    output = run_bash_command(status_cmd)
    
    # Check if job exists
    if "Invalid job id specified" in output:
        raise SlurmJobError(f"Job {slurm_job_id} not found")
    
    # Extract job information
    try:
        job_name = output.split(" ")[1].split("=")[1]
        job_state = output.split(" ")[9].split("=")[1]
    except IndexError:
        raise SlurmJobError(f"Could not parse job status for {slurm_job_id}")
    
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
    
    elif job_state in ["CANCELLED", "FAILED", "TIMEOUT", "PREEMPTED"]:
        return ModelStatus.SHUTDOWN, status_info
    
    elif job_state == "RUNNING":
        return check_server_status(job_name, slurm_job_id, log_dir, status_info)
    
    else:
        # Unknown state
        status_info["failed_reason"] = f"Unknown job state: {job_state}"
        return ModelStatus.FAILED, status_info


def check_server_status(
    job_name: str, 
    job_id: str, 
    log_dir: Optional[str], 
    status_info: Dict[str, Any]
) -> Tuple[ModelStatus, Dict[str, Any]]:
    """Check the status of a running inference server.
    
    Args:
        job_name: The name of the Slurm job
        job_id: The Slurm job ID
        log_dir: Optional path to Slurm log directory
        status_info: Dictionary to update with status information
        
    Returns:
        Tuple of (ModelStatus, updated status_info)
    """
    # Read error log to check if server is running
    log_content = read_slurm_log(job_name, int(job_id), "err", log_dir)
    if isinstance(log_content, str):
        status_info["failed_reason"] = log_content
        return ModelStatus.FAILED, status_info
    
    # Check for errors or if server is ready
    for line in log_content:
        if "error" in line.lower():
            status_info["failed_reason"] = line.strip("\n")
            return ModelStatus.FAILED, status_info
        
        if MODEL_READY_SIGNATURE in line:
            # Server is running, get URL and check health
            base_url = get_base_url(job_name, int(job_id), log_dir)
            if not isinstance(base_url, str) or not base_url.startswith("http"):
                status_info["failed_reason"] = f"Invalid base URL: {base_url}"
                return ModelStatus.FAILED, status_info
            
            status_info["base_url"] = base_url
            
            # Check if the server is healthy
            health_check_url = base_url.replace("v1", "health")
            try:
                response = requests.get(health_check_url)
                if response.status_code == 200:
                    return ModelStatus.READY, status_info
                else:
                    status_info["failed_reason"] = f"Health check failed with status code {response.status_code}"
                    return ModelStatus.FAILED, status_info
            except requests.exceptions.RequestException as e:
                status_info["failed_reason"] = f"Health check request error: {str(e)}"
                return ModelStatus.FAILED, status_info
    
    # If we get here, server is running but not yet ready
    return ModelStatus.LAUNCHING, status_info


def get_base_url(job_name: str, job_id: int, log_dir: Optional[str]) -> str:
    """Get the base URL of a running model.
    
    Args:
        job_name: The name of the Slurm job
        job_id: The Slurm job ID
        log_dir: Optional path to Slurm log directory
        
    Returns:
        The base URL string or an error message
    """
    log_content = read_slurm_log(job_name, job_id, "out", log_dir)
    if isinstance(log_content, str):
        return log_content

    for line in log_content:
        if SERVER_ADDRESS_SIGNATURE in line:
            return line.split(SERVER_ADDRESS_SIGNATURE)[1].strip("\n")
    return "URL_NOT_FOUND"


def get_metrics(job_name: str, job_id: int, log_dir: Optional[str]) -> Dict[str, str]:
    """Get the latest metrics for a model.
    
    Args:
        job_name: The name of the Slurm job
        job_id: The Slurm job ID
        log_dir: Optional path to Slurm log directory
        
    Returns:
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