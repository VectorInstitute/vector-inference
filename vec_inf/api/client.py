"""Vector Inference client for programmatic access.

This module provides the main client class for interacting with Vector Inference
services programmatically.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from vec_inf.api.models import (
    LaunchOptions,
    LaunchResponse,
    MetricsResponse,
    ModelInfo,
    StatusResponse,
)
from vec_inf.api.utils import (
    APIError,
    ModelNotFoundError,
    ServerError,
    SlurmJobError,
    get_metrics,
    get_model_status,
    load_models,
)
from vec_inf.shared.config import ModelConfig
from vec_inf.shared.models import ModelStatus, ModelType
from vec_inf.shared.utils import (
    parse_launch_output,
    run_bash_command,
)


class VecInfClient:
    """Client for interacting with Vector Inference programmatically.

    This class provides methods for launching models, checking their status,
    retrieving metrics, and shutting down models using the Vector Inference
    infrastructure.

    Examples
    --------
    >>> from vec_inf.api import VecInfClient
    >>> client = VecInfClient()
    >>> response = client.launch_model("Meta-Llama-3.1-8B-Instruct")
    >>> job_id = response.slurm_job_id
    >>> status = client.get_status(job_id)
    >>> if status.status == ModelStatus.READY:
    ...     print(f"Model is ready at {status.base_url}")
    >>> client.shutdown_model(job_id)

    """

    def __init__(self) -> None:
        """Initialize the Vector Inference client."""
        pass

    def list_models(self) -> List[ModelInfo]:
        """List all available models.

        Returns
        -------
        List[ModelInfo]
            ModelInfo objects containing information about available models.

        Raises
        ------
        APIError
            If there was an error retrieving model information.

        """
        try:
            model_configs = load_models()
            result = []

            for config in model_configs:
                info = ModelInfo(
                    name=config.model_name,
                    family=config.model_family,
                    variant=config.model_variant,
                    type=ModelType(config.model_type),
                    config=config.model_dump(exclude={"model_name", "venv", "log_dir"}),
                )
                result.append(info)

            return result
        except Exception as e:
            raise APIError(f"Failed to list models: {str(e)}") from e

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the configuration for a specific model.

        Parameters
        ----------
        model_name: str
            Name of the model to get configuration for.

        Returns
        -------
        ModelConfig
            Model configuration.

        Raises
        ------
        ModelNotFoundError
            Error if the specified model is not found.
        APIError
                Error if there was an error retrieving the model configuration.

        """
        try:
            model_configs = load_models()
            for config in model_configs:
                if config.model_name == model_name:
                    return config

            raise ModelNotFoundError(f"Model '{model_name}' not found")
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise APIError(f"Failed to get model configuration: {str(e)}") from e

    def launch_model(
        self, model_name: str, options: Optional[LaunchOptions] = None
    ) -> LaunchResponse:
        """Launch a model on the cluster.

        Parameters
        ----------
        model_name: str
            Name of the model to launch.
        options: LaunchOptions, optional
            Optional launch options to override default configuration.

        Returns
        -------
        LaunchResponse
            Information about the launched model.

        Raises
        ------
        ModelNotFoundError
            Error if the specified model is not found.
        APIError
            Error if there was an error launching the model.
        """
        try:
            # Build the launch command
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                "launch_server.sh",
            )
            base_command = f"bash {script_path}"

            # Get model configuration
            try:
                model_config = self.get_model_config(model_name)
            except ModelNotFoundError:
                raise

            # Apply options if provided
            params = model_config.model_dump(exclude={"model_name"})
            if options:
                options_dict = {k: v for k, v in vars(options).items() if v is not None}
                params.update(options_dict)

            # Build the command with parameters
            command = base_command
            for param_name, param_value in params.items():
                if param_value is None:
                    continue

                # Format boolean values
                if isinstance(param_value, bool):
                    formatted_value = "True" if param_value else "False"
                elif isinstance(param_value, Path):
                    formatted_value = str(param_value)
                else:
                    formatted_value = param_value

                arg_name = param_name.replace("_", "-")
                command += f" --{arg_name} {formatted_value}"

            # Execute the command
            output, _ = run_bash_command(command)

            # Parse the output
            job_id, config_dict = parse_launch_output(output)

            return LaunchResponse(
                slurm_job_id=job_id,
                model_name=model_name,
                config=config_dict,
                raw_output=output,
            )
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise APIError(f"Failed to launch model: {str(e)}") from e

    def get_status(
        self, slurm_job_id: str, log_dir: Optional[str] = None
    ) -> StatusResponse:
        """Get the status of a running model.

        Parameters
        ----------
        slurm_job_id: str
            The Slurm job ID to check.
        log_dir: str, optional
            Optional path to the Slurm log directory.

        Returns
        -------
        StatusResponse
            Model status information.

        Raises
        ------
        SlurmJobError
            Error if the specified job is not found or there's an error with the job.
        APIError
            Error if there was an error retrieving the status.
        """
        try:
            status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
            output, _ = run_bash_command(status_cmd)

            status, status_info = get_model_status(slurm_job_id, log_dir)

            return StatusResponse(
                slurm_job_id=slurm_job_id,
                model_name=status_info["model_name"],
                status=status,
                base_url=status_info["base_url"],
                pending_reason=status_info["pending_reason"],
                failed_reason=status_info["failed_reason"],
                raw_output=output,
            )
        except SlurmJobError:
            raise
        except Exception as e:
            raise APIError(f"Failed to get status: {str(e)}") from e

    def get_metrics(
        self, slurm_job_id: str, log_dir: Optional[str] = None
    ) -> MetricsResponse:
        """Get the performance metrics of a running model.

        Parameters
        ----------
        slurm_job_id : str
            The Slurm job ID to get metrics for.
        log_dir : str, optional
            Optional path to the Slurm log directory.

        Returns
        -------
        MetricsResponse
            Object containing the model's performance metrics.

        Raises
        ------
        SlurmJobError
            If the specified job is not found or there's an error with the job.
        APIError
            If there was an error retrieving the metrics.

        """
        try:
            # First check if the job exists and get the job name
            status_response = self.get_status(slurm_job_id, log_dir)

            # Get metrics
            metrics = get_metrics(
                status_response.model_name, int(slurm_job_id), log_dir
            )

            return MetricsResponse(
                slurm_job_id=slurm_job_id,
                model_name=status_response.model_name,
                metrics=metrics,
                timestamp=time.time(),
                raw_output="",  # No raw output needed for metrics
            )
        except SlurmJobError:
            raise
        except Exception as e:
            raise APIError(f"Failed to get metrics: {str(e)}") from e

    def shutdown_model(self, slurm_job_id: str) -> bool:
        """Shutdown a running model.

        Parameters
        ----------
        slurm_job_id: str
            The Slurm job ID to shut down.

        Returns
        -------
            True if the model was successfully shutdown, False otherwise.

        Raises
        ------
            APIError: If there was an error shutting down the model.
        """
        try:
            shutdown_cmd = f"scancel {slurm_job_id}"
            run_bash_command(shutdown_cmd)
            return True
        except Exception as e:
            raise APIError(f"Failed to shutdown model: {str(e)}") from e

    def wait_until_ready(
        self,
        slurm_job_id: str,
        timeout_seconds: int = 1800,
        poll_interval_seconds: int = 10,
        log_dir: Optional[str] = None,
    ) -> StatusResponse:
        """Wait until a model is ready or fails.

        Parameters
        ----------
        slurm_job_id: str
            The Slurm job ID to wait for.
        timeout_seconds: int
            Maximum time to wait in seconds (default: 30 mins).
        poll_interval_seconds: int
            How often to check status in seconds (default: 10s).
        log_dir: str, optional
            Optional path to the Slurm log directory.

        Returns
        -------
        StatusResponse
            Status, if the model is ready or failed.

        Raises
        ------
        SlurmJobError
            If the specified job is not found or there's an error with the job.
        ServerError
            If the server fails to start within the timeout period.
        APIError
            If there was an error checking the status.

        """
        start_time = time.time()

        while True:
            status = self.get_status(slurm_job_id, log_dir)

            if status.status == ModelStatus.READY:
                return status

            if status.status == ModelStatus.FAILED:
                error_message = status.failed_reason or "Unknown error"
                raise ServerError(f"Model failed to start: {error_message}")

            if status.status == ModelStatus.SHUTDOWN:
                raise ServerError("Model was shutdown before it became ready")

            # Check timeout
            if time.time() - start_time > timeout_seconds:
                raise ServerError(
                    f"Timed out waiting for model to become ready after {timeout_seconds} seconds"
                )

            # Wait before checking again
            time.sleep(poll_interval_seconds)
