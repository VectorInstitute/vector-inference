"""Vector Inference client for programmatic access.

This module provides the main client class for interacting with Vector Inference
services programmatically. It includes functionality for launching models, monitoring
their status, collecting metrics, and managing their lifecycle.

See Also
--------
vec_inf.client._helper : Helper classes for model inference server management
vec_inf.client.models : Data models for API responses
"""

import asyncio
import time
import warnings
from typing import Any, Optional, Union

from vec_inf.client._exceptions import (
    ServerError,
    SlurmJobError,
)
from vec_inf.client._helper import (
    ModelLauncher,
    ModelRegistry,
    ModelStatusMonitor,
    PerformanceMetricsCollector,
)
from vec_inf.client._utils import run_bash_command
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import (
    LaunchOptions,
    LaunchResponse,
    MetricsResponse,
    ModelInfo,
    ModelStatus,
    StatusResponse,
)


class VecInfClient:
    """Client for interacting with Vector Inference programmatically.

    This class provides methods for launching models, checking their status,
    retrieving metrics, and shutting down models using the Vector Inference
    infrastructure.

    Methods
    -------
    list_models()
        List all available models
    get_model_config(model_name)
        Get configuration for a specific model
    launch_model(model_name, options)
        Launch a model on the cluster
    get_status(slurm_job_id, log_dir)
        Get status of a running model
    get_metrics(slurm_job_id, log_dir)
        Get performance metrics of a running model
    shutdown_model(slurm_job_id)
        Shutdown a running model
    wait_until_ready(slurm_job_id, timeout_seconds, poll_interval_seconds, log_dir)
        Wait for a model to become ready

    Examples
    --------
    >>> import asyncio
    >>> from vec_inf.api import VecInfClient
    >>> async def main():
    ...     client = VecInfClient()
    ...     response = client.launch_model("Meta-Llama-3.1-8B-Instruct")
    ...     job_id = response.slurm_job_id
    ...     status = await client.get_status(job_id)
    ...     if status.server_status == ModelStatus.READY:
    ...         print(f"Model is ready at {status.base_url}")
    ...     await client.wait_until_ready(job_id)
    ...     client.shutdown_model(job_id)
    >>> asyncio.run(main())
    """

    def __init__(self) -> None:
        """Initialize the Vector Inference client."""
        pass

    def list_models(self) -> list[ModelInfo]:
        """List all available models.

        Returns
        -------
        list[ModelInfo]
            List of ModelInfo objects containing information about available models,
            including their configurations and specifications.
        """
        model_registry = ModelRegistry()
        return model_registry.get_all_models()

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the configuration for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model to get configuration for

        Returns
        -------
        ModelConfig
            Complete configuration for the specified model

        Raises
        ------
        ModelNotFoundError
            If the specified model is not found in the configuration
        """
        model_registry = ModelRegistry()
        return model_registry.get_single_model_config(model_name)

    def launch_model(
        self, model_name: str, options: Optional[LaunchOptions] = None
    ) -> LaunchResponse:
        """Launch a model on the cluster.

        Parameters
        ----------
        model_name : str
            Name of the model to launch
        options : LaunchOptions, optional
            Launch options to override default configuration

        Returns
        -------
        LaunchResponse
            Response containing launch details including:
            - SLURM job ID
            - Model configuration
            - Launch status

        Raises
        ------
        ModelConfigurationError
            If the model configuration is invalid
        SlurmJobError
            If there's an error launching the SLURM job
        """
        # Convert LaunchOptions to dictionary if provided
        options_dict: dict[str, Any] = {}
        if options:
            options_dict = {k: v for k, v in vars(options).items() if v is not None}

        # Create and use the API Launch Helper
        model_launcher = ModelLauncher(model_name, options_dict)
        return model_launcher.launch()

    async def get_status(
        self, slurm_job_id: int, log_dir: Optional[str] = None
    ) -> StatusResponse:
        """Get the status of a running model.

        Parameters
        ----------
        slurm_job_id : int
            The SLURM job ID to check
        log_dir : str, optional
            Path to the SLURM log directory. If None, uses default location

        Returns
        -------
        StatusResponse
            Status information including:
            - Model name
            - Server status
            - Job state
            - Base URL (if ready)
            - Error information (if failed)
        """
        return await asyncio.to_thread(
            self._get_status_sync, slurm_job_id, log_dir
        )

    def _get_status_sync(
        self, slurm_job_id: int, log_dir: Optional[str] = None
    ) -> StatusResponse:
        """Blocking helper that retrieves model status."""
        model_status_monitor = ModelStatusMonitor(slurm_job_id, log_dir)
        return model_status_monitor.process_model_status()

    def get_metrics(
        self, slurm_job_id: int, log_dir: Optional[str] = None
    ) -> MetricsResponse:
        """Get the performance metrics of a running model.

        Parameters
        ----------
        slurm_job_id : int
            The SLURM job ID to get metrics for
        log_dir : str, optional
            Path to the SLURM log directory. If None, uses default location

        Returns
        -------
        MetricsResponse
            Response containing:
            - Model name
            - Performance metrics or error message
            - Timestamp of collection
        """
        performance_metrics_collector = PerformanceMetricsCollector(
            slurm_job_id, log_dir
        )

        metrics: Union[dict[str, float], str]
        if not performance_metrics_collector.metrics_url.startswith("http"):
            metrics = performance_metrics_collector.metrics_url
        else:
            metrics = performance_metrics_collector.fetch_metrics()

        return MetricsResponse(
            model_name=performance_metrics_collector.status_info.model_name,
            metrics=metrics,
            timestamp=time.time(),
        )

    def shutdown_model(self, slurm_job_id: int) -> bool:
        """Shutdown a running model.

        Parameters
        ----------
        slurm_job_id : int
            The SLURM job ID to shut down

        Returns
        -------
        bool
            True if the model was successfully shutdown

        Raises
        ------
        SlurmJobError
            If there was an error shutting down the model
        """
        shutdown_cmd = f"scancel {slurm_job_id}"
        _, stderr = run_bash_command(shutdown_cmd)
        if stderr:
            raise SlurmJobError(f"Failed to shutdown model: {stderr}")
        return True

    async def wait_until_ready(
        self,
        slurm_job_id: int,
        timeout_seconds: int = 1800,
        poll_interval_seconds: int = 10,
        log_dir: Optional[str] = None,
    ) -> StatusResponse:
        """Wait until a model is ready or fails.

        Parameters
        ----------
        slurm_job_id : int
            The SLURM job ID to wait for
        timeout_seconds : int, optional
            Maximum time to wait in seconds, by default 1800 (30 mins)
        poll_interval_seconds : int, optional
            How often to check status in seconds, by default 10
        log_dir : str, optional
            Path to the SLURM log directory. If None, uses default location

        Returns
        -------
        StatusResponse
            Status information when the model becomes ready

        Raises
        ------
        SlurmJobError
            If the specified job is not found or there's an error with the job
        ServerError
            If the server fails to start within the timeout period
        APIError
            If there was an error checking the status

        Notes
        -----
        The timeout is reset if the model is still in PENDING state after the
        initial timeout period. This allows for longer queue times in the SLURM
        scheduler.
        """
        start_time = time.time()

        while True:
            status_info = await self.get_status(slurm_job_id, log_dir)

            if status_info.server_status == ModelStatus.READY:
                return status_info

            if status_info.server_status == ModelStatus.FAILED:
                error_message = status_info.failed_reason or "Unknown error"
                raise ServerError(f"Model failed to start: {error_message}")

            if status_info.server_status == ModelStatus.SHUTDOWN:
                raise ServerError("Model was shutdown before it became ready")

            # Check timeout
            if time.time() - start_time > timeout_seconds:
                if status_info.server_status == ModelStatus.PENDING:
                    warnings.warn(
                        f"Model is still pending after {timeout_seconds} seconds, resetting timer...",
                        UserWarning,
                        stacklevel=2,
                    )
                    start_time = time.time()
                raise ServerError(
                    f"Timed out waiting for model to become ready after {timeout_seconds} seconds"
                )

            # Wait before checking again
            await asyncio.sleep(poll_interval_seconds)
