"""Helper classes for the model.

This module provides utility classes for managing model deployment, status monitoring,
metrics collection, and model registry operations.
"""

import json
import time
import warnings
from pathlib import Path
from shutil import copy2
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse, urlunparse

import requests

import vec_inf.client._utils as utils
from vec_inf.client._client_vars import (
    BATCH_MODE_REQUIRED_MATCHING_ARGS,
    KEY_METRICS,
    SRC_DIR,
    VLLM_SHORT_TO_LONG_MAP,
)
from vec_inf.client._exceptions import (
    MissingRequiredFieldsError,
    ModelConfigurationError,
    ModelNotFoundError,
    SlurmJobError,
)
from vec_inf.client._slurm_script_generator import (
    BatchSlurmScriptGenerator,
    SlurmScriptGenerator,
)
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import (
    BatchLaunchResponse,
    LaunchResponse,
    ModelInfo,
    ModelStatus,
    ModelType,
    StatusResponse,
)


class ModelLauncher:
    """Helper class for handling inference server launch.

    A class that manages the launch process of inference servers, including
    configuration validation, parameter preparation, and SLURM job submission.

    Parameters
    ----------
    model_name: str
        Name of the model to launch
    kwargs: Optional[dict[str, Any]]
        Optional launch keyword arguments to override default configuration
    """

    def __init__(self, model_name: str, kwargs: Optional[dict[str, Any]]):
        self.model_name = model_name
        self.kwargs = kwargs or {}
        self.slurm_job_id = ""
        self.slurm_script_path = Path("")
        self.model_config = self._get_model_configuration()
        self.params = self._get_launch_params()

    def _warn(self, message: str) -> None:
        """Warn the user about a potential issue.

        Parameters
        ----------
        message : str
            Warning message to display
        """
        warnings.warn(message, UserWarning, stacklevel=2)

    def _get_model_configuration(self) -> ModelConfig:
        """Load and validate model configuration.

        Returns
        -------
        ModelConfig
            Validated configuration for the model

        Raises
        ------
        ModelNotFoundError
            If model weights parent directory cannot be determined
        ModelConfigurationError
            If model configuration is not found and weights don't exist
        """
        model_configs = utils.load_config()
        config = next(
            (m for m in model_configs if m.model_name == self.model_name), None
        )

        if config:
            return config

        # If model config not found, check for path from CLI kwargs or use fallback
        model_weights_parent_dir = self.kwargs.get(
            "model_weights_parent_dir",
            model_configs[0].model_weights_parent_dir if model_configs else None,
        )

        if not model_weights_parent_dir:
            raise ModelNotFoundError(
                "Could not determine model weights parent directory"
            )

        model_weights_path = Path(model_weights_parent_dir, self.model_name)

        # Only give a warning if weights exist but config missing
        if model_weights_path.exists():
            self._warn(
                f"Warning: '{self.model_name}' configuration not found in config, please ensure model configuration are properly set in command arguments",
            )
            # Return a dummy model config object with model name and weights parent dir
            return ModelConfig(
                model_name=self.model_name,
                model_family="model_family_placeholder",
                model_type="LLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=1000,
                model_weights_parent_dir=Path(str(model_weights_parent_dir)),
            )

        raise ModelConfigurationError(
            f"'{self.model_name}' not found in configuration and model weights "
            f"not found at expected path '{model_weights_path}'"
        )

    def _process_vllm_args(self, arg_string: str) -> dict[str, Any]:
        """Process the vllm_args string into a dictionary.

        Parameters
        ----------
        arg_string : str
            Comma-separated string of vLLM arguments

        Returns
        -------
        dict[str, Any]
            Processed vLLM arguments as key-value pairs
        """
        vllm_args: dict[str, str | bool] = {}
        for arg in arg_string.split(","):
            if "=" in arg:
                key, value = arg.split("=")
                if key.strip() in VLLM_SHORT_TO_LONG_MAP:
                    key = VLLM_SHORT_TO_LONG_MAP[key.strip()]
                vllm_args[key.strip()] = value.strip()
            elif "-O" in arg.strip():
                key = VLLM_SHORT_TO_LONG_MAP["-O"]
                vllm_args[key] = arg.strip()[2:].strip()
            else:
                vllm_args[arg.strip()] = True
        return vllm_args

    def _get_launch_params(self) -> dict[str, Any]:
        """Prepare launch parameters, set log dir, and validate required fields.

        Returns
        -------
        dict[str, Any]
            Dictionary of prepared launch parameters

        Raises
        ------
        MissingRequiredFieldsError
            If required fields are missing or tensor parallel size is not specified
            when using multiple GPUs
        """
        params = self.model_config.model_dump(exclude_none=True)

        # Override config defaults with CLI arguments
        if self.kwargs.get("vllm_args"):
            vllm_args = self._process_vllm_args(self.kwargs["vllm_args"])
            for key, value in vllm_args.items():
                params["vllm_args"][key] = value
            del self.kwargs["vllm_args"]

        for key, value in self.kwargs.items():
            params[key] = value

        # Validate resource allocation and parallelization settings
        if (
            int(params["gpus_per_node"]) > 1
            and params["vllm_args"].get("--tensor-parallel-size") is None
        ):
            raise MissingRequiredFieldsError(
                "--tensor-parallel-size is required when gpus_per_node > 1"
            )

        total_gpus_requested = int(params["gpus_per_node"]) * int(params["num_nodes"])
        if not utils.is_power_of_two(total_gpus_requested):
            raise ValueError("Total number of GPUs requested must be a power of two")

        total_parallel_sizes = int(
            params["vllm_args"].get("--tensor-parallel-size", "1")
        ) * int(params["vllm_args"].get("--pipeline-parallel-size", "1"))
        if total_gpus_requested != total_parallel_sizes:
            raise ValueError(
                "Mismatch between total number of GPUs requested and parallelization settings"
            )

        # Create log directory
        params["log_dir"] = Path(params["log_dir"], params["model_family"]).expanduser()
        params["log_dir"].mkdir(parents=True, exist_ok=True)
        params["src_dir"] = SRC_DIR

        # Construct slurm log file paths
        params["out_file"] = (
            f"{params['log_dir']}/{self.model_name}.%j/{self.model_name}.%j.out"
        )
        params["err_file"] = (
            f"{params['log_dir']}/{self.model_name}.%j/{self.model_name}.%j.err"
        )
        params["json_file"] = (
            f"{params['log_dir']}/{self.model_name}.$SLURM_JOB_ID/{self.model_name}.$SLURM_JOB_ID.json"
        )

        # Convert path to string for JSON serialization
        for field in params:
            if field == "vllm_args":
                continue
            params[field] = str(params[field])

        return params

    def _build_launch_command(self) -> str:
        """Generate the slurm script and construct the launch command.

        Returns
        -------
        str
            Complete SLURM launch command
        """
        self.slurm_script_path = SlurmScriptGenerator(self.params).write_to_log_dir()
        return f"sbatch {self.slurm_script_path}"

    def launch(self) -> LaunchResponse:
        """Launch the model.

        Returns
        -------
        LaunchResponse
            Response object containing launch details and status

        Raises
        ------
        SlurmJobError
            If SLURM job submission fails
        """
        # Build and execute the launch command
        command_output, stderr = utils.run_bash_command(self._build_launch_command())

        if stderr:
            raise SlurmJobError(f"Error: {stderr}")

        # Extract slurm job id from command output
        self.slurm_job_id = command_output.split(" ")[-1].strip().strip("\n")
        self.params["slurm_job_id"] = self.slurm_job_id

        # Create log directory and job json file, move slurm script to job log directory
        job_log_dir = Path(
            self.params["log_dir"], f"{self.model_name}.{self.slurm_job_id}"
        )
        job_log_dir.mkdir(parents=True, exist_ok=True)

        job_json = Path(
            job_log_dir,
            f"{self.model_name}.{self.slurm_job_id}.json",
        )
        job_json.touch(exist_ok=True)

        self.slurm_script_path.rename(
            job_log_dir / f"{self.model_name}.{self.slurm_job_id}.slurm"
        )

        with job_json.open("w") as file:
            json.dump(self.params, file, indent=4)

        return LaunchResponse(
            slurm_job_id=self.slurm_job_id,
            model_name=self.model_name,
            config=self.params,
            raw_output=command_output,
        )


class BatchModelLauncher:
    """Helper class for handling batch inference server launch.

    A class that manages the launch process of multiple inference servers, including
    configuration validation, and SLURM job submission.

    Parameters
    ----------
    model_names : list[str]
        List of model names to launch
    """

    def __init__(self, model_names: list[str], batch_config: Optional[str] = None):
        self.model_names = model_names
        self.batch_config = batch_config
        self.slurm_job_id = ""
        self.slurm_job_name = self._get_slurm_job_name()
        self.batch_script_path = Path("")
        self.launch_script_paths: list[Path] = []
        self.model_configs = self._get_model_configurations()
        self.params = self._get_launch_params()

    def _get_slurm_job_name(self) -> str:
        """Get the SLURM job name from the model names.

        Returns
        -------
        str
            SLURM job name
        """
        return "BATCH-" + "-".join(self.model_names)

    def _get_model_configurations(self) -> dict[str, ModelConfig]:
        """Load and validate model configurations.

        Returns
        -------
        dict[str, ModelConfig]
            Dictionary of validated model configurations

        Raises
        ------
        ModelNotFoundError
            If model weights parent directory cannot be determined
        ModelConfigurationError
            If model configuration is not found and weights don't exist
        """
        model_configs = utils.load_config(self.batch_config)

        model_configs_dict = {}
        for model_name in self.model_names:
            config = next(
                (m for m in model_configs if m.model_name == model_name), None
            )

            if config:
                model_configs_dict[model_name] = config
            else:
                raise ModelConfigurationError(
                    f"'{model_name}' not found in configuration, batch launch requires all models to be present in the configuration file"
                )

        return model_configs_dict

    def _get_launch_params(self) -> dict[str, Any]:
        """Prepare launch parameters, set log dir, and validate required fields.

        Returns
        -------
        dict[str, Any]
            Dictionary of prepared launch parameters

        Raises
        ------
        MissingRequiredFieldsError
            If required fields are missing or tensor parallel size is not specified
            when using multiple GPUs
        """
        params: dict[str, Any] = {
            "models": {},
            "slurm_job_name": self.slurm_job_name,
            "src_dir": str(SRC_DIR),
        }

        for i, (model_name, config) in enumerate(self.model_configs.items()):
            params["models"][model_name] = config.model_dump(exclude_none=True)
            params["models"][model_name]["het_group_id"] = i

            # Validate resource allocation and parallelization settings
            if (
                int(config.gpus_per_node) > 1
                and (config.vllm_args or {}).get("--tensor-parallel-size") is None
            ):
                raise MissingRequiredFieldsError(
                    f"--tensor-parallel-size is required when gpus_per_node > 1, check your configuration for {model_name}"
                )

            total_gpus_requested = int(config.gpus_per_node) * int(config.num_nodes)
            if not utils.is_power_of_two(total_gpus_requested):
                raise ValueError(
                    f"Total number of GPUs requested must be a power of two, check your configuration for {model_name}"
                )

            total_parallel_sizes = int(
                (config.vllm_args or {}).get("--tensor-parallel-size", "1")
            ) * int((config.vllm_args or {}).get("--pipeline-parallel-size", "1"))
            if total_gpus_requested != total_parallel_sizes:
                raise ValueError(
                    f"Mismatch between total number of GPUs requested and parallelization settings, check your configuration for {model_name}"
                )

            # Create log directory
            log_dir = Path(
                params["models"][model_name]["log_dir"], self.slurm_job_name
            ).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            params["models"][model_name]["log_dir"] = str(log_dir)

            # Convert model_weights_parent_dir to string for JSON serialization
            params["models"][model_name]["model_weights_parent_dir"] = str(
                params["models"][model_name]["model_weights_parent_dir"]
            )

            # Construct slurm log file paths
            params["models"][model_name]["out_file"] = (
                f"{params['models'][model_name]['log_dir']}/{self.slurm_job_name}.%j/{model_name}.%j.out"
            )
            params["models"][model_name]["err_file"] = (
                f"{params['models'][model_name]['log_dir']}/{self.slurm_job_name}.%j/{model_name}.%j.err"
            )
            params["models"][model_name]["json_file"] = (
                f"{params['models'][model_name]['log_dir']}/{self.slurm_job_name}.$SLURM_JOB_ID/{model_name}.$SLURM_JOB_ID.json"
            )

            # Create top level log files using the first model's log directory
            if not params.get("out_file"):
                params["out_file"] = (
                    f"{params['models'][model_name]['log_dir']}/{self.slurm_job_name}.%j/{self.slurm_job_name}.%j.out"
                )
            if not params.get("err_file"):
                params["err_file"] = (
                    f"{params['models'][model_name]['log_dir']}/{self.slurm_job_name}.%j/{self.slurm_job_name}.%j.err"
                )

            # Check if required matching arguments are matched
            for arg in BATCH_MODE_REQUIRED_MATCHING_ARGS:
                if not params.get(arg):
                    params[arg] = params["models"][model_name][arg]
                elif params[arg] != params["models"][model_name][arg]:
                    # Remove the created directory since we found a mismatch
                    log_dir.rmdir()
                    raise ValueError(
                        f"Mismatch found for {arg}: {params[arg]} != {params['models'][model_name][arg]}, check your configuration"
                    )

        return params

    def _build_launch_command(self) -> str:
        """Generate the slurm script and construct the launch command.

        Returns
        -------
        str
            Complete SLURM launch command
        """
        batch_script_generator = BatchSlurmScriptGenerator(self.params)
        self.batch_script_path = batch_script_generator.generate_batch_slurm_script()
        self.launch_script_paths = batch_script_generator.script_paths
        return f"sbatch {str(self.batch_script_path)}"

    def launch(self) -> BatchLaunchResponse:
        """Launch models in batch mode.

        Returns
        -------
        BatchLaunchResponse
            Response object containing launch details and status

        Raises
        ------
        SlurmJobError
            If SLURM job submission fails
        """
        # Build and execute the launch command
        command_output, stderr = utils.run_bash_command(self._build_launch_command())

        if stderr:
            raise SlurmJobError(f"Error: {stderr}")

        # Extract slurm job id from command output
        self.slurm_job_id = command_output.split(" ")[-1].strip().strip("\n")
        self.params["slurm_job_id"] = self.slurm_job_id

        # Create log directory and job json file, move slurm script to job log directory
        main_job_log_dir = Path("")

        for model_name in self.model_names:
            model_job_id = int(self.slurm_job_id) + int(
                self.params["models"][model_name]["het_group_id"]
            )

            job_log_dir = Path(
                self.params["log_dir"], f"{self.slurm_job_name}.{model_job_id}"
            )
            job_log_dir.mkdir(parents=True, exist_ok=True)

            if main_job_log_dir == Path(""):
                main_job_log_dir = job_log_dir

            job_json = Path(
                job_log_dir,
                f"{model_name}.{model_job_id}.json",
            )
            job_json.touch(exist_ok=True)

            with job_json.open("w") as file:
                json.dump(self.params["models"][model_name], file, indent=4)

        # Copy the launch scripts to the job log directory, the original scripts
        # cannot be deleted otherwise slurm will not be able to find them
        script_path_mapper = {}
        for script_path in self.launch_script_paths:
            old_path = script_path.name
            file_name = old_path.split("/")[-1]
            copy2(script_path, main_job_log_dir / file_name)
            new_path = script_path.name
            script_path_mapper[old_path] = new_path

        # Replace old launch script paths with new paths in batch slurm script
        with self.batch_script_path.open("r") as f:
            script_content = f.read()
        for old_path, new_path in script_path_mapper.items():
            script_content = script_content.replace(old_path, new_path)
        with self.batch_script_path.open("w") as f:
            f.write(script_content)

        # Move the batch script to the job log directory
        self.batch_script_path.rename(
            main_job_log_dir / f"{self.slurm_job_name}.{self.slurm_job_id}.slurm"
        )

        return BatchLaunchResponse(
            slurm_job_id=self.slurm_job_id,
            slurm_job_name=self.slurm_job_name,
            model_names=self.model_names,
            config=self.params,
            raw_output=command_output,
        )


class ModelStatusMonitor:
    """Class for handling server status information and monitoring.

    A class that monitors and reports the status of deployed model servers,
    including job state and health checks.

    Parameters
    ----------
    slurm_job_id : str
        ID of the SLURM job to monitor
    """

    def __init__(self, slurm_job_id: str):
        self.slurm_job_id = slurm_job_id
        self.output = self._get_raw_status_output()
        self.job_status = dict(
            field.split("=", 1) for field in self.output.split() if "=" in field
        )
        self.log_dir = self._get_log_dir()
        self.status_info = self._get_base_status_data()

    def _get_raw_status_output(self) -> str:
        """Get the raw server status output from slurm.

        Returns
        -------
        str
            Raw output from scontrol command

        Raises
        ------
        SlurmJobError
            If status check fails
        """
        status_cmd = f"scontrol show job {self.slurm_job_id} --oneliner"
        output, stderr = utils.run_bash_command(status_cmd)

        if stderr:
            raise SlurmJobError(f"Error: {stderr}")
        return output

    def _get_log_dir(self) -> str:
        """Get the log directory for the job.

        Returns
        -------
        str
            Log directory for the job
        """
        try:
            outfile_path = self.job_status["StdOut"]
            directory = Path(outfile_path).parent
            return str(directory)
        except KeyError as err:
            raise FileNotFoundError(
                f"Output file not found for job {self.slurm_job_id}"
            ) from err

    def _get_base_status_data(self) -> StatusResponse:
        """Extract basic job status information from scontrol output.

        Returns
        -------
        StatusResponse
            Basic status information for the job
        """
        try:
            job_name = self.job_status["JobName"]
            job_state = self.job_status["JobState"]
        except KeyError:
            job_name = "UNAVAILABLE"
            job_state = ModelStatus.UNAVAILABLE

        return StatusResponse(
            model_name=job_name,
            log_dir=self.log_dir,
            server_status=ModelStatus.UNAVAILABLE,
            job_state=job_state,
            raw_output=self.output,
            base_url="UNAVAILABLE",
            pending_reason=None,
            failed_reason=None,
        )

    def _check_model_health(self) -> None:
        """Check model health and update status accordingly."""
        status, status_code = utils.model_health_check(
            self.status_info.model_name, self.slurm_job_id, self.log_dir
        )
        if status == ModelStatus.READY:
            self.status_info.base_url = utils.get_base_url(
                self.status_info.model_name,
                self.slurm_job_id,
                self.log_dir,
            )
            self.status_info.server_status = status
        else:
            self.status_info.server_status = status
            self.status_info.failed_reason = cast(str, status_code)

    def _process_running_state(self) -> None:
        """Process RUNNING job state and check server status."""
        server_status = utils.is_server_running(
            self.status_info.model_name, self.slurm_job_id, self.log_dir
        )

        if isinstance(server_status, tuple):
            self.status_info.server_status, self.status_info.failed_reason = (
                server_status
            )
            return

        if server_status == "RUNNING":
            self._check_model_health()
        else:
            self.status_info.server_status = cast(ModelStatus, server_status)

    def _process_pending_state(self) -> None:
        """Process PENDING job state and update status information."""
        try:
            self.status_info.pending_reason = self.job_status["Reason"]
            self.status_info.server_status = ModelStatus.PENDING
        except KeyError:
            self.status_info.pending_reason = "Unknown pending reason"

    def process_model_status(self) -> StatusResponse:
        """Process different job states and update status information.

        Returns
        -------
        StatusResponse
            Complete status information for the model
        """
        if self.status_info.job_state == ModelStatus.PENDING:
            self._process_pending_state()
        elif self.status_info.job_state == "RUNNING":
            self._process_running_state()

        return self.status_info


class PerformanceMetricsCollector:
    """Class for handling metrics collection and processing.

    A class that collects and processes performance metrics from running model servers,
    including throughput and latency measurements.

    Parameters
    ----------
    slurm_job_id : str
        ID of the SLURM job to collect metrics from
    log_dir : str, optional
        Directory containing log files
    """

    def __init__(self, slurm_job_id: str):
        self.slurm_job_id = slurm_job_id
        self.status_info = self._get_status_info()
        self.log_dir = self.status_info.log_dir
        self.metrics_url = self._build_metrics_url()
        self.enabled_prefix_caching = self._check_prefix_caching()

        self._prev_prompt_tokens: float = 0.0
        self._prev_generation_tokens: float = 0.0
        self._last_updated: Optional[float] = None
        self._last_throughputs = {"prompt": 0.0, "generation": 0.0}

    def _get_status_info(self) -> StatusResponse:
        """Retrieve status info using existing StatusHelper.

        Returns
        -------
        StatusResponse
            Current status information for the model
        """
        status_helper = ModelStatusMonitor(self.slurm_job_id)
        return status_helper.process_model_status()

    def _build_metrics_url(self) -> str:
        """Construct metrics endpoint URL from base URL with version stripping.

        Returns
        -------
        str
            Complete metrics endpoint URL or status message
        """
        if self.status_info.job_state == ModelStatus.PENDING:
            return "Pending resources for server initialization"

        base_url = utils.get_base_url(
            self.status_info.model_name,
            self.slurm_job_id,
            self.log_dir,
        )
        if not base_url.startswith("http"):
            return "Server not ready"

        parsed = urlparse(base_url)
        clean_path = parsed.path.replace("/v1", "", 1).rstrip("/")
        return urlunparse(
            (parsed.scheme, parsed.netloc, f"{clean_path}/metrics", "", "", "")
        )

    def _check_prefix_caching(self) -> bool:
        """Check if prefix caching is enabled.

        Returns
        -------
        bool
            True if prefix caching is enabled, False otherwise
        """
        job_json = utils.read_slurm_log(
            self.status_info.model_name,
            self.slurm_job_id,
            "json",
            self.log_dir,
        )
        if isinstance(job_json, str):
            return False
        return bool(cast(dict[str, str], job_json).get("enable_prefix_caching", False))

    def _parse_metrics(self, metrics_text: str) -> dict[str, float]:
        """Parse metrics with latency count and sum.

        Parameters
        ----------
        metrics_text : str
            Raw metrics text from the server

        Returns
        -------
        dict[str, float]
            Parsed metrics as key-value pairs
        """
        key_metrics = KEY_METRICS

        if self.enabled_prefix_caching:
            key_metrics["vllm:gpu_prefix_cache_hit_rate"] = "gpu_prefix_cache_hit_rate"
            key_metrics["vllm:cpu_prefix_cache_hit_rate"] = "cpu_prefix_cache_hit_rate"

        parsed: dict[str, float] = {}
        for line in metrics_text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            metric_name = parts[0].split("{")[0]
            if metric_name in key_metrics:
                try:
                    parsed[key_metrics[metric_name]] = float(parts[1])
                except (ValueError, IndexError):
                    continue
        return parsed

    def fetch_metrics(self) -> Union[dict[str, float], str]:
        """Fetch metrics from the endpoint.

        Returns
        -------
        Union[dict[str, float], str]
            Dictionary of metrics or error message if request fails
        """
        try:
            response = requests.get(self.metrics_url, timeout=3)
            response.raise_for_status()
            current_metrics = self._parse_metrics(response.text)
            current_time = time.time()

            # Set defaults using last known throughputs
            current_metrics.setdefault(
                "prompt_tokens_per_sec", self._last_throughputs["prompt"]
            )
            current_metrics.setdefault(
                "generation_tokens_per_sec", self._last_throughputs["generation"]
            )

            if self._last_updated is None:
                self._prev_prompt_tokens = current_metrics.get(
                    "total_prompt_tokens", 0.0
                )
                self._prev_generation_tokens = current_metrics.get(
                    "total_generation_tokens", 0.0
                )
                self._last_updated = current_time
                return current_metrics

            time_diff = current_time - self._last_updated
            if time_diff > 0:
                current_prompt = current_metrics.get("total_prompt_tokens", 0.0)
                current_gen = current_metrics.get("total_generation_tokens", 0.0)

                delta_prompt = current_prompt - self._prev_prompt_tokens
                delta_gen = current_gen - self._prev_generation_tokens

                # Only update throughputs when we have new tokens
                prompt_tps = (
                    delta_prompt / time_diff
                    if delta_prompt > 0
                    else self._last_throughputs["prompt"]
                )
                gen_tps = (
                    delta_gen / time_diff
                    if delta_gen > 0
                    else self._last_throughputs["generation"]
                )

                current_metrics["prompt_tokens_per_sec"] = prompt_tps
                current_metrics["generation_tokens_per_sec"] = gen_tps

                # Persist calculated values regardless of activity
                self._last_throughputs["prompt"] = prompt_tps
                self._last_throughputs["generation"] = gen_tps

                # Update tracking state
                self._prev_prompt_tokens = current_prompt
                self._prev_generation_tokens = current_gen
                self._last_updated = current_time

            # Calculate average latency if data is available
            if (
                "request_latency_sum" in current_metrics
                and "request_latency_count" in current_metrics
            ):
                latency_sum = current_metrics["request_latency_sum"]
                latency_count = current_metrics["request_latency_count"]
                current_metrics["avg_request_latency"] = (
                    latency_sum / latency_count if latency_count > 0 else 0.0
                )

            return current_metrics

        except requests.RequestException as e:
            return f"Metrics request failed, `metrics` endpoint might not be ready yet: {str(e)}"


class ModelRegistry:
    """Class for handling model listing and configuration management.

    A class that provides functionality for listing available models and
    managing their configurations.
    """

    def __init__(self) -> None:
        """Initialize the model lister."""
        self.model_configs = utils.load_config()

    def get_all_models(self) -> list[ModelInfo]:
        """Get all available models.

        Returns
        -------
        list[ModelInfo]
            List of information about all available models
        """
        available_models = []
        for config in self.model_configs:
            info = ModelInfo(
                name=config.model_name,
                family=config.model_family,
                variant=config.model_variant,
                model_type=ModelType(config.model_type),
                config=config.model_dump(exclude={"model_name", "venv", "log_dir"}),
            )
            available_models.append(info)
        return available_models

    def get_single_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve configuration for

        Returns
        -------
        ModelConfig
            Configuration for the specified model

        Raises
        ------
        ModelNotFoundError
            If the specified model is not found in configuration
        """
        config = next(
            (c for c in self.model_configs if c.model_name == model_name), None
        )
        if not config:
            raise ModelNotFoundError(f"Model '{model_name}' not found in configuration")
        return config
