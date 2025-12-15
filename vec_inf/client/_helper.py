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
    ENGINE_SHORT_TO_LONG_MAP,
    KEY_METRICS,
    SRC_DIR,
    SUPPORTED_ENGINES,
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
from vec_inf.client._slurm_vars import CONTAINER_MODULE_NAME, IMAGE_PATH
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
        self.model_config = self._get_model_configuration(self.kwargs.get("config"))
        self.engine = ""
        self.params = self._get_launch_params()

    def _warn(self, message: str) -> None:
        """Warn the user about a potential issue.

        Parameters
        ----------
        message : str
            Warning message to display
        """
        warnings.warn(message, UserWarning, stacklevel=2)

    def _get_model_configuration(self, config_path: str | None = None) -> ModelConfig:
        """Load and validate model configuration.

        Parameters
        ----------
        config_path : str | None, optional
            Path to a yaml file with custom model config to use in place of the default

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
        model_configs = utils.load_config(config_path=config_path)
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

    def _process_engine_args(
        self, arg_string: str, engine_choice: str
    ) -> dict[str, Any]:
        """Process the engine_args string into a dictionary.

        Parameters
        ----------
        arg_string : str
            Comma-separated string of inference engine arguments

        Returns
        -------
        dict[str, Any]
            Processed inference engine arguments as key-value pairs
        """
        engine_args: dict[str, str | bool] = {}
        engine_arg_map = ENGINE_SHORT_TO_LONG_MAP[engine_choice]

        for arg in arg_string.split(","):
            if "=" in arg:
                key, value = arg.split("=")
                if key.strip() in engine_arg_map:
                    key = engine_arg_map[key.strip()]
                engine_args[key.strip()] = value.strip()
            elif "-O" in arg.strip():
                if engine_choice != "vllm":
                    raise ValueError("-O is only supported for vLLM")
                key = engine_arg_map["-O"]
                engine_args[key] = arg.strip()[2:].strip()
            else:
                engine_args[arg.strip()] = True
        return engine_args

    def _process_env_vars(self, env_arg: str) -> dict[str, str]:
        """Process the env string into a dictionary of environment variables.

        Parameters
        ----------
        env_arg : str
            String containing comma separated list of environment variable definitions
            (eg. MY_VAR=1), file paths containing environment variable definitions
            (separated by newlines), or a combination of both
            (eg. 'MY_VAR=5,my_env.env')

        Returns
        -------
        dict[str, str]
            Processed environment variables as key-value pairs.
        """
        env_vars: dict[str, str] = {}
        for arg in env_arg.split(","):
            if "=" in arg:  # Arg is an env var definition
                key, value = arg.split("=")
                env_vars[key.strip()] = value.strip()
            else:  # Arg is a path to a file
                with open(arg, "r") as file:
                    lines = [line.rstrip() for line in file]
                for line in lines:
                    if "=" in line:
                        key, value = line.split("=")
                        env_vars[key.strip()] = value.strip()
                    else:
                        print(f"WARNING: Could not parse env var: {line}")
        return env_vars

    def _engine_check_override(self, params: dict[str, Any]) -> None:
        """Check for engine override in CLI args and warn user.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of launch parameters to check
        """

        def overwrite_engine_args(params: dict[str, Any]) -> None:
            engine_args = self._process_engine_args(
                self.kwargs[f"{self.engine}_args"], self.engine
            )
            for key, value in engine_args.items():
                params["engine_args"][key] = value
            del self.kwargs[f"{self.engine}_args"]

        # Infer engine name from engine-specific args if provided
        extracted_engine = ""
        for engine in SUPPORTED_ENGINES:
            if self.kwargs.get(f"{engine}_args"):
                if not extracted_engine:
                    extracted_engine = engine
                else:
                    raise ValueError(
                        "Cannot provide engine-specific args for multiple engines, please choose one"
                    )
        # Check for mismatch between provided engine arg and engine-specific args
        input_engine = self.kwargs.get("engine", "")

        if input_engine and extracted_engine:
            if input_engine != extracted_engine:
                raise ValueError(
                    f"Mismatch between provided engine '{input_engine}' and engine-specific args '{extracted_engine}'"
                )
            self.engine = input_engine
            params["engine_args"] = params[f"{self.engine}_args"]
            overwrite_engine_args(params)
        elif input_engine:
            # Only engine arg in CLI, use default engine args from config
            self.engine = input_engine
            params["engine_args"] = params[f"{self.engine}_args"]
        elif extracted_engine:
            # Only engine-specific args in CLI, infer engine and warn user
            self.engine = extracted_engine
            params["engine_inferred"] = True
            params["engine_args"] = params[f"{self.engine}_args"]
            overwrite_engine_args(params)
        else:
            # No engine-related args in CLI, use defaults from config
            self.engine = params.get("engine", "vllm")
            params["engine_args"] = params[f"{self.engine}_args"]

        # Remove $ENGINE_NAME_args from params as they won't get populated to sjob json.
        for engine in SUPPORTED_ENGINES:
            del params[f"{engine}_args"]

    def _apply_cli_overrides(self, params: dict[str, Any]) -> None:
        """Apply CLI argument overrides to params.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of launch parameters to override
        """
        self._engine_check_override(params)

        if self.kwargs.get("env"):
            env_vars = self._process_env_vars(self.kwargs["env"])
            for key, value in env_vars.items():
                params["env"][key] = str(value)
            del self.kwargs["env"]

        if self.kwargs.get("bind") and params.get("bind"):
            params["bind"] = f"{params['bind']},{self.kwargs['bind']}"
            del self.kwargs["bind"]

        for key, value in self.kwargs.items():
            params[key] = value

    def _validate_resource_allocation(self, params: dict[str, Any]) -> None:
        """Validate resource allocation and parallelization settings.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of launch parameters to validate

        Raises
        ------
        MissingRequiredFieldsError
            If tensor parallel size is not specified when using multiple GPUs
        ValueError
            If total # of GPUs requested is not a power of two
            If mismatch between total # of GPUs requested and parallelization settings
        """
        if (
            int(params["gpus_per_node"]) > 1
            and params["engine_args"].get("--tensor-parallel-size") is None
        ):
            raise MissingRequiredFieldsError(
                "--tensor-parallel-size is required when gpus_per_node > 1"
            )

        total_gpus_requested = int(params["gpus_per_node"]) * int(params["num_nodes"])
        if not utils.is_power_of_two(total_gpus_requested):
            raise ValueError("Total number of GPUs requested must be a power of two")

        total_parallel_sizes = int(
            params["engine_args"].get("--tensor-parallel-size", "1")
        ) * int(params["engine_args"].get("--pipeline-parallel-size", "1"))
        if total_gpus_requested != total_parallel_sizes:
            raise ValueError(
                "Mismatch between total number of GPUs requested and parallelization settings"
            )

    def _setup_log_files(self, params: dict[str, Any]) -> None:
        """Set up log directory and file paths.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of launch parameters to set up log files
        """
        params["log_dir"] = Path(params["log_dir"], params["model_family"]).expanduser()
        params["log_dir"].mkdir(parents=True, exist_ok=True)
        params["src_dir"] = SRC_DIR

        params["out_file"] = (
            f"{params['log_dir']}/{self.model_name}.%j/{self.model_name}.%j.out"
        )
        params["err_file"] = (
            f"{params['log_dir']}/{self.model_name}.%j/{self.model_name}.%j.err"
        )
        params["json_file"] = (
            f"{params['log_dir']}/{self.model_name}.$SLURM_JOB_ID/{self.model_name}.$SLURM_JOB_ID.json"
        )

    def _get_launch_params(self) -> dict[str, Any]:
        """Prepare launch parameters, set log dir, and validate required fields.

        Returns
        -------
        dict[str, Any]
            Dictionary of prepared launch parameters
        """
        params = self.model_config.model_dump(exclude_none=True)

        # Override config defaults with CLI arguments
        self._apply_cli_overrides(params)

        # Check for required fields without default vals, will raise an error if missing
        utils.check_required_fields(params)

        # Validate resource allocation and parallelization settings
        self._validate_resource_allocation(params)

        # Convert gpus_per_node and resource_type to gres
        resource_type = params.get("resource_type")
        if resource_type:
            params["gres"] = f"gpu:{resource_type}:{params['gpus_per_node']}"
        else:
            params["gres"] = f"gpu:{params['gpus_per_node']}"

        # Setup log files
        self._setup_log_files(params)

        # Convert path to string for JSON serialization
        for field in params:
            # Keep structured fields (dicts/bools) intact
            if field in ["engine_args", "env", "engine_inferred"]:
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
            job_log_dir / f"{self.model_name}.{self.slurm_job_id}.sbatch"
        )

        # Replace venv with image path if using container
        if self.params["venv"] == CONTAINER_MODULE_NAME:
            self.params["venv"] = IMAGE_PATH[self.params["engine"]]

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

    def __init__(
        self,
        model_names: list[str],
        batch_config: Optional[str] = None,
        account: Optional[str] = None,
        work_dir: Optional[str] = None,
    ):
        self.model_names = model_names
        self.batch_config = batch_config
        self.slurm_job_id = ""
        self.slurm_job_name = self._get_slurm_job_name()
        self.batch_script_path = Path("")
        self.launch_script_paths: list[Path] = []
        self.model_configs = self._get_model_configurations()
        self.params = self._get_launch_params(account, work_dir)

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

    def _validate_resource_and_parallel_settings(
        self,
        config: ModelConfig,
        model_engine_args: dict[str, Any] | None,
        model_name: str,
    ) -> None:
        """Validate resource allocation and parallelization settings for each model.

        Parameters
        ----------
        config : ModelConfig
            Configuration of the model to validate
        model_engine_args : dict[str, Any] | None
            Inference engine arguments of the model to validate
        model_name : str
            Name of the model to validate

        Raises
        ------
        MissingRequiredFieldsError
            If tensor parallel size is not specified when using multiple GPUs
        ValueError
            If total # of GPUs requested is not a power of two
            If mismatch between total # of GPUs requested and parallelization settings
        """
        if (
            int(config.gpus_per_node) > 1
            and (model_engine_args or {}).get("--tensor-parallel-size") is None
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
            (model_engine_args or {}).get("--tensor-parallel-size", "1")
        ) * int((model_engine_args or {}).get("--pipeline-parallel-size", "1"))
        if total_gpus_requested != total_parallel_sizes:
            raise ValueError(
                f"Mismatch between total number of GPUs requested and parallelization settings, check your configuration for {model_name}"
            )

    def _get_launch_params(
        self, account: Optional[str] = None, work_dir: Optional[str] = None
    ) -> dict[str, Any]:
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
        common_params: dict[str, Any] = {
            "slurm_job_name": self.slurm_job_name,
            "src_dir": str(SRC_DIR),
            "account": account,
            "work_dir": work_dir,
        }

        params: dict[str, Any] = common_params.copy()
        params["models"] = {}

        for i, (model_name, config) in enumerate(self.model_configs.items()):
            params["models"][model_name] = config.model_dump(exclude_none=True)
            params["models"][model_name]["het_group_id"] = i

            model_engine_args = getattr(config, f"{config.engine}_args", None)
            params["models"][model_name]["engine_args"] = model_engine_args
            for engine in SUPPORTED_ENGINES:
                del params["models"][model_name][f"{engine}_args"]

            # Validate resource allocation and parallelization settings
            self._validate_resource_and_parallel_settings(
                config, model_engine_args, model_name
            )

            # Convert gpus_per_node and resource_type to gres
            params["models"][model_name]["gres"] = (
                f"gpu:{config.resource_type}:{config.gpus_per_node}"
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
            # Check for required fields and return environment variable overrides
            env_overrides = utils.check_required_fields(
                {**params["models"][model_name], **common_params}
            )

            for arg, value in env_overrides.items():
                if arg in common_params:
                    params[arg] = value
                else:
                    params["models"][model_name][arg] = value

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
            main_job_log_dir / f"{self.slurm_job_name}.{self.slurm_job_id}.sbatch"
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
            job_name = self.job_status["JobName"].removesuffix("-vec-inf")
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
        return sorted(available_models, key=lambda x: x.name)

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
            (c for c in self.model_configs if c.model_name == model_name),
            None,
        )
        if not config:
            raise ModelNotFoundError(f"Model '{model_name}' not found in configuration")
        return config
