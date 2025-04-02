"""Helper classes for the API."""

import json
import warnings
from pathlib import Path
from typing import Any, Optional

from vec_inf.api._models import LaunchResponse, ModelInfo, ModelType
from vec_inf.shared._helper import LaunchHelper, ListHelper


class APILaunchHelper(LaunchHelper):
    """API Helper class for handling inference server launch."""

    def __init__(self, model_name: str, kwargs: Optional[dict[str, Any]]):
        super().__init__(model_name, kwargs)

    def _warn(self, message: str) -> None:
        """Warn the user about a potential issue."""
        warnings.warn(message, UserWarning, stacklevel=2)

    def post_launch_processing(self, command_output: str) -> LaunchResponse:
        """Process and display launch output."""
        slurm_job_id = command_output.split(" ")[-1].strip().strip("\n")
        self.params["slurm_job_id"] = slurm_job_id
        job_json = Path(
            self.params["log_dir"],
            f"{self.model_name}.{slurm_job_id}",
            f"{self.model_name}.{slurm_job_id}.json",
        )
        job_json.parent.mkdir(parents=True, exist_ok=True)
        job_json.touch(exist_ok=True)

        with job_json.open("w") as file:
            json.dump(self.params, file, indent=4)

        return LaunchResponse(
            slurm_job_id=int(slurm_job_id),
            model_name=self.model_name,
            config=self.params,
            raw_output=command_output,
        )


class APIListHelper(ListHelper):
    """API Helper class for handling model listing."""

    def __init__(self):
        super().__init__()

    def get_all_models(self) -> list[ModelInfo]:
        """Get all available models."""
        available_models = []
        for config in self.model_configs:
            info = ModelInfo(
                name=config.model_name,
                family=config.model_family,
                variant=config.model_variant,
                type=ModelType(config.model_type),
                config=config.model_dump(exclude={"model_name", "venv", "log_dir"}),
            )
            available_models.append(info)
        return available_models
