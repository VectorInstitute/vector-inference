"""Model configuration."""

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from vec_inf.client.slurm_vars import DEFAULT_ARGS, PARTITION, QOS


class ModelConfig(BaseModel):
    """Pydantic model for validating and managing model deployment configurations."""

    model_name: str = Field(..., min_length=3, pattern=r"^[a-zA-Z0-9\-_\.]+$")
    model_family: str = Field(..., min_length=2)
    model_variant: Optional[str] = Field(
        default=None, description="Specific variant/version of the model family"
    )
    model_type: Literal["LLM", "VLM", "Text_Embedding", "Reward_Modeling"] = Field(
        ..., description="Type of model architecture"
    )
    gpus_per_node: int = Field(..., gt=0, le=8, description="GPUs per node")
    num_nodes: int = Field(..., gt=0, le=16, description="Number of nodes")
    cpus_per_task: int = Field(
        default=DEFAULT_ARGS["cpus_per_task"],
        gt=0,
        le=128,
        description="CPUs per task",
    )
    mem_per_node: str = Field(
        default=DEFAULT_ARGS["mem_per_node"],
        pattern=r"^\d{1,4}G$",
        description="Memory per node",
    )
    vocab_size: int = Field(..., gt=0, le=1_000_000)
    qos: Union[QOS, str] = Field(
        default=DEFAULT_ARGS["qos"], description="Quality of Service tier"
    )
    time: str = Field(
        default=DEFAULT_ARGS["time"],
        pattern=r"^\d{2}:\d{2}:\d{2}$",
        description="HH:MM:SS time limit",
    )
    partition: Union[PARTITION, str] = Field(
        default=DEFAULT_ARGS["partition"], description="GPU partition type"
    )
    venv: str = Field(
        default="singularity", description="Virtual environment/container system"
    )
    log_dir: Path = Field(
        default=Path(DEFAULT_ARGS["log_dir"]), description="Log directory path"
    )
    model_weights_parent_dir: Path = Field(
        default=Path(DEFAULT_ARGS["model_weights_parent_dir"]),
        description="Base directory for model weights",
    )
    vllm_args: Optional[dict[str, Any]] = Field(
        default={}, description="vLLM engine arguments"
    )

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, validate_default=True, frozen=True
    )
