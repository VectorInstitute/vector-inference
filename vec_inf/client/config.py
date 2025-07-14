"""Model configuration.

This module provides a Pydantic model for validating and managing model deployment
configurations, including hardware requirements and model specifications.
"""

from pathlib import Path
from typing import Any, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from vec_inf.client.slurm_vars import (
    DEFAULT_ARGS,
    MAX_CPUS_PER_TASK,
    MAX_GPUS_PER_NODE,
    MAX_NUM_NODES,
    PARTITION,
    QOS,
)


class ModelConfig(BaseModel):
    """Pydantic model for validating and managing model deployment configurations.

    A configuration class that handles validation and management of model deployment
    settings, including model specifications, hardware requirements, and runtime
    parameters.

    Parameters
    ----------
    model_name : str
        Name of the model, must be alphanumeric with allowed characters: '-', '_', '.'
    model_family : str
        Family/architecture of the model
    model_variant : str, optional
        Specific variant or version of the model family
    model_type : {'LLM', 'VLM', 'Text_Embedding', 'Reward_Modeling'}
        Type of model architecture
    gpus_per_node : int
        Number of GPUs to use per node (1-MAX_GPUS_PER_NODE)
    num_nodes : int
        Number of nodes to use for deployment (1-MAX_NUM_NODES)
    cpus_per_task : int, optional
        Number of CPU cores per task (1-MAX_CPUS_PER_TASK)
    mem_per_node : str, optional
        Memory allocation per node in GB format (e.g., '32G')
    vocab_size : int
        Size of the model's vocabulary (1-1,000,000)
    account : Optional[str], optional
        Charge resources used by this job to specified account.
    qos : Union[QOS, str], optional
        Quality of Service tier for job scheduling
    time : str, optional
        Time limit for the job in HH:MM:SS format
    partition : Union[PARTITION, str], optional
        GPU partition type for job scheduling
    venv : str, optional
        Virtual environment or container system to use
    log_dir : Path, optional
        Directory path for storing logs
    model_weights_parent_dir : Path, optional
        Base directory containing model weights
    vllm_args : dict[str, Any], optional
        Additional arguments for vLLM engine configuration

    Notes
    -----
    All fields are validated using Pydantic's validation system. The model is
    configured to be immutable (frozen) and forbids extra fields.
    """

    model_name: str = Field(..., min_length=3, pattern=r"^[a-zA-Z0-9\-_\.]+$")
    model_family: str = Field(..., min_length=2)
    model_variant: Optional[str] = Field(
        default=None, description="Specific variant/version of the model family"
    )
    model_type: Literal["LLM", "VLM", "Text_Embedding", "Reward_Modeling"] = Field(
        ..., description="Type of model architecture"
    )
    gpus_per_node: int = Field(
        ..., gt=0, le=MAX_GPUS_PER_NODE, description="GPUs per node"
    )
    num_nodes: int = Field(..., gt=0, le=MAX_NUM_NODES, description="Number of nodes")
    cpus_per_task: int = Field(
        default=cast(int, DEFAULT_ARGS["cpus_per_task"]),
        gt=0,
        le=MAX_CPUS_PER_TASK,
        description="CPUs per task",
    )
    mem_per_node: str = Field(
        default=cast(str, DEFAULT_ARGS["mem_per_node"]),
        pattern=r"^\d{1,4}G$",
        description="Memory per node",
    )
    vocab_size: int = Field(..., gt=0, le=1_000_000)
    account: Optional[str] = Field(
        default=None, description="Account name for job scheduling"
    )
    qos: Union[QOS, str] = Field(
        default=cast(str, DEFAULT_ARGS["qos"]), description="Quality of Service tier"
    )
    time: str = Field(
        default=cast(str, DEFAULT_ARGS["time"]),
        pattern=r"^\d{2}:\d{2}:\d{2}$",
        description="HH:MM:SS time limit",
    )
    partition: Union[PARTITION, str] = Field(
        default=cast(str, DEFAULT_ARGS["partition"]), description="GPU partition type"
    )
    exclude: Optional[str] = Field(
        default=None,
        description="Exclude certain nodes from the resources granted to the job",
    )
    nodelist: Optional[str] = Field(
        default=None, description="Request a specific list of nodes for deployment"
    )
    bind: Optional[str] = Field(
        default=None, description="Additional binds for the singularity container"
    )
    venv: str = Field(
        default="singularity", description="Virtual environment/container system"
    )
    log_dir: Path = Field(
        default=Path(cast(str, DEFAULT_ARGS["log_dir"])),
        description="Log directory path",
    )
    model_weights_parent_dir: Path = Field(
        default=Path(cast(str, DEFAULT_ARGS["model_weights_parent_dir"])),
        description="Base directory for model weights",
    )
    vllm_args: Optional[dict[str, Any]] = Field(
        default={}, description="vLLM engine arguments"
    )

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, validate_default=True, frozen=True
    )
