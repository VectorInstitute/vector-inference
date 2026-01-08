"""Data models for Vector Inference API.

This module contains the data model classes used by the Vector Inference API
for both request parameters and response objects.

Classes
-------
ModelStatus : Enum
    Status states of a model
ModelType : Enum
    Types of supported models
LaunchResponse : dataclass
    Response from model launch operation
StatusResponse : dataclass
    Response from model status check
MetricsResponse : dataclass
    Response from metrics collection
LaunchOptions : dataclass
    Options for model launch
LaunchOptionsDict : TypedDict
    Dictionary representation of launch options
ModelInfo : datacitten
    Information about available models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union, get_args

from vec_inf.client._slurm_vars import MODEL_TYPES


class ModelStatus(str, Enum):
    """Enum representing the possible status states of a model.

    Attributes
    ----------
    PENDING : str
        Model is waiting for Slurm to allocate resources
    LAUNCHING : str
        Model is in the process of starting
    READY : str
        Model is running and ready to serve requests
    FAILED : str
        Model failed to start or encountered an error
    SHUTDOWN : str
        Model was intentionally stopped
    UNAVAILABLE : str
        Model status cannot be determined
    """

    PENDING = "PENDING"
    LAUNCHING = "LAUNCHING"
    READY = "READY"
    FAILED = "FAILED"
    SHUTDOWN = "SHUTDOWN"
    UNAVAILABLE = "UNAVAILABLE"


# Extract model type values from the Literal type
_MODEL_TYPE_VALUES = get_args(MODEL_TYPES)


def _model_type_to_enum_name(model_type: str) -> str:
    """Convert a model type string to a valid enum attribute name."""
    # Convert to uppercase and replace hyphens with underscores
    return model_type.upper().replace("-", "_")


# Create ModelType enum dynamically from MODEL_TYPES
ModelType = Enum(  # type: ignore[misc]
    "ModelType",
    {_model_type_to_enum_name(mt): mt for mt in _MODEL_TYPE_VALUES},
    type=str,
    module=__name__,
)


@dataclass
class LaunchResponse:
    """Response from launching a model.

    Parameters
    ----------
    slurm_job_id : str
        ID of the launched SLURM job
    model_name : str
        Name of the launched model
    config : dict[str, Any]
        Configuration used for the launch
    raw_output : str
        Raw output from the launch command (hidden from repr)
    """

    slurm_job_id: str
    model_name: str
    config: dict[str, Any]
    raw_output: str = field(repr=False)


@dataclass
class BatchLaunchResponse:
    """Response from launching multiple models in batch mode.

    Parameters
    ----------
    slurm_job_id : str
        ID of the launched SLURM job
    slurm_job_name : str
        Name of the launched SLURM job
    model_names : list[str]
        Names of the launched models
    config : dict[str, Any]
        Configuration used for the launch
    raw_output : str
        Raw output from the launch command (hidden from repr)
    """

    slurm_job_id: str
    slurm_job_name: str
    model_names: list[str]
    config: dict[str, Any]
    raw_output: str = field(repr=False)


@dataclass
class StatusResponse:
    """Response from checking a model's status.

    Parameters
    ----------
    model_name : str
        Name of the model
    log_dir : str
        Path to the SLURM log directory
    server_status : ModelStatus
        Current status of the server
    job_state : Union[str, ModelStatus]
        Current state of the SLURM job
    raw_output : str
        Raw output from status check (hidden from repr)
    base_url : str, optional
        Base URL of the model server if ready
    pending_reason : str, optional
        Reason for pending state if applicable
    failed_reason : str, optional
        Reason for failure if applicable
    """

    model_name: str
    log_dir: str
    server_status: ModelStatus
    job_state: Union[str, ModelStatus]
    raw_output: str = field(repr=False)
    base_url: Optional[str] = None
    pending_reason: Optional[str] = None
    failed_reason: Optional[str] = None


@dataclass
class MetricsResponse:
    """Response from retrieving model metrics.

    Parameters
    ----------
    model_name : str
        Name of the model
    metrics : Union[dict[str, float], str]
        Either a dictionary of metrics or an error message
    timestamp : float
        Unix timestamp of when metrics were collected
    """

    model_name: str
    metrics: Union[dict[str, float], str]
    timestamp: float


@dataclass
class LaunchOptions:
    """Options for launching a model.

    Parameters
    ----------
    model_family : str, optional
        Family/architecture of the model
    model_variant : str, optional
        Specific variant/version of the model
    partition : str, optional
        SLURM partition to use
    resource_type : str, optional
        Type of resource to request for the job
    num_nodes : int, optional
        Number of nodes to allocate
    gpus_per_node : int, optional
        Number of GPUs per node
    cpus_per_task : int, optional
        Number of CPUs per task
    mem_per_node : str, optional
        Memory per node
    account : str, optional
        Account name for job scheduling
    work_dir : str, optional
        Set working directory for the batch job
    qos : str, optional
        Quality of Service level
    time : str, optional
        Time limit for the job
    exclude : str, optional
        Exclude certain nodes from the resources granted to the job
    node_list : str, optional
        Request a specific list of nodes for deployment
    bind : str, optional
        Additional binds for the container as a comma separated list of bind paths
    vocab_size : int, optional
        Size of model vocabulary
    data_type : str, optional
        Data type for model weights
    venv : str, optional
        Virtual environment to use
    log_dir : str, optional
        Directory for logs
    model_weights_parent_dir : str, optional
        Parent directory containing model weights
    engine: str, optional
        Inference engine to use
    vllm_args : str, optional
        vLLM engine arguments
    sglang_args : str, optional
        SGLang engine arguments
    env : str, optional
        Environment variables to be set
    config : str, optional
        Path to custom model config yaml
    """

    model_family: Optional[str] = None
    model_variant: Optional[str] = None
    partition: Optional[str] = None
    resource_type: Optional[str] = None
    num_nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    mem_per_node: Optional[str] = None
    account: Optional[str] = None
    work_dir: Optional[str] = None
    qos: Optional[str] = None
    exclude: Optional[str] = None
    nodelist: Optional[str] = None
    bind: Optional[str] = None
    time: Optional[str] = None
    vocab_size: Optional[int] = None
    data_type: Optional[str] = None
    venv: Optional[str] = None
    log_dir: Optional[str] = None
    model_weights_parent_dir: Optional[str] = None
    engine: Optional[str] = None
    vllm_args: Optional[str] = None
    sglang_args: Optional[str] = None
    env: Optional[str] = None
    config: Optional[str] = None


@dataclass
class ModelInfo:
    """Information about an available model.

    Parameters
    ----------
    name : str
        Name of the model
    family : str
        Family/architecture of the model
    variant : str, optional
        Specific variant/version of the model
    model_type : ModelType
        Type of the model
    config : dict[str, Any]
        Additional configuration parameters
    """

    name: str
    family: str
    variant: Optional[str]
    model_type: ModelType
    config: dict[str, Any]
