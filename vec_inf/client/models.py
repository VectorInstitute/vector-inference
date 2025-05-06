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
from typing import Any, Optional, Union


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


class ModelType(str, Enum):
    """Enum representing the possible model types.

    Attributes
    ----------
    LLM : str
        Large Language Model
    VLM : str
        Vision Language Model
    TEXT_EMBEDDING : str
        Text Embedding Model
    REWARD_MODELING : str
        Reward Modeling Model
    """

    LLM = "LLM"
    VLM = "VLM"
    TEXT_EMBEDDING = "Text_Embedding"
    REWARD_MODELING = "Reward_Modeling"


@dataclass
class LaunchResponse:
    """Response from launching a model.

    Parameters
    ----------
    slurm_job_id : int
        ID of the launched SLURM job
    model_name : str
        Name of the launched model
    config : dict[str, Any]
        Configuration used for the launch
    raw_output : str
        Raw output from the launch command (hidden from repr)
    """

    slurm_job_id: int
    model_name: str
    config: dict[str, Any]
    raw_output: str = field(repr=False)


@dataclass
class StatusResponse:
    """Response from checking a model's status.

    Parameters
    ----------
    model_name : str
        Name of the model
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
    num_nodes : int, optional
        Number of nodes to allocate
    gpus_per_node : int, optional
        Number of GPUs per node
    account : str, optional
        Account name for job scheduling
    qos : str, optional
        Quality of Service level
    time : str, optional
        Time limit for the job
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
    vllm_args : str, optional
        Additional arguments for vLLM
    """

    model_family: Optional[str] = None
    model_variant: Optional[str] = None
    partition: Optional[str] = None
    num_nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    time: Optional[str] = None
    vocab_size: Optional[int] = None
    data_type: Optional[str] = None
    venv: Optional[str] = None
    log_dir: Optional[str] = None
    model_weights_parent_dir: Optional[str] = None
    vllm_args: Optional[str] = None


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
