"""
Data models for Vector Inference API.

This module contains the data model classes used by the Vector Inference API
for both request parameters and response objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypedDict, Union

from typing_extensions import NotRequired


class ModelStatus(str, Enum):
    """Enum representing the possible status states of a model."""

    PENDING = "PENDING"
    LAUNCHING = "LAUNCHING"
    READY = "READY"
    FAILED = "FAILED"
    SHUTDOWN = "SHUTDOWN"
    UNAVAILABLE = "UNAVAILABLE"


class ModelType(str, Enum):
    """Enum representing the possible model types."""

    LLM = "LLM"
    VLM = "VLM"
    TEXT_EMBEDDING = "Text_Embedding"
    REWARD_MODELING = "Reward_Modeling"


@dataclass
class LaunchResponse:
    """Response from launching a model."""

    slurm_job_id: int
    model_name: str
    config: dict[str, Any]
    raw_output: str = field(repr=False)


@dataclass
class StatusResponse:
    """Response from checking a model's status."""

    model_name: str
    server_status: ModelStatus
    job_state: Union[str, ModelStatus]
    raw_output: str = field(repr=False)
    base_url: Optional[str] = None
    pending_reason: Optional[str] = None
    failed_reason: Optional[str] = None


@dataclass
class MetricsResponse:
    """Response from retrieving model metrics."""

    model_name: str
    metrics: Union[dict[str, float], str]
    timestamp: float


@dataclass
class LaunchOptions:
    """Options for launching a model."""

    model_family: Optional[str] = None
    model_variant: Optional[str] = None
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    enable_prefix_caching: Optional[bool] = None
    enable_chunked_prefill: Optional[bool] = None
    max_num_batched_tokens: Optional[int] = None
    partition: Optional[str] = None
    num_nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    qos: Optional[str] = None
    time: Optional[str] = None
    vocab_size: Optional[int] = None
    data_type: Optional[str] = None
    venv: Optional[str] = None
    log_dir: Optional[str] = None
    model_weights_parent_dir: Optional[str] = None
    pipeline_parallelism: Optional[bool] = None
    compilation_config: Optional[str] = None
    enforce_eager: Optional[bool] = None


class LaunchOptionsDict(TypedDict):
    """TypedDict for LaunchOptions."""

    model_family: NotRequired[Optional[str]]
    model_variant: NotRequired[Optional[str]]
    max_model_len: NotRequired[Optional[int]]
    max_num_seqs: NotRequired[Optional[int]]
    gpu_memory_utilization: NotRequired[Optional[float]]
    enable_prefix_caching: NotRequired[Optional[bool]]
    enable_chunked_prefill: NotRequired[Optional[bool]]
    max_num_batched_tokens: NotRequired[Optional[int]]
    partition: NotRequired[Optional[str]]
    num_nodes: NotRequired[Optional[int]]
    gpus_per_node: NotRequired[Optional[int]]
    qos: NotRequired[Optional[str]]
    time: NotRequired[Optional[str]]
    vocab_size: NotRequired[Optional[int]]
    data_type: NotRequired[Optional[str]]
    venv: NotRequired[Optional[str]]
    log_dir: NotRequired[Optional[str]]
    model_weights_parent_dir: NotRequired[Optional[str]]
    pipeline_parallelism: NotRequired[Optional[bool]]
    compilation_config: NotRequired[Optional[str]]
    enforce_eager: NotRequired[Optional[bool]]


@dataclass
class ModelInfo:
    """Information about an available model."""

    name: str
    family: str
    variant: Optional[str]
    type: ModelType
    config: dict[str, Any]
