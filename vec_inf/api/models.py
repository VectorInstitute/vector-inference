"""Data models for Vector Inference API.

This module contains the data model classes used by the Vector Inference API
for both request parameters and response objects.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypedDict

from typing_extensions import NotRequired

from vec_inf.shared.models import ModelStatus, ModelType


@dataclass
class ModelInfo:
    """Information about an available model."""

    name: str
    family: str
    variant: Optional[str]
    type: ModelType
    config: Dict[str, Any]


@dataclass
class LaunchResponse:
    """Response from launching a model."""

    slurm_job_id: str
    model_name: str
    config: Dict[str, Any]
    raw_output: str = field(repr=False)


@dataclass
class StatusResponse:
    """Response from checking a model's status."""

    slurm_job_id: str
    model_name: str
    status: ModelStatus
    raw_output: str = field(repr=False)
    base_url: Optional[str] = None
    pending_reason: Optional[str] = None
    failed_reason: Optional[str] = None


@dataclass
class MetricsResponse:
    """Response from retrieving model metrics."""

    slurm_job_id: str
    model_name: str
    metrics: Dict[str, str]
    timestamp: float
    raw_output: str = field(repr=False)


@dataclass
class LaunchOptions:
    """Options for launching a model."""

    model_family: Optional[str] = None
    model_variant: Optional[str] = None
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    partition: Optional[str] = None
    num_nodes: Optional[int] = None
    num_gpus: Optional[int] = None
    qos: Optional[str] = None
    time: Optional[str] = None
    vocab_size: Optional[int] = None
    data_type: Optional[str] = None
    venv: Optional[str] = None
    log_dir: Optional[str] = None
    model_weights_parent_dir: Optional[str] = None
    pipeline_parallelism: Optional[bool] = None
    enforce_eager: Optional[bool] = None


class LaunchOptionsDict(TypedDict):
    """TypedDict for LaunchOptions."""

    model_family: NotRequired[Optional[str]]
    model_variant: NotRequired[Optional[str]]
    max_model_len: NotRequired[Optional[int]]
    max_num_seqs: NotRequired[Optional[int]]
    partition: NotRequired[Optional[str]]
    num_nodes: NotRequired[Optional[int]]
    num_gpus: NotRequired[Optional[int]]
    qos: NotRequired[Optional[str]]
    time: NotRequired[Optional[str]]
    vocab_size: NotRequired[Optional[int]]
    data_type: NotRequired[Optional[str]]
    venv: NotRequired[Optional[str]]
    log_dir: NotRequired[Optional[str]]
    model_weights_parent_dir: NotRequired[Optional[str]]
    pipeline_parallelism: NotRequired[Optional[bool]]
    enforce_eager: NotRequired[Optional[bool]]
