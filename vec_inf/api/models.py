"""Data models for Vector Inference API.

This module contains the data model classes used by the Vector Inference API
for both request parameters and response objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ModelStatus(str, Enum):
    """Enum representing the possible status states of a model."""

    PENDING = "PENDING"
    LAUNCHING = "LAUNCHING"
    READY = "READY"
    FAILED = "FAILED"
    SHUTDOWN = "SHUTDOWN"


class ModelType(str, Enum):
    """Enum representing the possible model types."""

    LLM = "LLM"
    VLM = "VLM"
    TEXT_EMBEDDING = "Text_Embedding"
    REWARD_MODELING = "Reward_Modeling"


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    model_name: str
    model_family: str
    model_variant: Optional[str] = None
    model_type: ModelType = ModelType.LLM
    num_gpus: int = 1
    num_nodes: int = 1
    vocab_size: int = 0
    max_model_len: int = 0
    max_num_seqs: int = 256
    pipeline_parallelism: bool = True
    enforce_eager: bool = False
    qos: str = "m2"
    time: str = "08:00:00"
    partition: str = "a40"
    data_type: str = "auto"
    venv: str = "singularity"
    log_dir: Optional[Path] = None
    model_weights_parent_dir: Optional[Path] = None


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
