"""Programmatic API for Vector Inference.

This module provides a Python API for launching and managing inference servers
using `vec_inf`. It is an alternative to the command-line interface, and allows
users direct control over the lifecycle of inference servers via python scripts.
"""

from vec_inf.client.api import VecInfClient
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import (
    LaunchOptions,
    LaunchResponse,
    MetricsResponse,
    ModelInfo,
    ModelStatus,
    ModelType,
    StatusResponse,
)


__all__ = [
    "VecInfClient",
    "LaunchResponse",
    "StatusResponse",
    "ModelInfo",
    "MetricsResponse",
    "ModelStatus",
    "ModelType",
    "LaunchOptions",
    "ModelConfig",
]
