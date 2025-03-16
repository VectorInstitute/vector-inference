"""Programmatic API for Vector Inference.

This module provides a Python API for launching and managing inference servers
using `vec_inf`. It is an alternative to the command-line interface, and allows
users direct control over the lifecycle of inference servers via python scripts.
"""

from vec_inf.api.client import VecInfClient
from vec_inf.api.models import (
    LaunchOptions,
    LaunchOptionsDict,
    LaunchResponse,
    MetricsResponse,
    ModelInfo,
    StatusResponse,
)
from vec_inf.shared.models import ModelStatus, ModelType


__all__ = [
    "VecInfClient",
    "LaunchResponse",
    "StatusResponse",
    "ModelInfo",
    "MetricsResponse",
    "ModelStatus",
    "ModelType",
    "LaunchOptions",
    "LaunchOptionsDict",
]
