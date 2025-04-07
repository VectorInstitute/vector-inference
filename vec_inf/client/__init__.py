"""Programmatic API for Vector Inference.

This module provides a Python API for launching and managing inference servers
using `vec_inf`. It is an alternative to the command-line interface, and allows
users direct control over the lifecycle of inference servers via python scripts.
"""

from vec_inf.client._models import (
    LaunchOptions,
    LaunchOptionsDict,
    LaunchResponse,
    MetricsResponse,
    ModelInfo,
    ModelStatus,
    ModelType,
    StatusResponse,
)
from vec_inf.client.api import VecInfClient


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
