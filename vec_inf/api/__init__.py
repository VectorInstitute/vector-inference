"""Programmatic API for Vector Inference.

This module provides a Python API for interacting with Vector Inference.
It allows for launching and managing inference servers programmatically
without relying on the command-line interface.
"""

from vec_inf.api.client import VecInfClient
from vec_inf.api.models import (
    LaunchOptions,
    LaunchOptionsDict,
    LaunchResponse,
    MetricsResponse,
    ModelConfig,
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
    "ModelConfig",
    "MetricsResponse",
    "ModelStatus",
    "ModelType",
    "LaunchOptions",
    "LaunchOptionsDict",
]
