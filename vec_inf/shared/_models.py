"""Shared data models for Vector Inference."""

from enum import Enum


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
