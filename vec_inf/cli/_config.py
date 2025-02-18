"""Model configuration."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Pydantic model for validating and managing model deployment configurations."""

    model_name: str = Field(..., min_length=3, pattern=r"^[a-zA-Z0-9\-_\.]+$")
    model_family: str = Field(..., min_length=2)
    model_variant: Optional[str] = Field(
        default=None, description="Specific variant/version of the model family"
    )
    model_type: Literal["LLM", "VLM", "Text_Embedding", "Reward_Modeling"] = Field(
        ..., description="Type of model architecture"
    )
    num_gpus: int = Field(..., gt=0, le=8, description="GPUs per node")
    num_nodes: int = Field(..., gt=0, le=16, description="Number of nodes")
    vocab_size: int = Field(..., gt=0, le=1_000_000)
    max_model_len: int = Field(
        ..., gt=0, le=131_072, description="Maximum context length supported"
    )
    max_num_seqs: int = Field(
        default=256, gt=0, le=1024, description="Maximum concurrent request sequences"
    )
    pipeline_parallelism: bool = Field(
        default=True, description="Enable pipeline parallelism"
    )
    enforce_eager: bool = Field(default=False, description="Force eager mode execution")
    qos: Literal[
        "normal",
        "m",
        "m2",
        "m3",
        "m4",
        "m5",
        "long",
        "deadline",
        "high",
        "scavenger",
        "llm",
    ] = Field(default="m2", description="Quality of Service tier")
    time: str = Field(
        default="08:00:00",
        pattern=r"^\d{2}:\d{2}:\d{2}$",
        description="HH:MM:SS time limit",
    )
    partition: Literal["a40", "t4v1", "t4v2", "rtx6000"] = Field(
        default="a40", description="GPU partition type"
    )
    data_type: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto", description="Model precision format"
    )
    venv: str = Field(
        default="singularity", description="Virtual environment/container system"
    )
    log_dir: Path = Field(
        default=Path("~/.vec-inf-logs").expanduser(), description="Log directory path"
    )
    model_weights_parent_dir: Path = Field(
        default=Path("/model-weights"), description="Base directory for model weights"
    )

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, validate_default=True, frozen=True
    )

    @field_validator("log_dir", "model_weights_parent_dir", mode="after")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """Ensure paths are resolved and absolute."""
        if not v.is_absolute():
            v = v.resolve()
        return v
