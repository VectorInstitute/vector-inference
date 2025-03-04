"""Model configuration."""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal


PARTITION = Literal["a40", "a100", "t4v1", "t4v2", "rtx6000"]

DATA_TYPE = Literal["auto", "float16", "bfloat16", "float32"]


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
        ..., gt=0, le=1_010_000, description="Maximum context length supported"
    )
    max_num_seqs: int = Field(
        default=256, gt=0, le=1024, description="Maximum concurrent request sequences"
    )
    pipeline_parallelism: bool = Field(
        default=True, description="Enable pipeline parallelism"
    )
    enforce_eager: bool = Field(default=False, description="Force eager mode execution")
    time: str = Field(
        default="08:00:00",
        pattern=r"^\d{2}:\d{2}:\d{2}$",
        description="HH:MM:SS time limit",
    )
    partition: Union[PARTITION, str] = Field(
        default="a100", description="GPU partition type"
    )
    data_type: Union[DATA_TYPE, str] = Field(
        default="auto", description="Model data type"
    )
    venv: str = Field(default="singularity", description="Virtual environment path")
    log_dir: str = Field(default="default", description="Slurm log directory path")
    model_weights_parent_dir: str = Field(
        default="/model-weights", description="Parent directory containing model weights"
    )

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, validate_default=True, frozen=True
    )
