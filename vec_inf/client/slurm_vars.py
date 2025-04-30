"""Slurm cluster configuration variables."""

from pathlib import Path

from typing_extensions import Literal


CACHED_CONFIG = Path("/", "model-weights", "vec-inf-shared", "models_latest.yaml")
LD_LIBRARY_PATH = "/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"
SINGULARITY_IMAGE = "/model-weights/vec-inf-shared/vector-inference_latest.sif"
SINGULARITY_LOAD_CMD = "module load singularity-ce/3.8.2"
VLLM_NCCL_SO_PATH = "/vec-inf/nccl/libnccl.so.2.18.1"
MAX_GPUS_PER_NODE = 8
MAX_NUM_NODES = 16
MAX_CPUS_PER_TASK = 128

QOS = Literal[
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
    "a100",
]

PARTITION = Literal[
    "a40",
    "a100",
    "t4v1",
    "t4v2",
    "rtx6000",
]

DEFAULT_ARGS = {
    "cpus_per_task": 16,
    "mem_per_node": "64G",
    "qos": "m2",
    "time": "08:00:00",
    "partition": "a40",
    "data_type": "auto",
    "log_dir": "~/.vec-inf-logs",
    "model_weights_parent_dir": "/model-weights",
}
