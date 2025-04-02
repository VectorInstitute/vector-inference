"""Global variables for the vector inference package."""

from pathlib import Path

MODEL_READY_SIGNATURE = "INFO:     Application startup complete."
CACHED_CONFIG = Path("/", "model-weights", "vec-inf-shared", "models.yaml")
SRC_DIR = str(Path(__file__).parent.parent)
LD_LIBRARY_PATH = "/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"

# Maps model types to vLLM tasks
VLLM_TASK_MAP = {
    "LLM": "generate",
    "VLM": "generate",
    "TEXT_EMBEDDING": "embed",
    "REWARD_MODELING": "reward",
}  

# Required fields for model configuration
REQUIRED_FIELDS = {
    "model_family",
    "model_type",
    "gpus_per_node",
    "num_nodes",
    "vocab_size",
    "max_model_len",
}

# Boolean fields for model configuration
BOOLEAN_FIELDS = {
    "pipeline_parallelism",
    "enforce_eager",
    "enable_prefix_caching",
    "enable_chunked_prefill",
}