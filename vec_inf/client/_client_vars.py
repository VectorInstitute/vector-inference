"""Global variables for Vector Inference.

This module contains configuration constants and templates used throughout the
Vector Inference package, including model configurations, and metric definitions.

Constants
---------
MODEL_READY_SIGNATURE : str
    Signature string indicating successful model server startup
SRC_DIR : str
    Absolute path to the package source directory
KEY_METRICS : dict
    Mapping of vLLM metrics to their human-readable names
SLURM_JOB_CONFIG_ARGS : dict
    Mapping of SLURM configuration arguments to their parameter names
VLLM_SHORT_TO_LONG_MAP : dict
    Mapping of vLLM short arguments to their long names
"""

from pathlib import Path


MODEL_READY_SIGNATURE = "INFO:     Application startup complete."
SRC_DIR = str(Path(__file__).parent.parent)


# Key production metrics for inference servers
KEY_METRICS = {
    "vllm:prompt_tokens_total": "total_prompt_tokens",
    "vllm:generation_tokens_total": "total_generation_tokens",
    "vllm:e2e_request_latency_seconds_sum": "request_latency_sum",
    "vllm:e2e_request_latency_seconds_count": "request_latency_count",
    "vllm:request_queue_time_seconds_sum": "queue_time_sum",
    "vllm:request_success_total": "successful_requests_total",
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
    "vllm:num_requests_swapped": "requests_swapped",
    "vllm:gpu_cache_usage_perc": "gpu_cache_usage",
    "vllm:cpu_cache_usage_perc": "cpu_cache_usage",
}

# Slurm job configuration arguments
SLURM_JOB_CONFIG_ARGS = {
    "job-name": "model_name",
    "partition": "partition",
    "account": "account",
    "chdir": "work_dir",
    "qos": "qos",
    "time": "time",
    "nodes": "num_nodes",
    "exclude": "exclude",
    "nodelist": "nodelist",
    "gres": "gres",
    "cpus-per-task": "cpus_per_task",
    "mem": "mem_per_node",
    "output": "out_file",
    "error": "err_file",
}

# vLLM engine args mapping between short and long names
VLLM_SHORT_TO_LONG_MAP = {
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
    "-n": "--nnodes",
    "-r": "--node-rank",
    "-dcp": "--decode-context-parallel-size",
    "-pcp": "--prefill-context-parallel-size",
    "-dp": "--data-parallel-size",
    "-dpn": "--data-parallel-rank",
    "-dpr": "--data-parallel-start-rank",
    "-dpl": "--data-parallel-size-local",
    "-dpa": "--data-parallel-address",
    "-dpp": "--data-parallel-rpc-port",
    "-dpb": "--data-parallel-backend",
    "-dph": "--data-parallel-hybrid-lb",
    "-dpe": "--data-parallel-external-lb",
    "-O": "--compilation-config",
    "-q": "--quantization",
}

# SGLang engine args mapping between short and long names
SGLANG_SHORT_TO_LONG_MAP = {
    "--tp": "--tensor-parallel-size",
    "--tp-size": "--tensor-parallel-size",
    "--pp": "--pipeline-parallel-size",
    "--pp-size": "--pipeline-parallel-size",
    "--dp": "--data-parallel-size",
    "--dp-size": "--data-parallel-size",
    "--ep": "--expert-parallel-size",
    "--ep-size": "--expert-parallel-expert-size",
}

# Mapping of engine short names to their argument mappings
ENGINE_SHORT_TO_LONG_MAP = {
    "vllm": VLLM_SHORT_TO_LONG_MAP,
    "sglang": SGLANG_SHORT_TO_LONG_MAP,
}

# Required matching arguments for batch mode
BATCH_MODE_REQUIRED_MATCHING_ARGS = ["venv", "log_dir"]

# Supported engines
SUPPORTED_ENGINES = ["vllm", "sglang"]
