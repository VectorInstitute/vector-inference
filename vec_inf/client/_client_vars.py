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
    "qos": "qos",
    "time": "time",
    "nodes": "num_nodes",
    "exclude": "exclude",
    "nodelist": "node_list",
    "gpus-per-node": "gpus_per_node",
    "cpus-per-task": "cpus_per_task",
    "mem": "mem_per_node",
    "output": "out_file",
    "error": "err_file",
}

# vLLM engine args mapping between short and long names
VLLM_SHORT_TO_LONG_MAP = {
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
    "-dp": "--data-parallel-size",
    "-dpl": "--data-parallel-size-local",
    "-dpa": "--data-parallel-address",
    "-dpp": "--data-parallel-rpc-port",
    "-O": "--compilation-config",
    "-q": "--quantization",
}

# Required matching arguments for batch mode
BATCH_MODE_REQUIRED_MATCHING_ARGS = ["venv", "log_dir"]
