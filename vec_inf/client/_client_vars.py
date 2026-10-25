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
    Mapping of vLLM and SGLang Prometheus metrics to output keys and aggregation
KEY_METRICS_EXCLUDED_LABELS : dict
    Label values to skip when aggregating a metric
SGLANG_PRIMARY_RANK_LABELS : tuple
    SGLang rank labels whose non-zero ranks repeat rank 0's scheduler metrics
SLURM_JOB_CONFIG_ARGS : dict
    Mapping of SLURM configuration arguments to their parameter names
VLLM_SHORT_TO_LONG_MAP : dict
    Mapping of vLLM short arguments to their long names
"""

from pathlib import Path


MODEL_READY_SIGNATURE = "INFO:     Application startup complete."
SRC_DIR = str(Path(__file__).parent.parent)


# Key production metrics for inference servers.
#
# Keys are Prometheus sample names as exposed on the server's /metrics endpoint.
# Values are (output key, aggregation). A metric is exported once per label set
# (per engine for vLLM data parallel, per finish reason for request counts, per
# rank for SGLang), so every matching sample is combined with the aggregation:
# "sum" adds them and "mean" averages them.
#
# vLLM names match vllm/v1/metrics/loggers.py in vLLM 0.32.0. vLLM V1 no longer
# exports gpu_cache_usage_perc, cpu_cache_usage_perc, num_requests_swapped or
# the *_prefix_cache_hit_rate gauges; the prefix cache hit rate is derived from
# the prefix_cache_hits / prefix_cache_queries counters instead.
# SGLang names match sglang/srt/observability/metrics_collector.py in SGLang
# 0.5.12.post1. SGLang only serves /metrics when launched with --enable-metrics.
KEY_METRICS: dict[str, tuple[str, str]] = {
    # vLLM
    "vllm:prompt_tokens_total": ("total_prompt_tokens", "sum"),
    "vllm:generation_tokens_total": ("total_generation_tokens", "sum"),
    "vllm:e2e_request_latency_seconds_sum": ("request_latency_sum", "sum"),
    "vllm:e2e_request_latency_seconds_count": ("request_latency_count", "sum"),
    "vllm:request_queue_time_seconds_sum": ("queue_time_sum", "sum"),
    "vllm:request_queue_time_seconds_count": ("queue_time_count", "sum"),
    "vllm:request_success_total": ("successful_requests_total", "sum"),
    "vllm:num_requests_running": ("requests_running", "sum"),
    "vllm:num_requests_waiting": ("requests_waiting", "sum"),
    "vllm:num_preemptions_total": ("preemptions_total", "sum"),
    "vllm:kv_cache_usage_perc": ("kv_cache_usage", "mean"),
    "vllm:prefix_cache_queries_total": ("prefix_cache_queries", "sum"),
    "vllm:prefix_cache_hits_total": ("prefix_cache_hits", "sum"),
    # SGLang
    "sglang:prompt_tokens_total": ("total_prompt_tokens", "sum"),
    "sglang:generation_tokens_total": ("total_generation_tokens", "sum"),
    "sglang:e2e_request_latency_seconds_sum": ("request_latency_sum", "sum"),
    "sglang:e2e_request_latency_seconds_count": ("request_latency_count", "sum"),
    "sglang:queue_time_seconds_sum": ("queue_time_sum", "sum"),
    "sglang:queue_time_seconds_count": ("queue_time_count", "sum"),
    # Counts every finished request, including aborted ones
    "sglang:num_requests_total": ("finished_requests_total", "sum"),
    "sglang:num_running_reqs": ("requests_running", "sum"),
    "sglang:num_queue_reqs": ("requests_waiting", "sum"),
    "sglang:num_retracted_requests_total": ("preemptions_total", "sum"),
    "sglang:token_usage": ("kv_cache_usage", "mean"),
    "sglang:cache_hit_rate": ("prefix_cache_hit_rate", "mean"),
}

# Samples to skip when aggregating, as {metric name: {label: excluded values}}.
# vllm:request_success_total is labelled by finished_reason, and requests that
# were aborted by the client or hit an internal error are not successes.
KEY_METRICS_EXCLUDED_LABELS: dict[str, dict[str, set[str]]] = {
    "vllm:request_success_total": {"finished_reason": {"abort", "error"}},
}

# SGLang reports scheduler metrics once per tensor, pipeline and expert parallel
# rank (with pipeline parallelism, or --enable-metrics-for-all-schedulers), and
# every rank sees the same requests. Only samples from rank 0 of these groups
# are counted, so running/waiting requests are not multiplied by the number of
# ranks. Data parallel ranks (dp_rank) serve different requests and are summed.
SGLANG_PRIMARY_RANK_LABELS = ("tp_rank", "pp_rank", "moe_ep_rank")

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

# vLLM engine args mapping between short and long names (vLLM 0.32.0 aliases)
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
    "-dpm": "--data-parallel-multi-port-external-lb",
    "-asc": "--api-server-count",
    "-dc": "--diffusion-config",
    "-ep": "--enable-expert-parallel",
    "-sc": "--speculative-config",
    "-cc": "--compilation-config",
    "-ac": "--attention-config",
    # vLLM expands -O<n> (e.g. -O3) to --optimization-level <n>
    "-O": "--optimization-level",
    "-q": "--quantization",
}

# SGLang engine args mapping between short and long names. SGLang itself accepts
# --tp-size, --pp-size, --dp-size, --ep-size and --ep as aliases; --tp, --pp and
# --dp are vec-inf shorthands.
SGLANG_SHORT_TO_LONG_MAP = {
    "--tp": "--tensor-parallel-size",
    "--tp-size": "--tensor-parallel-size",
    "--pp": "--pipeline-parallel-size",
    "--pp-size": "--pipeline-parallel-size",
    "--dp": "--data-parallel-size",
    "--dp-size": "--data-parallel-size",
    "--ep": "--expert-parallel-size",
    "--ep-size": "--expert-parallel-size",
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
