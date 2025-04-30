"""Global variables for the vector inference package."""

from pathlib import Path
from typing import TypedDict
from vec_inf.client.slurm_vars import SINGULARITY_LOAD_CMD


MODEL_READY_SIGNATURE = "INFO:     Application startup complete."
SRC_DIR = str(Path(__file__).parent.parent)


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
}

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
    "qos": "qos",
    "time": "time",
    "nodes": "num_nodes",
    "gpus-per-node": "gpus_per_node",
    "cpus-per-task": "cpus_per_task",
    "mem": "mem_per_node",
    "output": "out_file",
    "error": "err_file",
}

# Slurm script templates
class ShebangConfig(TypedDict):
    base: str
    multinode: list[str]


class ServerSetupConfig(TypedDict):
    single_node: list[str]
    multinode: list[str]


class SlurmScriptTemplate(TypedDict):
    shebang: ShebangConfig
    singularity_setup: list[str]
    imports: str
    singularity_command: str
    activate_venv: str
    server_setup: ServerSetupConfig
    find_vllm_port: list[str]
    write_to_json: list[str]
    launch_cmd: list[str]


SLURM_SCRIPT_TEMPLATE: SlurmScriptTemplate = {
    "shebang": {
        "base": "#!/bin/bash",
        "multinode": [
            "#SBATCH --exclusive",
            "#SBATCH --tasks-per-node=1",
        ],
    },
    "singularity_setup": [
        SINGULARITY_LOAD_CMD,
        "singularity exec $SINGULARITY_IMAGE ray stop",
    ],
    "imports": "source {src_dir}/find_port.sh",
    "singularity_command": "singularity exec --nv --bind {model_weights_path}:{model_weights_path} --containall $SINGULARITY_IMAGE",
    "activate_venv": "source {venv}/bin/activate",
    "server_setup": {
        "single_node": [
            "\n# Find available port",
            "head_node_ip=${SLURMD_NODENAME}",
        ],
        "multinode": [
            "\n# Get list of nodes",
            'nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            "nodes_array=($nodes)",
            "head_node=${nodes_array[0]}",
            'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)',
            "\n# Start Ray head node",
            "head_node_port=$(find_available_port $head_node_ip 8080 65535)",
            "ray_head=$head_node_ip:$head_node_port",
            'echo "Ray Head IP: $ray_head"',
            'echo "Starting HEAD at $head_node"',
            'srun --nodes=1 --ntasks=1 -w "$head_node" \\',
            "    SINGULARITY_PLACEHOLDER \\",
            '    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \\',
            '    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &',
            "sleep 10",
            "\n# Start Ray worker nodes",
            "worker_num=$((SLURM_JOB_NUM_NODES - 1))",
            "for ((i = 1; i <= worker_num; i++)); do",
            "    node_i=${nodes_array[$i]}",
            '    echo "Starting WORKER $i at $node_i"',
            '    srun --nodes=1 --ntasks=1 -w "$node_i" \\',
            "        SINGULARITY_PLACEHOLDER \\",
            '        ray start --address "$ray_head" \\',
            '        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &',
            "    sleep 5",
            "done",
        ],
    },
    "find_vllm_port": [
        "\nvllm_port_number=$(find_available_port $head_node_ip 8080 65535)",
        'server_address="http://${head_node_ip}:${vllm_port_number}/v1"',
    ],
    "write_to_json": [
        '\njson_path="{log_dir}/{model_name}.$SLURM_JOB_ID/{model_name}.$SLURM_JOB_ID.json"',
        'jq --arg server_addr "$server_address" \\',
        "    '. + {{\"server_address\": $server_addr}}' \\",
        '    "$json_path" > temp.json \\',
        '    && mv temp.json "$json_path"',
    ],
    "launch_cmd": [
        "python3.10 -m vllm.entrypoints.openai.api_server \\",
        "    --model {model_weights_path} \\",
        "    --served-model-name {model_name} \\",
        '    --host "0.0.0.0" \\',
        "    --port $vllm_port_number \\",
        "    --trust-remote-code \\",
    ],
}
