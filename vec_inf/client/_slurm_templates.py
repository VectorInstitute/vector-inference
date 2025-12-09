"""SLURM script templates for Vector Inference.

This module contains the SLURM script templates for Vector Inference, including
single-node, multi-node, and batch mode templates.
"""

from typing import TypedDict

from vec_inf.client._slurm_vars import (
    CONTAINER_LOAD_CMD,
    CONTAINER_MODULE_NAME,
    PYTHON_VERSION,
)


CONTAINER_MODULE_NAME_UPPER = CONTAINER_MODULE_NAME.upper()


class ShebangConfig(TypedDict):
    """TypedDict for SLURM script shebang configuration.

    Parameters
    ----------
    base : str
        Base shebang line for all SLURM scripts
    multinode : list[str]
        Additional SLURM directives for multi-node configurations
    """

    base: str
    multinode: list[str]


class ServerSetupConfig(TypedDict):
    """TypedDict for server setup configuration.

    Parameters
    ----------
    single_node : list[str]
        Setup commands for single-node deployments
    multinode_vllm : list[str]
        Setup commands for multi-node vLLM deployments
    multinode_sglang : list[str]
        Setup commands for multi-node SGLang deployments
    """

    single_node: list[str]
    multinode_vllm: list[str]
    multinode_sglang: list[str]


class LaunchCmdConfig(TypedDict):
    """TypedDict for launch command configuration.

    Parameters
    ----------
    vllm : list[str]
        Launch commands for vLLM inference server
    sglang : list[str]
        Launch commands for SGLang inference server
    sglang_multinode : list[str]
        Launch commands for multi-node SGLang inference server
    """

    vllm: list[str]
    sglang: list[str]
    sglang_multinode: list[str]


class SlurmScriptTemplate(TypedDict):
    """TypedDict for complete SLURM script template configuration.

    Parameters
    ----------
    shebang : ShebangConfig
        Shebang and SLURM directive configuration
    container_setup : list[str]
        Commands for container setup
    imports : str
        Import statements and source commands
    bind_path : str
        Bind path environment variable for the container
    container_command : str
        Template for container execution command
    activate_venv : str
        Template for virtual environment activation
    server_setup : ServerSetupConfig
        Server initialization commands for different deployment modes
    find_server_port : list[str]
        Commands to find available ports for inference server
    write_to_json : list[str]
        Commands to write server configuration to JSON
    launch_cmd : LaunchCmdConfig
        Inference server launch commands
    """

    shebang: ShebangConfig
    container_setup: list[str]
    imports: str
    bind_path: str
    container_command: str
    activate_venv: str
    server_setup: ServerSetupConfig
    find_server_port: list[str]
    write_to_json: list[str]
    launch_cmd: LaunchCmdConfig


SLURM_SCRIPT_TEMPLATE: SlurmScriptTemplate = {
    "shebang": {
        "base": "#!/bin/bash",
        "multinode": [
            "#SBATCH --ntasks-per-node=1",
        ],
    },
    "container_setup": [
        CONTAINER_LOAD_CMD,
    ],
    "imports": "source {src_dir}/find_port.sh",
    "bind_path": f"export {CONTAINER_MODULE_NAME.upper()}_BINDPATH=${CONTAINER_MODULE_NAME.upper()}_BINDPATH,/dev,/tmp,{{model_weights_path}}{{additional_binds}}",
    "container_command": f"{CONTAINER_MODULE_NAME} exec --nv {{env_str}} --containall {{image_path}} \\",
    "activate_venv": "source {venv}/bin/activate",
    "server_setup": {
        "single_node": [
            "\n# Find available port",
            "head_node=${SLURMD_NODENAME}",
        ],
        "multinode_vllm": [
            "\n# Get list of nodes",
            'nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            "nodes_array=($nodes)",
            "head_node=${{nodes_array[0]}}",
            'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)',
            "\n# Check for RDMA devices and set environment variable accordingly",
            "if ! command -v ibv_devices >/dev/null 2>&1; then",
            '   echo "ibv_devices not found; forcing TCP. (No RDMA userland on host?)"',
            "   export NCCL_IB_DISABLE=1",
            '   export NCCL_ENV_ARG="--env NCCL_IB_DISABLE=1"',
            "else",
            "   # Pick GID index based on link layer (IB vs RoCE)",
            '   if ibv_devinfo 2>/dev/null | grep -q "link_layer:.*Ethernet"; then',
            "       # RoCEv2 typically needs a nonzero GID index; 3 is common, try 2 if your fabric uses it",
            "       export NCCL_IB_GID_INDEX={{NCCL_IB_GID_INDEX:-3}}",
            '       export NCCL_ENV_ARG="--env NCCL_IB_GID_INDEX={{NCCL_IB_GID_INDEX:-3}}"',
            "   else",
            "       # Native InfiniBand => GID 0",
            "       export NCCL_IB_GID_INDEX={{NCCL_IB_GID_INDEX:-0}}",
            '       export NCCL_ENV_ARG="--env NCCL_IB_GID_INDEX={{NCCL_IB_GID_INDEX:-0}}"',
            "   fi",
            "fi",
            "\n# Start Ray head node",
            "head_node_port=$(find_available_port $head_node 8080 65535)",
            "ray_head=$head_node_ip:$head_node_port",
            'echo "Ray Head IP: $ray_head"',
            'echo "Starting HEAD at $head_node"',
            'srun --nodes=1 --ntasks=1 -w "$head_node" \\',
            "    CONTAINER_PLACEHOLDER",
            '    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \\',
            '    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus {gpus_per_node} --block &',
            "sleep 10",
            "\n# Start Ray worker nodes",
            "worker_num=$((SLURM_JOB_NUM_NODES - 1))",
            "for ((i = 1; i <= worker_num; i++)); do",
            "    node_i=${{nodes_array[$i]}}",
            '    echo "Starting WORKER $i at $node_i"',
            '    srun --nodes=1 --ntasks=1 -w "$node_i" \\',
            "        CONTAINER_PLACEHOLDER",
            '        ray start --address "$ray_head" \\',
            '        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus {gpus_per_node} --block &',
            "    sleep 5",
            "done",
        ],
        "multinode_sglang": [
            "\n# Set NCCL initialization address using the hostname of the head node",
            'nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            "nodes_array=($nodes)",
            "head_node=${nodes_array[0]}",
            "NCCL_PORT=$(find_available_port $head_node 8000 65535)",
            'NCCL_INIT_ADDR="${head_node}:${NCCL_PORT}"',
            'echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"',
        ],
    },
    "find_server_port": [
        "\nserver_port_number=$(find_available_port $head_node 8080 65535)",
        'server_address="http://${head_node}:${server_port_number}/v1"',
    ],
    "write_to_json": [
        '\njson_path="{log_dir}/{model_name}.$SLURM_JOB_ID/{model_name}.$SLURM_JOB_ID.json"',
        'jq --arg server_addr "$server_address" \\',
        "    '. + {{\"server_address\": $server_addr}}' \\",
        '    "$json_path" > temp.json \\',
        '    && mv temp.json "$json_path"',
    ],
    "launch_cmd": {
        "vllm": [
            "vllm serve {model_weights_path} \\",
            "    --served-model-name {model_name} \\",
            '    --host "0.0.0.0" \\',
            "    --port $server_port_number \\",
        ],
        "sglang": [
            f"{PYTHON_VERSION} -m sglang.launch_server \\",
            "    --model-path {model_weights_path} \\",
            "    --served-model-name {model_name} \\",
            '    --host "0.0.0.0" \\',
            "    --port $server_port_number \\",
        ],
        "sglang_multinode": [
            "for ((i = 0; i < $SLURM_JOB_NUM_NODES; i++)); do",
            "    node_i=${{nodes_array[$i]}}",
            '    echo "Launching SGLang server on $node_i"',
            '    srun --ntasks=1 --nodes=1 -w "$node_i" \\',
            "    CONTAINER_PLACEHOLDER",
            f"       {PYTHON_VERSION} -m sglang.launch_server \\",
            "            --model-path {model_weights_path} \\",
            "            --served-model-name {model_name} \\",
            '            --host "0.0.0.0" \\',
            "            --port $server_port_number \\",
            '            --nccl-init-addr "$NCCL_INIT_ADDR" \\',
            "            --nnodes {num_nodes} \\",
            '            --node-rank "$i" \\',
            "SGLANG_ARGS_PLACEHOLDER &",
            "done",
            "\nwait",
        ],
    },
}


class BatchSlurmScriptTemplate(TypedDict):
    """TypedDict for batch SLURM script template configuration.

    Parameters
    ----------
    shebang : str
        Shebang line for the script
    hetjob : str
        SLURM directive for hetjob
    permission_update : str
        Command to update permissions of the script
    launch_model_scripts : list[str]
        Commands to run server launch scripts
    """

    shebang: str
    hetjob: str
    permission_update: str
    launch_model_scripts: list[str]


BATCH_SLURM_SCRIPT_TEMPLATE: BatchSlurmScriptTemplate = {
    "shebang": "#!/bin/bash",
    "hetjob": "#SBATCH hetjob\n",
    "permission_update": "chmod +x {script_name}",
    "launch_model_scripts": [
        "\nsrun --het-group={het_group_id} \\",
        "    --output={out_file} \\",
        "    --error={err_file} \\",
        "    {script_name} &\n",
    ],
}


class BatchModelLaunchScriptTemplate(TypedDict):
    """TypedDict for batch model launch script template configuration.

    Parameters
    ----------
    shebang : str
        Shebang line for the script
    container_setup : list[str]
        Commands for container setup
    bind_path : str
        Bind path environment variable for the container
    server_address_setup : list[str]
        Commands to setup the server address
    launch_cmd : list[str]
        Commands to launch the inference server
    container_command : str
        Commands to setup the container command
    """

    shebang: str
    container_setup: str
    bind_path: str
    server_address_setup: list[str]
    write_to_json: list[str]
    launch_cmd: dict[str, list[str]]
    container_command: str


BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE: BatchModelLaunchScriptTemplate = {
    "shebang": "#!/bin/bash\n",
    "container_setup": f"{CONTAINER_LOAD_CMD}\n",
    "bind_path": f"export {CONTAINER_MODULE_NAME.upper()}_BINDPATH=${CONTAINER_MODULE_NAME.upper()}_BINDPATH,/dev,/tmp,{{model_weights_path}}{{additional_binds}}",
    "server_address_setup": [
        "source {src_dir}/find_port.sh",
        "head_node_ip=${{SLURMD_NODENAME}}",
        "server_port_number=$(find_available_port $head_node_ip 8080 65535)",
        'server_address="http://${{head_node_ip}}:${{server_port_number}}/v1"\n',
        "echo $server_address\n",
    ],
    "write_to_json": [
        "het_job_id=$(($SLURM_JOB_ID+{het_group_id}))",
        'json_path="{log_dir}/{slurm_job_name}.$het_job_id/{model_name}.$het_job_id.json"',
        'jq --arg server_addr "$server_address" \\',
        "    '. + {{\"server_address\": $server_addr}}' \\",
        '    "$json_path" > temp_{model_name}.json \\',
        '    && mv temp_{model_name}.json "$json_path"\n',
    ],
    "container_command": f"{CONTAINER_MODULE_NAME} exec --nv --containall {{image_path}} \\",
    "launch_cmd": {
        "vllm": [
            "vllm serve {model_weights_path} \\",
            "    --served-model-name {model_name} \\",
            '    --host "0.0.0.0" \\',
            "    --port $server_port_number \\",
        ],
        "sglang": [
            f"{PYTHON_VERSION} -m sglang.launch_server \\",
            "    --model-path {model_weights_path} \\",
            "    --served-model-name {model_name} \\",
            '    --host "0.0.0.0" \\',
            "    --port $server_port_number \\",
        ],
    },
}
