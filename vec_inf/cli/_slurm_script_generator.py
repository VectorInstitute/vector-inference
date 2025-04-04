from pathlib import Path

VLLM_TASK_MAP = {
    "LLM": "generate",
    "VLM": "generate",
    "Text_Embedding": "embed",
    "Reward_Modeling": "reward",
}

class SlurmScriptGenerator:
    def __init__(self, params: dict, src_dir: str, is_multinode: bool = False):
        self.params = params
        self.src_dir = src_dir
        self.is_multinode = is_multinode
        self.model_weights_path = Path(params["model_weights_parent_dir"], params["model_name"])
        self.task = VLLM_TASK_MAP[self.params["model_type"]]

    def _generate_script_content(self) -> str:
        return self._generate_multinode_script() if self.is_multinode else self._generate_single_node_script()

    def _generate_preamble(self, is_multinode: bool = False) -> str:
        base = [
            "#!/bin/bash",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=64G",
        ]
        if is_multinode:
            base += [
                "#SBATCH --exclusive",
                "#SBATCH --tasks-per-node=1",
            ]
        base += [f"source {self.src_dir}/find_port.sh", ""]
        return "\n".join(base)

    def _export_parallel_vars(self) -> str:
        if self.is_multinode:
            return """if [ "$PIPELINE_PARALLELISM" = "True" ]; then
export PIPELINE_PARALLEL_SIZE=$SLURM_JOB_NUM_NODES
export TENSOR_PARALLEL_SIZE=$SLURM_GPUS_PER_NODE
else
export PIPELINE_PARALLEL_SIZE=1
export TENSOR_PARALLEL_SIZE=$((SLURM_JOB_NUM_NODES*SLURM_GPUS_PER_NODE))
fi
"""
        else:
            return "export TENSOR_PARALLEL_SIZE=$SLURM_GPUS_PER_NODE\n"

    def _generate_shared_args(self) -> list[str]:
        args = [
            f"--model {self.model_weights_path} \\",
            f"--served-model-name {self.params['model_name']} \\",
            "--host \"0.0.0.0\" \\",
            "--port $vllm_port_number \\",
            "--tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \\",
            f"--dtype {self.params['data_type']} \\",
            "--trust-remote-code \\",
            f"--max-logprobs {self.params['vocab_size']} \\",
            f"--max-model-len {self.params['max_model_len']} \\",
            f"--max-num-seqs {self.params['max_num_seqs']} \\",
            f"--gpu-memory-utilization {self.params['gpu_memory_utilization']} \\",
            f"--compilation-config {self.params['compilation_config']} \\",
            f"--task {self.task} \\",
        ]
        if self.is_multinode:
            args.insert(4, "--pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \\")
        if self.params.get("max_num_batched_tokens"):
            args.append(f"--max-num-batched-tokens={self.params['max_num_batched_tokens']} \\")
        if self.params.get("enable_prefix_caching") == "True":
            args.append("--enable-prefix-caching \\")
        if self.params.get("enable_chunked_prefill") == "True":
            args.append("--enable-chunked-prefill \\")
        if self.params.get("enforce_eager") == "True":
            args.append("--enforce-eager")

        return args

    def _generate_single_node_script(self) -> str:
        preamble = self._generate_preamble(is_multinode=False)

        server = f"""hostname=${{SLURMD_NODENAME}}
vllm_port_number=$(find_available_port ${{hostname}} 8080 65535)

SERVER_ADDR="http://${{hostname}}:${{vllm_port_number}}/v1"
echo "Server address: $SERVER_ADDR"

JSON_PATH="{self.params['log_dir']}/{self.params['model_name']}.$SLURM_JOB_ID/{self.params['model_name']}.$SLURM_JOB_ID.json"
echo "Updating server address in $JSON_PATH"
jq --arg server_addr "$SERVER_ADDR" \\
    '. + {{"server_address": $server_addr}}' \\
    "$JSON_PATH" > temp.json \\
    && mv temp.json "$JSON_PATH" \\
    && rm -f temp.json
"""

        env_exports = self._export_parallel_vars()

        if self.params["venv"] == "singularity":
            launcher = f"""export SINGULARITY_IMAGE=/model-weights/vec-inf-shared/vector-inference_latest.sif
export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
module load singularity-ce/3.8.2
singularity exec $SINGULARITY_IMAGE ray stop
singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} $SINGULARITY_IMAGE \\
python3.10 -m vllm.entrypoints.openai.api_server \\
"""
        else:
            launcher = f"""source {self.params['venv']}/bin/activate
python3 -m vllm.entrypoints.openai.api_server \\
"""

        args = "\n".join(self._generate_shared_args())
        return preamble + server + env_exports + launcher + args
    
    
    def _generate_multinode_script(self) -> str:
        preamble = self._generate_preamble(is_multinode=True)

        cluster_setup = []
        if self.params["venv"] == "singularity":
            cluster_setup.append(f"""export SINGULARITY_IMAGE=/model-weights/vec-inf-shared/vector-inference_latest.sif
export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
module load singularity-ce/3.8.2
singularity exec $SINGULARITY_IMAGE ray stop
""")

        cluster_setup.append(f"""nodes=$(scontrol show hostnames "${{SLURM_JOB_NODELIST}}")
nodes_array=(${{nodes}})

head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

head_node_port=$(find_available_port $head_node_ip 8080 65535)
vllm_port_number=$(find_available_port $head_node_ip 8080 65535)

ip_head=$head_node_ip:$head_node_port
export ip_head
echo "IP Head: $ip_head"
                             
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \\""")

        if self.params["venv"] == "singularity":
            cluster_setup.append(f"""    singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} $SINGULARITY_IMAGE \\""")

        cluster_setup.append(f"""    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \\
    --num-cpus "${{SLURM_CPUS_PER_TASK}}" --num-gpus "${{SLURM_GPUS_PER_NODE}}" --block &

sleep 10
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${{nodes_array[$i]}}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \\""")

        if self.params["venv"] == "singularity":
            cluster_setup.append(f"""        singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} $SINGULARITY_IMAGE \\""")
        cluster_setup.append(f"""        ray start --address "$ip_head" \\
        --num-cpus "${{SLURM_CPUS_PER_TASK}}" --num-gpus "${{SLURM_GPUS_PER_NODE}}" --block &
    sleep 5
done

SERVER_ADDR="http://$head_node_ip:$vllm_port_number/v1"
echo "Server address: $SERVER_ADDR"

JSON_PATH="{self.params['log_dir']}/{self.params['model_name']}.$SLURM_JOB_ID/{self.params['model_name']}.$SLURM_JOB_ID.json"
echo "Updating server address in $JSON_PATH"
jq --arg server_addr "$SERVER_ADDR" \\
    '. + {{"server_address": $server_addr}}' \\
    "$JSON_PATH" > temp.json \\
    && mv temp.json "$JSON_PATH" \\
    && rm -f temp.json                       
""")
        cluster_setup = "\n".join(cluster_setup)
        env_exports = self._export_parallel_vars()


        if self.params["venv"] == "singularity":
            launcher = f"""singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} $SINGULARITY_IMAGE \\
python3.10 -m vllm.entrypoints.openai.api_server \\
"""
        else:
            launcher = f"""source {self.params['venv']}/bin/activate
python3 -m vllm.entrypoints.openai.api_server \\
"""

        args = "\n".join(self._generate_shared_args())
        return preamble + cluster_setup + env_exports + launcher + args

    def write_to_log_dir(self) -> Path:
        log_subdir = Path(self.params["log_dir"]) / self.params["model_name"]
        log_subdir.mkdir(parents=True, exist_ok=True)
        script_path = log_subdir / "launch.slurm"
        content = self._generate_script_content()
        script_path.write_text(content)
        return script_path
