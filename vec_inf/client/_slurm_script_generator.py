"""Class for generating SLURM scripts to run vLLM servers."""

from datetime import datetime
from pathlib import Path
from typing import Any

from vec_inf.client._vars import VLLM_TASK_MAP


class SlurmScriptGenerator:
    """A class to generate SLURM scripts for running vLLM servers.

    This class handles the generation of SLURM scripts for both single-node and
    multi-node configurations, supporting different virtualization environments
    (venv or singularity).

    Args:
        params (dict[str, Any]): Configuration parameters for the SLURM script
        src_dir (str): Source directory path containing necessary scripts
    """

    def __init__(self, params: dict[str, Any], src_dir: str):
        """Initialize the SlurmScriptGenerator with configuration parameters.

        Args:
            params (dict[str, Any]): Configuration parameters for the SLURM script
            src_dir (str): Source directory path containing necessary scripts
        """
        self.params = params
        self.src_dir = src_dir
        self.is_multinode = int(self.params["num_nodes"]) > 1
        self.model_weights_path = str(
            Path(params["model_weights_parent_dir"], params["model_name"])
        )
        self.task = VLLM_TASK_MAP[self.params["model_type"]]

    def _generate_script_content(self) -> str:
        """Generate the complete SLURM script content.

        Returns
        -------
            str: The complete SLURM script as a string
        """
        preamble = self._generate_preamble()
        server = self._generate_server_script()
        launcher = self._generate_launcher()
        args = self._generate_shared_args()
        return preamble + server + launcher + args

    def _generate_preamble(self) -> str:
        """Generate the SLURM script preamble with job specifications.

        Returns
        -------
            str: SLURM preamble containing resource requests and job parameters
        """
        base = [
            "#!/bin/bash",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=64G",
        ]
        if self.is_multinode:
            base += [
                "#SBATCH --exclusive",
                "#SBATCH --tasks-per-node=1",
            ]
        base += [""]
        return "\n".join(base)

    def _generate_shared_args(self) -> str:
        """Generate the command-line arguments for the vLLM server.

        Handles both single-node and multi-node configurations, setting appropriate
        parallel processing parameters based on the configuration.

        Returns
        -------
            str: Command-line arguments for the vLLM server
        """
        if self.is_multinode and not self.params["pipeline_parallelism"]:
            tensor_parallel_size = (
                self.params["num_nodes"] * self.params["gpus_per_node"]
            )
            pipeline_parallel_size = 1
        else:
            tensor_parallel_size = self.params["gpus_per_node"]
            pipeline_parallel_size = self.params["num_nodes"]

        args = [
            f"--model {self.model_weights_path} \\",
            f"--served-model-name {self.params['model_name']} \\",
            '--host "0.0.0.0" \\',
            "--port $vllm_port_number \\",
            f"--tensor-parallel-size {tensor_parallel_size} \\",
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
            args.insert(4, f"--pipeline-parallel-size {pipeline_parallel_size} \\")
        if self.params.get("max_num_batched_tokens"):
            args.append(
                f"--max-num-batched-tokens={self.params['max_num_batched_tokens']} \\"
            )
        if self.params.get("enable_prefix_caching") == "True":
            args.append("--enable-prefix-caching \\")
        if self.params.get("enable_chunked_prefill") == "True":
            args.append("--enable-chunked-prefill \\")
        if self.params.get("enforce_eager") == "True":
            args.append("--enforce-eager")

        return "\n".join(args)

    def _generate_server_script(self) -> str:
        """Generate the server initialization script.

        Creates the script section that handles server setup, including Ray
        initialization for multi-node setups and port configuration.

        Returns
        -------
            str: Server initialization script content
        """
        server_script = [""]
        if self.params["venv"] == "singularity":
            server_script.append(
                "module load singularity-ce/3.8.2\n"
                "singularity exec $SINGULARITY_IMAGE ray stop\n"
            )
        server_script.append(f"source {self.src_dir}/find_port.sh\n")
        server_script.append(
            self._generate_multinode_server_script()
            if self.is_multinode
            else self._generate_single_node_server_script()
        )
        server_script.append(
            f'json_path="{self.params["log_dir"]}/{self.params["model_name"]}.$SLURM_JOB_ID/{self.params["model_name"]}.$SLURM_JOB_ID.json"\n'
            'jq --arg server_addr "$server_address" \\\n'
            '    \'. + {{"server_address": $server_addr}}\' \\\n'
            '    "$json_path" > temp.json \\\n'
            '    && mv temp.json "$json_path"\n\n'
        )
        return "\n".join(server_script)

    def _generate_single_node_server_script(self) -> str:
        """Generate the server script for single-node deployment.

        Returns
        -------
            str: Script content for single-node server setup
        """
        return (
            "hostname=${SLURMD_NODENAME}\n"
            "vllm_port_number=$(find_available_port ${hostname} 8080 65535)\n\n"
            'server_address="http://${hostname}:${vllm_port_number}/v1"\n'
            'echo "Server address: $server_address"\n'
        )

    def _generate_multinode_server_script(self) -> str:
        """Generate the server script for multi-node deployment.

        Creates a script that initializes Ray cluster with head and worker nodes,
        configuring networking and GPU resources appropriately.

        Returns
        -------
            str: Script content for multi-node server setup
        """
        server_script = []
        server_script.append(
            'nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")\n'
            "nodes_array=($nodes)\n\n"
            "head_node=${nodes_array[0]}\n"
            'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)\n\n'
            "head_node_port=$(find_available_port $head_node_ip 8080 65535)\n\n"
            "ip_head=$head_node_ip:$head_node_port\n"
            "export ip_head\n"
            'echo "IP Head: $ip_head"\n\n'
            'echo "Starting HEAD at $head_node"\n'
            'srun --nodes=1 --ntasks=1 -w "$head_node" \\'
        )

        if self.params["venv"] == "singularity":
            server_script.append(
                f"    singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} "
                "--containall $SINGULARITY_IMAGE \\"
            )

        server_script.append(
            '    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \\\n'
            '    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &\n\n'
            "sleep 10\n"
            "worker_num=$((SLURM_JOB_NUM_NODES - 1))\n\n"
            "for ((i = 1; i <= worker_num; i++)); do\n"
            "    node_i=${nodes_array[$i]}\n"
            '    echo "Starting WORKER $i at $node_i"\n'
            '    srun --nodes=1 --ntasks=1 -w "$node_i" \\'
        )

        if self.params["venv"] == "singularity":
            server_script.append(
                f"        singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} "
                "--containall $SINGULARITY_IMAGE \\"
            )

        server_script.append(
            '        ray start --address "$ip_head" \\\n'
            '        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &\n'
            "    sleep 5\n"
            "done\n\n"
            "vllm_port_number=$(find_available_port $head_node_ip 8080 65535)\n\n"
            'server_address="http://${head_node_ip}:${vllm_port_number}/v1"\n'
            'echo "Server address: $server_address"\n\n'
        )
        return "\n".join(server_script)

    def _generate_launcher(self) -> str:
        """Generate the vLLM server launch command.

        Creates the command to launch the vLLM server, handling different virtualization
        environments (venv or singularity).

        Returns
        -------
            str: Server launch command
        """
        if self.params["venv"] == "singularity":
            launcher_script = [
                f"""singularity exec --nv --bind {self.model_weights_path}:{self.model_weights_path} --containall $SINGULARITY_IMAGE \\"""
            ]
        else:
            launcher_script = [f"""source {self.params["venv"]}/bin/activate"""]
        launcher_script.append(
            """python3.10 -m vllm.entrypoints.openai.api_server \\\n"""
        )
        return "\n".join(launcher_script)

    def write_to_log_dir(self) -> Path:
        """Write the generated SLURM script to the log directory.

        Creates a timestamped script file in the configured log directory.

        Returns
        -------
            Path: Path to the generated SLURM script file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path: Path = (
            Path(self.params["log_dir"])
            / f"launch_{self.params['model_name']}_{timestamp}.slurm"
        )

        content = self._generate_script_content()
        script_path.write_text(content)
        return script_path
