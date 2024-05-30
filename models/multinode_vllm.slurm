#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --tasks-per-node=1

# NOTE: Instead of invoking this script directly via sbatch, use `bash openai_entrypoint.sh`
# to keep track of environment configurations.

# Load CUDA, change to the cuda version on your environment if different
module load cuda-12.3
nvidia-smi

# Activate vllm venv
source ${VENV_BASE}/bin/activate

source $(dirname ${MODEL_DIR})/find_port.sh

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Find port for head node
head_node_port=$(find_available_port $head_node_ip 8080 65535)

# Starting the Ray head node
ip_head=$head_node_ip:$head_node_port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &

# Starting the Ray worker nodes
# Optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
    sleep 5
done


vllm_port_number=$(find_available_port $head_node_ip 8080 65535)

echo "Server address: http://${head_node_ip}:${vllm_port_number}/v1"
echo "http://${head_node_ip}:${vllm_port_number}/v1" > ${VLLM_BASE_URL_FILENAME}

python3 -m vllm.entrypoints.openai.api_server \
--model ${VLLM_MODEL_WEIGHTS} \
--host "0.0.0.0" \
--port ${vllm_port_number} \
--tensor-parallel-size $((NUM_NODES*NUM_GPUS)) \
--dtype ${VLLM_DATA_TYPE} \
--load-format safetensors \
--trust-remote-code \
--max-model-len 22192 \
--max-logprobs ${VLLM_MAX_LOGPROBS} 
