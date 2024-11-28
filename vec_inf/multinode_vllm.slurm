#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --tasks-per-node=1

# Load CUDA, change to the cuda version on your environment if different
source /opt/lmod/lmod/init/profile
module load cuda-12.3
nvidia-smi

source ${SRC_DIR}/find_port.sh

if [ "$VENV_BASE" = "singularity" ]; then
    export SINGULARITY_IMAGE=/projects/aieng/public/vector-inference_0.6.4.post1.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    module load singularity-ce/3.8.2
    singularity exec $SINGULARITY_IMAGE ray stop
fi

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
if [ "$VENV_BASE" = "singularity" ]; then
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        singularity exec --nv --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} $SINGULARITY_IMAGE \
        ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
else
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
fi

# Starting the Ray worker nodes
# Optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    if [ "$VENV_BASE" = "singularity" ]; then
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            singularity exec --nv --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} $SINGULARITY_IMAGE \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
    else
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
    fi

    sleep 5
done


vllm_port_number=$(find_available_port $head_node_ip 8080 65535)

echo "Server address: http://${head_node_ip}:${vllm_port_number}/v1"

if [ "$PIPELINE_PARALLELISM" = "True" ]; then
    export PIPELINE_PARALLEL_SIZE=$NUM_NODES
    export TENSOR_PARALLEL_SIZE=$NUM_GPUS
else
    export PIPELINE_PARALLEL_SIZE=1
    export TENSOR_PARALLEL_SIZE=$((NUM_NODES*NUM_GPUS))
fi

if [ "$ENFORCE_EAGER" = "True" ]; then
    export ENFORCE_EAGER="--enforce-eager"
else
    export ENFORCE_EAGER=""
fi

# Activate vllm venv
if [ "$VENV_BASE" = "singularity" ]; then
    singularity exec --nv --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${VLLM_DATA_TYPE} \
    --trust-remote-code \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    ${ENFORCE_EAGER}
else
    source ${VENV_BASE}/bin/activate
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${VLLM_DATA_TYPE} \
    --trust-remote-code \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    ${ENFORCE_EAGER}
fi
