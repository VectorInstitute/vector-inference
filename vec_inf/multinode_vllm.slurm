#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --tasks-per-node=1

source ${SRC_DIR}/find_port.sh

if [ "$VENV_BASE" = "singularity" ]; then
    export SINGULARITY_IMAGE=/model-weights/vec-inf-shared/vector-inference_latest.sif
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
        singularity exec --nv --bind ${MODEL_WEIGHTS}:${MODEL_WEIGHTS} $SINGULARITY_IMAGE \
        ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
else
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
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
            singularity exec --nv --bind ${MODEL_WEIGHTS}:${MODEL_WEIGHTS} $SINGULARITY_IMAGE \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    else
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    fi

    sleep 5
done


vllm_port_number=$(find_available_port $head_node_ip 8080 65535)

SERVER_ADDR="http://${head_node_ip}:${vllm_port_number}/v1"
echo "Server address: $SERVER_ADDR"

jq --arg server_addr "$SERVER_ADDR" \
  '. + {"server_address": $server_addr}' \
  "$LOG_DIR/$MODEL_NAME.$SLURM_JOB_ID/$MODEL_NAME.$SLURM_JOB_ID.json" > temp.json \
  && mv temp.json "$LOG_DIR/$MODEL_NAME.$SLURM_JOB_ID/$MODEL_NAME.$SLURM_JOB_ID.json" \
  && rm temp.json

if [ "$PIPELINE_PARALLELISM" = "True" ]; then
    export PIPELINE_PARALLEL_SIZE=$SLURM_JOB_NUM_NODES
    export TENSOR_PARALLEL_SIZE=$SLURM_GPUS_PER_NODE
else
    export PIPELINE_PARALLEL_SIZE=1
    export TENSOR_PARALLEL_SIZE=$((SLURM_JOB_NUM_NODES*SLURM_GPUS_PER_NODE))
fi

if [ "$ENFORCE_EAGER" = "True" ]; then
    export ENFORCE_EAGER="--enforce-eager"
else
    export ENFORCE_EAGER=""
fi

if [ "$ENABLE_PREFIX_CACHING" = "True" ]; then
    export ENABLE_PREFIX_CACHING="--enable-prefix-caching"
else
    export ENABLE_PREFIX_CACHING=""
fi

if [ "$ENABLE_CHUNKED_PREFILL" = "True" ]; then
    export ENABLE_CHUNKED_PREFILL="--enable-chunked-prefill"
else
    export ENABLE_CHUNKED_PREFILL=""
fi

if [ -z "$MAX_NUM_BATCHED_TOKENS" ]; then
    export MAX_NUM_BATCHED_TOKENS=""
else
    export MAX_NUM_BATCHED_TOKENS="--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS"
fi

# Activate vllm venv
if [ "$VENV_BASE" = "singularity" ]; then
    singularity exec --nv --bind ${MODEL_WEIGHTS}:${MODEL_WEIGHTS} $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_WEIGHTS} \
    --served-model-name ${MODEL_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${DATA_TYPE} \
    --trust-remote-code \
    --max-logprobs ${MAX_LOGPROBS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --compilation-config ${COMPILATION_CONFIG} \
    --task ${TASK} \
    ${MAX_NUM_BATCHED_TOKENS} \
    ${ENABLE_PREFIX_CACHING} \
    ${ENABLE_CHUNKED_PREFILL} \
    ${ENFORCE_EAGER}
else
    source ${VENV_BASE}/bin/activate
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_WEIGHTS} \
    --served-model-name ${MODEL_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${DATA_TYPE} \
    --trust-remote-code \
    --max-logprobs ${MAX_LOGPROBS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --compilation-config ${COMPILATION_CONFIG} \
    --task ${TASK} \
    ${MAX_NUM_BATCHED_TOKENS} \
    ${ENABLE_PREFIX_CACHING} \
    ${ENABLE_CHUNKED_PREFILL} \
    ${ENFORCE_EAGER}
fi
