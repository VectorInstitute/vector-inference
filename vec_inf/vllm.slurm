#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ${SRC_DIR}/find_port.sh

# Write server url to file
hostname=${SLURMD_NODENAME}
vllm_port_number=$(find_available_port $hostname 8080 65535)

SERVER_ADDR="http://${hostname}:${vllm_port_number}/v1"
echo "Server address: $SERVER_ADDR"

jq --arg server_addr "$SERVER_ADDR" \
  '. + {"server_address": $server_addr}' \
  "$LOG_DIR/$MODEL_NAME.$SLURM_JOB_ID/$MODEL_NAME.$SLURM_JOB_ID.json" > temp.json \
  && mv temp.json "$LOG_DIR/$MODEL_NAME.$SLURM_JOB_ID/$MODEL_NAME.$SLURM_JOB_ID.json" \
  && rm temp.json

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
    export SINGULARITY_IMAGE=/model-weights/vec-inf-shared/vector-inference_latest.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    module load singularity-ce/3.8.2
    singularity exec $SINGULARITY_IMAGE ray stop
    singularity exec --nv --bind ${MODEL_WEIGHTS}:${MODEL_WEIGHTS} $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_WEIGHTS} \
    --served-model-name ${MODEL_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${SLURM_GPUS_PER_NODE} \
    --dtype ${DATA_TYPE} \
    --max-logprobs ${MAX_LOGPROBS} \
    --trust-remote-code \
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
    --tensor-parallel-size ${SLURM_GPUS_PER_NODE} \
    --dtype ${DATA_TYPE} \
    --max-logprobs ${MAX_LOGPROBS} \
    --trust-remote-code \
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
