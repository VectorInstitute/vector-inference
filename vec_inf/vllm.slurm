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

# Activate vllm venv
if [ "$VENV_BASE" = "singularity" ]; then
    export SINGULARITY_IMAGE=/model-weights/vec-inf-shared/vector-inference_latest.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    module load singularity-ce/3.8.2
    singularity exec $SINGULARITY_IMAGE ray stop
    singularity exec --nv --bind ${VLLM_MODEL_WEIGHTS}:${VLLM_MODEL_WEIGHTS} $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${MODEL_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${SLURM_GPUS_PER_NODE} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
else
    source ${VENV_BASE}/bin/activate
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${MODEL_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${SLURM_GPUS_PER_NODE} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
fi
