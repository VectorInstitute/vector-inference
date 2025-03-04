#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Load CUDA, change to the cuda version on your environment if different
# source /opt/lmod/lmod/init/profile
module purge
module load gcc/11.3.0
module load cuda/11.7.0
module load cudnn/8.4.0.27-11.6
#module load anaconda3_gpu/4.13.0
# module load cuda-12.2.1
nvidia-smi

source ${SRC_DIR}/find_port.sh

# Write server url to file
hostname=${SLURMD_NODENAME}
vllm_port_number=$(find_available_port $hostname 8080 65535)

echo "Server address: http://${hostname}:${vllm_port_number}/v1"

if [ "$ENFORCE_EAGER" = "True" ]; then
    export ENFORCE_EAGER="--enforce-eager"
else
    export ENFORCE_EAGER=""
fi

echo "VENV_BASE: $VENV_BASE"

# Activate vllm venv
if [ "$VENV_BASE" = "singularity" ]; then
    export SINGULARITY_IMAGE=/projects/bbwb/avd6/vector-inference/vector-inference-3.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    module load singularity-ce/3.8.2
    singularity exec $SINGULARITY_IMAGE ray stop
    singularity exec --nv --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
elif [ "$VENV_BASE" = "apptainer" ]; then
    export SINGULARITY_IMAGE=/projects/bbwb/avd6/vector-inference/vector-inference-3.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    #apptainer exec $SINGULARITY_IMAGE ray stop
    apptainer exec --nv --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} $SINGULARITY_IMAGE \
    python3.10 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
else
    source ${VENV_BASE}/bin/activate
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
    --task ${VLLM_TASK} \
    ${ENFORCE_EAGER}
fi
