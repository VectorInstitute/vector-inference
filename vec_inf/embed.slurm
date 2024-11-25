#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Load CUDA, change to the cuda version on your environment if different
source /opt/lmod/lmod/init/profile
module load cuda-12.3
nvidia-smi

source ${SRC_DIR}/find_port.sh

# Write server url to file
hostname=${SLURMD_NODENAME}
vllm_port_number=$(find_available_port $hostname 8080 65535)

echo "Server address: http://${hostname}:${vllm_port_number}/v1"
echo "http://${hostname}:${vllm_port_number}/v1" > ${VLLM_BASE_URL_FILENAME}

# Activate vllm venv
if [ "$VENV_BASE" = "singularity" ]; then
    export SINGULARITY_IMAGE=/projects/aieng/public/vector-inference_0.3.4.sif
    export VLLM_NCCL_SO_PATH=/vec-inf/nccl/libnccl.so.2.18.1
    module load singularity-ce/3.8.2
    singularity exec $SINGULARITY_IMAGE ray stop
    singularity exec --nv \
    --bind ${MODEL_WEIGHTS_PARENT_DIR}:${MODEL_WEIGHTS_PARENT_DIR} \
    --bind ${SRC_DIR}:${SRC_DIR} \
    $SINGULARITY_IMAGE \
    python3.10 ${SRC_DIR}/embeddings/openai_api_server.py \
    --model ${VLLM_MODEL_WEIGHTS} \
    --port ${vllm_port_number} \
    --trust-remote-code \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS}
else
    source ${VENV_BASE}/bin/activate
    python3 ${SRC_DIR}/embeddings/openai_api_server.py \
    --model ${VLLM_MODEL_WEIGHTS} \
    --port ${vllm_port_number} \
    --trust-remote-code \
    --max-num-seqs ${VLLM_MAX_NUM_SEQS}
fi