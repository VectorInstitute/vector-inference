#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# NOTE: Instead of invoking this script directly via sbatch, use `bash openai_entrypoint.sh`
# to keep track of environment configurations.

# Load CUDA, change to the cuda version on your environment if different
module load cuda-12.3
nvidia-smi

# Activate vllm venv
source ${VENV_BASE}/bin/activate

source $(dirname ${MODEL_DIR})/find_port.sh

# Write server url to file
hostname=${SLURMD_NODENAME}
vllm_port_number=$(find_available_port $hostname 8080 65535)

echo "Server address: http://${hostname}:${vllm_port_number}/v1"
echo "http://${hostname}:${vllm_port_number}/v1" > ${VLLM_BASE_URL_FILENAME}

python3 -m vllm.entrypoints.openai.api_server \
--model ${VLLM_MODEL_WEIGHTS} \
--host "0.0.0.0" \
--port ${vllm_port_number} \
--tensor-parallel-size ${NUM_GPUS} \
--dtype ${VLLM_DATA_TYPE} \
--max-logprobs ${VLLM_MAX_LOGPROBS} 
