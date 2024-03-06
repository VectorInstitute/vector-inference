#!/bin/bash
#SBATCH --output logs/slurm.log
#SBATCH --error logs/slurm.log

# Note to users: 
# instead of invoking this script directly via 
# sbatch, use `bash openai_entrypoint.sh`
# to keep track of environment configurations.
source ${VENV_BASE}/bin/activate


# Write base url to file
job_id_last_3_digits=$(echo $SLURM_JOB_ID | grep -oE '[0-9]{3}$')
vllm_port_number="19${job_id_last_3_digits}"
hostname=${SLURMD_NODENAME}
vllm_base_url="http://${hostname}:${vllm_port_number}/v1"

nvidia-smi

echo "vllm_base_url: ${vllm_base_url}" 
echo "Writing vllm_base_url to ${VLLM_BASE_URL_FILENAME}"
echo

echo ${vllm_base_url} > ${VLLM_BASE_URL_FILENAME}

export 
cd ~
python3 -m vllm.entrypoints.openai.api_server \
--model ${VLLM_MODEL_NAME} \
--host "0.0.0.0" \
--port ${vllm_port_number} \
--tensor-parallel-size ${NUM_GPUS} \
--dtype ${VLLM_DATA_TYPE}