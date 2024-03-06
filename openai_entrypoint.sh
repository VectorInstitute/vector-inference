#!/bin/bash

source slurm.env

# Check if JOB_PARTITION is non-blank
if [ -n "$JOB_PARTITION" ]; then
    PARTITION_OPTION="--partition ${JOB_PARTITION}"
fi

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"
fp16_gres="turing volta rtx8000 v100 48gb 32gb"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ ${fp16_partitions} =~ ${JOB_PARTITION} ]]; then
    export VLLM_DATA_TYPE="float16"
else
    export VLLM_DATA_TYPE="auto"
fi

echo Job Name: ${JOB_NAME}
echo Partition: ${JOB_PARTITION}
echo JOB_GRES: ${JOB_GRES}
echo "Setting VLLM_DATA_TYPE to ${VLLM_DATA_TYPE}"

echo ""

sbatch \
--job-name ${JOB_NAME} \
${PARTITION_OPTION} \
-c 8 \
--mem ${JOB_CPU_MEMORY} \
--gres ${JOB_GRES} \
--qos m5 \
--time "1:00:00" \
--requeue \
openai_entrypoint.slurm.sh
