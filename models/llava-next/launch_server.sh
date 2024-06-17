#!/bin/bash

# ================================= Set environment variables ======================================

# Model and entrypoint configuration. API Server URL (host, port) are set automatically based on the
# SLURM job and are written to the file specified at VLLM_BASE_URL_FILENAME
export MODEL_NAME="llava-v1.6"
export MODEL_VARIANT="mistral-7b-hf"
export MODEL_DIR="$(dirname $(realpath "$0"))"
export VLLM_BASE_URL_FILENAME="${MODEL_DIR}/.vLLM_${MODEL_NAME}-${MODEL_VARIANT}_url"
 
# Variables specific to your working environment, below are examples for the Vector cluster
export VENV_BASE="singularity"
export VLLM_MODEL_WEIGHTS=/model-weights/${MODEL_NAME}-${MODEL_VARIANT}
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"

# Slurm job configuration
export JOB_NAME="vLLM/${MODEL_NAME}-${MODEL_VARIANT}"
export NUM_NODES=1
export NUM_GPUS=1
export JOB_PARTITION="a40"
export QOS="m3"
export TIME="04:00:00"

# Model configuration
relative_path=$(realpath --relative-to="$(pwd)" "$MODEL_DIR")
export CHAT_TEMPLATE="$(pwd)/$relative_path/chat_template.jinja"

export VLLM_MAX_LOGPROBS=32064
export IMAGE_INPUT_TYPE="pixel_values"
export IMAGE_TOKEN_ID=32000
export IMAGE_INPUT_SHAPE="1,3,560,560"
export IMAGE_FEATURE_SIZE=2928

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ ${fp16_partitions} =~ ${JOB_PARTITION} ]]; then
    export VLLM_DATA_TYPE="float16"
else
    export VLLM_DATA_TYPE="auto"
fi

# ======================================= Optional Settings ========================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition) partition="$2"; shift ;;
        --num-gpus) num_gpus="$2"; shift ;;
        --qos) qos="$2"; shift ;;
        --time) time="$2"; shift ;;
        --data-type) data_type="$2"; shift ;;
        --venv) virtual_env="$2"; shift ;;
        --image-input-type) image_input_type="$2"; shift ;;
        --image-token-id) image_token_id="$2"; shift ;;
        --image-input-shape) image_input_shape="$2"; shift ;;
        --image-feature-size) image_feature_size="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$partition" ]; then
    export JOB_PARTITION="$partition"
    echo "Partition set to: ${JOB_PARTITION}"
fi

if [ -n "$num_gpus" ]; then
    export NUM_GPUS="$num_gpus"
    echo "Number of GPUs set to: ${NUM_GPUS}"
fi

if [ -n "$qos" ]; then
    export QOS="$qos"
    echo "QOS set to: ${QOS}"
fi

if [ -n "$time" ]; then
    export TIME="$time"
    echo "Walltime set to: ${TIME}"
fi

if [ -n "$data_type" ]; then
    export VLLM_DATA_TYPE="$data_type"
    echo "Data type set to: ${VLLM_DATA_TYPE}"
fi

if [ -n "$virtual_env" ]; then
    export VENV_BASE="$virtual_env"
    echo "Virtual environment set to: ${VENV_BASE}"
fi

if [ -n "$image_input_type" ]; then
    export IMAGE_INPUT_TYPE="$image_input_type"
    echo "Image input type set to: ${IMAGE_INPUT_TYPE}"
fi

if [ -n "$image_token_id" ]; then
    export IMAGE_TOKEN_ID="$image_token_id"
    echo "Image token ID set to: ${IMAGE_TOKEN_ID}"
fi

if [ -n "$image_input_shape" ]; then
    export IMAGE_INPUT_SHAPE="$image_input_shape"
    echo "Image input shape set to: ${IMAGE_INPUT_SHAPE}"
fi

if [ -n "$image_feature_size" ]; then
    export IMAGE_FEATURE_SIZE="$image_feature_size"
    echo "Image feature size set to: ${IMAGE_FEATURE_SIZE}"
fi

# ========================================= Launch Server ==========================================

# Create a file to store the API server URL if it doesn't exist
if [ -f ${VLLM_BASE_URL_FILENAME} ]; then
    touch ${VLLM_BASE_URL_FILENAME}
fi

echo Job Name: ${JOB_NAME}
echo Partition: ${JOB_PARTITION}
echo Generic Resource Scheduling: gpu:${NUM_GPUS}
echo Data Type: ${VLLM_DATA_TYPE}

sbatch --job-name ${JOB_NAME} \
    --partition ${JOB_PARTITION} \
    --nodes ${NUM_NODES} \
    --gres gpu:${NUM_GPUS} \
    --qos ${QOS} \
    --time ${TIME} \
    --output ${MODEL_DIR}/vllm-${MODEL_NAME}-${MODEL_VARIANT}.%j.out \
    --error ${MODEL_DIR}/vllm-${MODEL_NAME}-${MODEL_VARIANT}.%j.err \
    $(dirname ${MODEL_DIR})/vlm_vllm.slurm
