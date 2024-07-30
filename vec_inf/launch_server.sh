#!/bin/bash
# ================================= Read Named Args ======================================
is_vlm=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-family) model_family="$2"; shift ;;
        --model-variant) model_variant="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --num-nodes) num_nodes="$2"; shift ;;
        --num-gpus) num_gpus="$2"; shift ;;
        --qos) qos="$2"; shift ;;
        --time) time="$2"; shift ;;
        --data-type) data_type="$2"; shift ;;
        --venv) virtual_env="$2"; shift ;;
        --is-vlm) is_vlm=true ;;
        --image-input-type) image_input_type="$2"; shift ;;
        --image-token-id) image_token_id="$2"; shift ;;
        --image-input-shape) image_input_shape="$2"; shift ;;
        --image-feature-size) image_feature_size="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$model_family" ]; then
    echo "Error: Missing required --model-family argument."
    exit 1
fi
# ================================= Set default environment variables ======================================
export MODEL_FAMILY=$model_family
export SRC_DIR="$(dirname "$0")"

# Load the configuration file for the specified model family
CONFIG_FILE="${SRC_DIR}/models/${MODEL_FAMILY}/config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi
source "$CONFIG_FILE"

# Model and entrypoint configuration. API Server URL (host, port) are set automatically based on the
# SLURM job and are written to the file specified at VLLM_BASE_URL_FILENAME
export MODEL_DIR="${SRC_DIR}/models/${MODEL_FAMILY}"
export VLLM_BASE_URL_FILENAME="${MODEL_DIR}/.${MODEL_NAME}-${MODEL_VARIANT}_url"
export VLLM_DATA_TYPE="auto"
 
# Variables specific to your working environment, below are examples for the Vector cluster
export VENV_BASE="singularity"
export VLLM_MODEL_WEIGHTS="/model-weights/${MODEL_NAME}-${MODEL_VARIANT}"
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"

# Slurm job configuration
export JOB_NAME="${MODEL_NAME}-${MODEL_VARIANT}"
export JOB_PARTITION="a40"
export QOS="m3"
export TIME="04:00:00"

# VLM configuration
if [ "$is_vlm" = true ]; then
    export CHAT_TEMPLATE="${SRC_DIR}/../models/${MODEL_FAMILY}/chat_template.jinja"
fi
# ======================================= Overwrite Env Vars ========================================

if [ -n "$partition" ]; then
    export JOB_PARTITION=$partition
fi

if [ -n "$num_nodes" ]; then
    export NUM_NODES=$num_nodes
fi

if [ -n "$num_gpus" ]; then
    export NUM_GPUS=$num_gpus
fi

if [ -n "$qos" ]; then
    export QOS=$qos
fi

if [ -n "$time" ]; then
    export TIME=$time
fi

if [ -n "$data_type" ]; then
    export VLLM_DATA_TYPE=$data_type
fi

if [ -n "$virtual_env" ]; then
    export VENV_BASE=$virtual_env
fi

if [ -n "$model_variant" ]; then
    export MODEL_VARIANT=$model_variant
    export VLLM_MODEL_WEIGHTS="/model-weights/${MODEL_NAME}-${MODEL_VARIANT}"
    export JOB_NAME="${MODEL_NAME}-${MODEL_VARIANT}"
    export VLLM_BASE_URL_FILENAME="${MODEL_DIR}/.${MODEL_NAME}-${MODEL_VARIANT}_url"
fi

if [ -n "$image_input_type" ]; then
    export IMAGE_INPUT_TYPE="$image_input_type"
fi

if [ -n "$image_token_id" ]; then
    export IMAGE_TOKEN_ID="$image_token_id"
fi

if [ -n "$image_input_shape" ]; then
    export IMAGE_INPUT_SHAPE="$image_input_shape"
fi

if [ -n "$image_feature_size" ]; then
    export IMAGE_FEATURE_SIZE="$image_feature_size"
fi

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ ${fp16_partitions} =~ ${JOB_PARTITION} ]]; then
    export VLLM_DATA_TYPE="float16"
    echo "Data type set to due to non-Ampere GPUs used: ${VLLM_DATA_TYPE}"
fi
# ========================================= Launch Server ==========================================

# Create a file to store the API server URL if it doesn't exist
if [ -f ${VLLM_BASE_URL_FILENAME} ]; then
    touch ${VLLM_BASE_URL_FILENAME}
fi

echo Job Name: ${JOB_NAME}
echo Partition: ${JOB_PARTITION}
echo Num Nodes: ${NUM_NODES}
echo GPUs per Node: ${NUM_GPUS}
echo QOS: ${QOS}
echo Walltime: ${TIME}
echo Data Type: ${VLLM_DATA_TYPE}

is_special=""
if [ "$NUM_NODES" -gt 1 ]; then
    is_special="multinode_"
fi

if [ "$is_vlm" = true ]; then
    is_special="vlm_"
fi

sbatch --job-name ${JOB_NAME} \
    --partition ${JOB_PARTITION} \
    --nodes ${NUM_NODES} \
    --gres gpu:${NUM_GPUS} \
    --qos ${QOS} \
    --time ${TIME} \
    --output ${MODEL_DIR}/${MODEL_NAME}-${MODEL_VARIANT}.%j.out \
    --error ${MODEL_DIR}/${MODEL_NAME}-${MODEL_VARIANT}.%j.err \
    ${SRC_DIR}/${is_special}vllm.slurm