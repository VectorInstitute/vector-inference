#!/bin/bash
# ================================= Set config ======================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --num-nodes) num_nodes="$2"; shift ;;
        --num-gpus) num_gpus="$2"; shift ;;
        --qos) qos="$2"; shift ;;
        --time) time="$2"; shift ;;
        --data-type) data_type="$2"; shift ;;
        --venv) virtual_env="$2"; shift ;;
        --model-variant) model_variant="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$model" ]; then
    echo "Error: Missing required -m model argument."
    exit 1
fi

export MODEL=$model

CONFIG_FILE="$(dirname $(realpath "$0"))/models/${MODEL}/config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi
source "$CONFIG_FILE"
# ================================= Set environment variables ======================================

# Model and entrypoint configuration. API Server URL (host, port) are set automatically based on the
# SLURM job and are written to the file specified at VLLM_BASE_URL_FILENAME
export MODEL_NAME
export MODEL_VARIANT
export MODEL_DIR="$(dirname $(realpath "$0"))/models/${MODEL}"
export VLLM_BASE_URL_FILENAME="${MODEL_DIR}/.vLLM_${MODEL_NAME}-${MODEL_VARIANT}_url"
 
# Variables specific to your working environment, below are examples for the Vector cluster
export VENV_BASE=/projects/aieng/public/mixtral_vllm_env
export VLLM_MODEL_WEIGHTS="/model-weights/${MODEL_NAME}-${MODEL_VARIANT}"
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"

# Slurm job configuration
export JOB_NAME="vLLM/${MODEL_NAME}-${MODEL_VARIANT}"
export NUM_NODES
export NUM_GPUS
export JOB_PARTITION="a40"
export QOS="m3"
export TIME="04:00:00"

# Model configuration
export VLLM_MAX_LOGPROBS

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ ${fp16_partitions} =~ ${JOB_PARTITION} ]]; then
    export VLLM_DATA_TYPE="float16"
else
    export VLLM_DATA_TYPE="auto"
fi
# ======================================= Optional Settings ========================================

if [ -n "$partition" ]; then
    export JOB_PARTITION=$partition
    echo "Partition set to: ${JOB_PARTITION}"
fi

if [ -n "$num_nodes" ]; then
    export NUM_NODES=$num_nodes
    echo "Number of nodes set to: ${NUM_NODES}"
fi

if [ -n "$num_gpus" ]; then
    export NUM_GPUS=$num_gpus
    echo "Number of GPUs set to: ${NUM_GPUS}"
fi

if [ -n "$qos" ]; then
    export QOS=$qos
    echo "QOS set to: ${QOS}"
fi

if [ -n "$time" ]; then
    export TIME=$time
    echo "Walltime set to: ${TIME}"
fi

if [ -n "$data_type" ]; then
    export VLLM_DATA_TYPE=$data_type
    echo "Data type set to: ${VLLM_DATA_TYPE}"
fi

if [ -n "$virtual_env" ]; then
    export VENV_BASE=$virtual_env
    echo "Virtual environment set to: ${VENV_BASE}"
fi

if [ -n "$model_variant" ]; then
    export MODEL_VARIANT=$model_variant
    echo "Model variant set to: ${MODEL_VARIANT}"
    export VLLM_MODEL_WEIGHTS="/model-weights/${MODEL_NAME}-${MODEL_VARIANT}"
    export JOB_NAME="vLLM/${MODEL_NAME}-${MODEL_VARIANT}"
    export VLLM_BASE_URL_FILENAME="$(dirname $(realpath "$0"))/.vLLM_${MODEL_NAME}-${MODEL_VARIANT}_url"
fi

# ========================================= Launch Server ==========================================

# Create a file to store the API server URL if it doesn't exist
if [ -f ${VLLM_BASE_URL_FILENAME} ]; then
    touch ${VLLM_BASE_URL_FILENAME}
fi

echo Job Name: ${JOB_NAME}
echo Partition: ${JOB_PARTITION}
echo Generic Resource Scheduling: gpu:$((NUM_NODES*NUM_GPUS))
echo Data Type: ${VLLM_DATA_TYPE}

is_multi=""
if [ "$NUM_NODES" -gt 1 ]; then
    is_multi="multinode_"
fi

sbatch --job-name ${JOB_NAME} \
    --partition ${JOB_PARTITION} \
    --nodes ${NUM_NODES} \
    --gres gpu:${NUM_GPUS} \
    --qos ${QOS} \
    --time ${TIME} \
    --output ${MODEL_DIR}/vLLM-${MODEL_NAME}-${MODEL_VARIANT}.%j.out \
    --error ${MODEL_DIR}/vLLM-${MODEL_NAME}-${MODEL_VARIANT}.%j.err \
    $(dirname ${MODEL_DIR})/${is_multi}vllm.slurm