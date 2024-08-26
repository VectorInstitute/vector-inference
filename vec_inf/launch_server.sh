#!/bin/bash

# ================================= Read Named Args ======================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-family) model_family="$2"; shift ;;
        --model-variant) model_variant="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --qos) qos="$2"; shift ;;
        --time) walltime="$2"; shift ;;
        --num-nodes) num_nodes="$2"; shift ;;
        --num-gpus) num_gpus="$2"; shift ;;
        --max-model-len) max_model_len="$2"; shift ;;
        --vocab-size) vocab_size="$2"; shift ;;
        --data-type) data_type="$2"; shift ;;
        --venv) virtual_env="$2"; shift ;;
        --log-dir) log_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

required_vars=(model_family model_variant partition qos walltime num_nodes num_gpus max_model_len vocab_size data_type virtual_env log_dir)

for var in "$required_vars[@]"; do
    if [ -z "$!var" ]; then
        echo "Error: Missing required --$var//_/- argument."
        exit 1
    fi
done

export MODEL_FAMILY=$model_family
export MODEL_VARIANT=$model_variant
export JOB_PARTITION=$partition
export QOS=$qos
export WALLTIME=$walltime
export NUM_NODES=$num_nodes
export NUM_GPUS=$num_gpus
export VLLM_MAX_MODEL_LEN=$max_model_len
export VLLM_MAX_LOGPROBS=$vocab_size
export VLLM_DATA_TYPE=$data_type
export VENV_BASE=$virtual_env
export LOG_DIR=$log_dir

# ================================= Set default environment variables ======================================
# Slurm job configuration
export JOB_NAME="$MODEL_FAMILY-$MODEL_VARIANT"
if [ "$LOG_DIR" = "default" ]; then
    export LOG_DIR="$HOME/.vec-inf-logs/$MODEL_FAMILY"
fi
mkdir -p $LOG_DIR

# Model and entrypoint configuration. API Server URL (host, port) are set automatically based on the
# SLURM job and are written to the file specified at VLLM_BASE_URL_FILENAME
export SRC_DIR="$(dirname "$0")"
export MODEL_DIR="${SRC_DIR}/models/${MODEL_FAMILY}"
export VLLM_BASE_URL_FILENAME="${MODEL_DIR}/.${JOB_NAME}_url"
 
# Variables specific to your working environment, below are examples for the Vector cluster
export VLLM_MODEL_WEIGHTS="/model-weights/$JOB_NAME"
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"


# ================================ Validate Inputs & Launch Server =================================

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ $fp16_partitions =~ $JOB_PARTITION ]]; then
    export VLLM_DATA_TYPE="float16"
    echo "Data type set to due to non-Ampere GPUs used: $VLLM_DATA_TYPE"
fi

# Create a file to store the API server URL if it doesn't exist
if [ -f $VLLM_BASE_URL_FILENAME ]; then
    touch $VLLM_BASE_URL_FILENAME
fi

echo Job Name: $JOB_NAME
echo Partition: $JOB_PARTITION
echo Num Nodes: $NUM_NODES
echo GPUs per Node: $NUM_GPUS
echo QOS: $QOS
echo Walltime: $WALLTIME
echo Data Type: $VLLM_DATA_TYPE

is_special=""
if [ "$NUM_NODES" -gt 1 ]; then
    is_special="multinode_"
fi

sbatch --job-name $JOB_NAME \
    --partition $JOB_PARTITION \
    --nodes $NUM_NODES \
    --gres gpu:$NUM_GPUS \
    --qos $QOS \
    --time $WALLTIME \
    --output $LOG_DIR/$JOB_NAME.%j.out \
    --error $LOG_DIR/$JOB_NAME.%j.err \
    $SRC_DIR/${is_special}vllm.slurm