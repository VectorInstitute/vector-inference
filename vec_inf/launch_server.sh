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
        --max-num-seqs) max_num_seqs="$2"; shift ;;
        --vocab-size) vocab_size="$2"; shift ;;
        --data-type) data_type="$2"; shift ;;
        --venv) venv="$2"; shift ;;
        --log-dir) log_dir="$2"; shift ;;
        --model-weights-parent-dir) model_weights_parent_dir="$2"; shift ;;
        --pipeline-parallelism) pipeline_parallelism="$2"; shift ;;
        --enforce-eager) enforce_eager="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

required_vars=(model_family model_variant partition qos walltime num_nodes num_gpus max_model_len vocab_size data_type venv log_dir model_weights_parent_dir)

for var in "$required_vars[@]"; do
    if [ -z "$!var" ]; then
        echo "Error: Missing required --$var argument."
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
export VENV_BASE=$venv
export LOG_DIR=$log_dir
export MODEL_WEIGHTS_PARENT_DIR=$model_weights_parent_dir

if [ -n "$max_num_seqs" ]; then
    export VLLM_MAX_NUM_SEQS=$max_num_seqs
else 
    export VLLM_MAX_NUM_SEQS=256
fi

if [ -n "$pipeline_parallelism" ]; then
    export PIPELINE_PARALLELISM=$pipeline_parallelism
else
    export PIPELINE_PARALLELISM="False"
fi

if [ -n "$enforce_eager" ]; then
    export ENFORCE_EAGER=$enforce_eager
else
    export ENFORCE_EAGER="False"
fi

# ================================= Set default environment variables ======================================
# Slurm job configuration
export JOB_NAME="$MODEL_FAMILY-$MODEL_VARIANT"
if [ "$LOG_DIR" = "default" ]; then
    export LOG_DIR="$HOME/.vec-inf-logs/$MODEL_FAMILY"
fi
mkdir -p $LOG_DIR

# Model and entrypoint configuration. API Server URL (host, port) are set automatically based on the
# SLURM job 
export SRC_DIR="$(dirname "$0")"
export MODEL_DIR="${SRC_DIR}/models/${MODEL_FAMILY}"

# Variables specific to your working environment, below are examples for the Vector cluster
export VLLM_MODEL_WEIGHTS="${MODEL_WEIGHTS_PARENT_DIR}/${JOB_NAME}"
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"


# ================================ Validate Inputs & Launch Server =================================

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ $fp16_partitions =~ $JOB_PARTITION ]]; then
    export VLLM_DATA_TYPE="float16"
    echo "Data type set to due to non-Ampere GPUs used: $VLLM_DATA_TYPE"
fi

echo Job Name: $JOB_NAME
echo Partition: $JOB_PARTITION
echo Num Nodes: $NUM_NODES
echo GPUs per Node: $NUM_GPUS
echo QOS: $QOS
echo Walltime: $WALLTIME
echo Data Type: $VLLM_DATA_TYPE
echo Max Model Length: $VLLM_MAX_MODEL_LEN
echo Max Num Seqs: $VLLM_MAX_NUM_SEQS
echo Vocabulary Size: $VLLM_MAX_LOGPROBS
echo Pipeline Parallelism: $PIPELINE_PARALLELISM
echo Enforce Eager: $ENFORCE_EAGER
echo Log Directory: $LOG_DIR
echo Model Weights Parent Directory: $MODEL_WEIGHTS_PARENT_DIR

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
