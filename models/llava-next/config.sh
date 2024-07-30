export MODEL_NAME="llava-v1.6"
export MODEL_VARIANT="mistral-7b-hf"
export NUM_NODES=1
export NUM_GPUS=1
export VLLM_MAX_LOGPROBS=32064

export IMAGE_INPUT_TYPE="pixel_values"
export IMAGE_TOKEN_ID=32000
export IMAGE_INPUT_SHAPE="1,3,560,560"
export IMAGE_FEATURE_SIZE=2928