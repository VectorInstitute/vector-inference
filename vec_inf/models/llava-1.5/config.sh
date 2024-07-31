export MODEL_NAME="llava-1.5"
export MODEL_VARIANT="13b-hf"
export NUM_NODES=1
export NUM_GPUS=1
export VLLM_MAX_LOGPROBS=32000
export CHAT_TEMPLATE=${SRC_DIR}/models/llava-1.5/chat_template.jinja