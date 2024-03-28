# Did you modify this line in openai_entrypoint.sh?
# If so, modify it here accordingly.
model_type="llama2"
top_directory=$(dirname $(dirname $(realpath "$0")))
VLLM_BASE_URL_FILENAME=${top_directory}/models/${model_type}/.vllm_api_base_url

API_BASE_URL=$(cat ${VLLM_BASE_URL_FILENAME})

curl ${API_BASE_URL}/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "/model-weights/Llama-2-7b-hf",
       "prompt": "What is the capital of Canada?",
       "max_tokens": 20
   }'