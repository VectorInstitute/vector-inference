# Did you modify this line in openai_entrypoint.sh?
# If so, modify it here accordingly.
parent_directory=$(dirname "$(pwd)")
model_type="mixtral"
VLLM_BASE_URL_FILENAME=${parent_directory}/models/${model_type}/.vllm_api_base_url

API_BASE_URL=$(cat ${VLLM_BASE_URL_FILENAME})
echo "API_BASE_URL: ${API_BASE_URL}"

curl ${API_BASE_URL}/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "/model-weights/Mixtral-8x7B-Instruct-v0.1",
       "prompt": "What is the capital of Canada?",
       "max_tokens": 20
   }'