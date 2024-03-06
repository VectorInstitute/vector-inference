# Did you modify this line in openai_entrypoint.sh?
# If so, modify it here accordingly.
VLLM_BASE_URL_FILENAME=~/.vllm_api_base_url

API_BASE_URL=$(cat ${VLLM_BASE_URL_FILENAME})
echo "API_BASE_URL: ${API_BASE_URL}"
echo
curl ${API_BASE_URL}/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "google/gemma-2b-it",
  "messages": [
    {"role": "user", "content": "What is color of the sky?"}
  ]
}'