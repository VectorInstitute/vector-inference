# The url is located in the .vLLM_model-variant_url file in the corresponding model directory.
export API_BASE_URL=http://gpuXXX:XXXX/v1

# Update the model path accordingly
curl ${API_BASE_URL}/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "/model-weights/Meta-Llama-3-8B",
       "prompt": "What is the capital of Canada?",
       "max_tokens": 20
   }'
