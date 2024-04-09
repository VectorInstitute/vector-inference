# The url is located in the .vllm_model-variant_url file in the corresponding model directory.
export API_BASE_URL=http://gpuXXX:XXXXX/v1

curl ${API_BASE_URL}/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "/model-weights/Llama-2-13b-hf",
       "prompt": "What is the capital of Canada?",
       "max_tokens": 20
   }'