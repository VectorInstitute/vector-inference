#!bin/bash

# The url can be found with vec-inf status $JOB_ID
export API_BASE_URL=http://gpuXXX:XXXX/v1

# Update the model path accordingly
curl ${API_BASE_URL}/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "Meta-Llama-3.1-8B-Instruct",
       "prompt": "What is the capital of Canada?",
       "max_tokens": 20
   }'
