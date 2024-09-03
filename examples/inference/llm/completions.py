from openai import OpenAI

# The url is located in the .vLLM_model-variant_url file in the corresponding model directory.
client = OpenAI(base_url="http://gpuXXX:XXXX/v1", api_key="EMPTY")

# Update the model path accordingly
completion = client.completions.create(
    model="/model-weights/Meta-Llama-3-8B",
    prompt="Where is the capital of Canada?",
    max_tokens=20,
)

print(completion)
