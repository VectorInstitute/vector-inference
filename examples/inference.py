from openai import OpenAI

# The url is located in the .vllm_model-variant_url file in the corresponding model directory.
client = OpenAI(base_url="http://gpuXXX:XXXXX/v1", api_key="EMPTY")

completion = client.completions.create(
    model="/model-weights/Llama-2-70b-hf",
    prompt="Where is the capital of Canada?",
    max_tokens=20,
)

print(completion)