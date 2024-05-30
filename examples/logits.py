from openai import OpenAI

# The url is located in the .vllm_model-variant_url file in the corresponding model directory.
client = OpenAI(base_url="http://gpuXXX:XXXXX/v1", api_key="EMPTY")

completion = client.completions.create(
    model="/model-weights/Llama-2-7b-hf",
    prompt="Where is the capital of Canada?",
    max_tokens=20,
    logprobs=32000 # Set to model vocab size to get logits
)

print(completion.choices[0].logprobs)
