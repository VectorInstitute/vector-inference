"""Example of how to get logits from the model."""

from openai import OpenAI


# The url is located in the .vLLM_model-variant_url file in the corresponding
# model directory.
client = OpenAI(base_url="http://gpuXXX:XXXXX/v1", api_key="EMPTY")

completion = client.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    prompt="Where is the capital of Canada?",
    max_tokens=1,
    logprobs=128256,  # Set to model vocab size to get logits
)

print(completion.choices[0].logprobs)
