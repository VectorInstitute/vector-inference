import os
from openai import OpenAI

model_type = "llama2"
vec_inf_dir = os.path.dirname(os.getcwd())
with open(f"{vec_inf_dir}/models/{model_type}/.vllm_api_base_url", "r") as f:
    base_url = f.read()

client = OpenAI(base_url=base_url, api_key="EMPTY")

completion = client.completions.create(
    model="/model-weights/Llama-2-7b-hf",
    prompt="Where is the capital of Canada?",
    temperature=0.01,
    max_tokens=20,
)

print(completion)