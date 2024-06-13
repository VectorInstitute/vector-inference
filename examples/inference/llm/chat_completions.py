from openai import OpenAI

# The url is located in the .vLLM_model-variant_url file in the corresponding model directory.
client = OpenAI(base_url="http://gpuXXX:XXXX/v1", api_key="EMPTY")

# Update the model path accordingly
completion = client.chat.completions.create(
  model="/model-weights/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
  ]
)

print(completion)