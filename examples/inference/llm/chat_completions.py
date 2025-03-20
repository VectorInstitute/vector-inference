"""Example of how to use the OpenAI API to generate chat completions."""

from openai import OpenAI


# The url can be found with vec-inf status $JOB_ID
client = OpenAI(base_url="http://gpuXXX:XXXX/v1", api_key="EMPTY")

# Update the model path accordingly
completion = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ],
)

print(completion.model_dump_json())
