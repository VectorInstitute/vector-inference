"""Example of using the OpenAI API to generate completions for vision tasks."""

from openai import OpenAI


# The url can be found with vec-inf status $JOB_ID
client = OpenAI(base_url="http://gpuXXX:XXXX/v1", api_key="EMPTY")

# Update the model path accordingly
completion = client.chat.completions.create(
    model="llava-1.5-13b-hf",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=50,
)

print(completion)
