"""Example of how to use the OpenAI API to generate embeddings."""

from openai import OpenAI


# The url can be found with vec-inf status $JOB_ID
client = OpenAI(base_url="http://gpuXXX:XXXX/v1", api_key="EMPTY")

model_name = "bge-base-en-v1.5"

input_texts = [
    "The chef prepared a delicious meal.",
]

# test single embedding
embedding_response = client.embeddings.create(
    model=model_name,
    input=input_texts,
    encoding_format="float",
)

print(embedding_response)
