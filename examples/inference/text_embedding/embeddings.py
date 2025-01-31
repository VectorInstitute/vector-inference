from openai import OpenAI

# The url is located in the .vLLM_model-variant_url file in the corresponding model directory.
client = OpenAI(base_url="http://gpu031:8081/v1", api_key="EMPTY")

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
