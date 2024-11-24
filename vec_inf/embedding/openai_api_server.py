import argparse
import asyncio
import base64
from asyncio import Queue
from typing import List, Optional, Union
import sys

import torch
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


# Define request and response models
class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = "float"  # Default to 'float'
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str
    embedding: Union[List[float], str]  # Can be a list of floats or a base64 string
    index: int


class EmbeddingsResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: dict


parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--port", type=int)
parser.add_argument("--max-num-seqs", type=int)
parser.add_argument("--trust-remote-code", type=bool, action="store_true")
args = parser.parse_args()


# Initialize the FastAPI app
app = FastAPI()

# Load the tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

# Initialize the request queue and batch processing parameters
request_queue = Queue()
BATCH_TIMEOUT = 0.01  # in seconds


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """
    Handle incoming embedding requests by adding them to the processing queue.
    """
    # Create a Future to hold the result
    future = asyncio.get_event_loop().create_future()
    # Put the request into the queue
    await request_queue.put((request, future))
    # Wait for the result
    result = await future
    return result


@app.get("/health")
def status_check():
    """
    Returns 200.
    """
    return Response(status_code=200)


async def process_queue():
    """
    Continuously process requests from the queue in batches.
    """
    while True:
        requests_futures = []
        try:
            # Wait for at least one item
            request_future = await request_queue.get()
            requests_futures.append(request_future)
            # Now, try to get more items with a timeout
            try:
                while len(requests_futures) < args.max_num_seqs:
                    request_future = await asyncio.wait_for(
                        request_queue.get(), timeout=BATCH_TIMEOUT
                    )
                    requests_futures.append(request_future)
            except asyncio.TimeoutError:
                pass
        except Exception:
            continue
        # Process the batch
        requests = [rf[0] for rf in requests_futures]
        futures = [rf[1] for rf in requests_futures]
        # Collect input texts and track counts
        batched_input_texts = []
        input_counts = []
        encoding_formats = []
        for request in requests:
            input_text = request.input
            if isinstance(input_text, str):
                input_text = [input_text]
            input_counts.append(len(input_text))
            batched_input_texts.extend(input_text)
            encoding_formats.append(request.encoding_format)
        # Tokenize and compute embeddings
        inputs = tokenizer(
            batched_input_texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        # Split embeddings back to individual requests
        idx = 0
        for request, future, count, encoding_format in zip(
            requests, futures, input_counts, encoding_formats
        ):
            request_embeddings = embeddings[idx : idx + count]
            idx += count
            # Prepare response
            data = []
            for i, embedding in enumerate(request_embeddings):
                if encoding_format == "base64":
                    # Convert list of floats to bytes
                    embedding_bytes = (
                        torch.tensor(embedding, dtype=torch.float32).numpy().tobytes()
                    )
                    # Encode bytes to base64 string
                    embedding_base64 = base64.b64encode(embedding_bytes).decode("utf-8")
                    data.append(
                        EmbeddingData(
                            object="embedding", embedding=embedding_base64, index=i
                        )
                    )
                else:
                    data.append(
                        EmbeddingData(object="embedding", embedding=embedding, index=i)
                    )
            response = EmbeddingsResponse(
                object="list",
                data=data,
                model=request.model,
                usage={
                    "prompt_tokens": len(inputs["input_ids"]),  # type: ignore
                    "total_tokens": len(inputs["input_ids"]),  # type: ignore
                },
            )
            # Set the result
            future.set_result(response)


@app.on_event("startup")
async def startup_event():
    """
    Start the background task to process the request queue.
    """
    asyncio.create_task(process_queue())


if __name__ == "__main__":
    print("INFO:     Application startup complete.", file=sys.stderr)
    uvicorn.run("embedding_server:app", host="0.0.0.0", port=args.port)
