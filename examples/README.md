# Examples
- [`inference`](inference): Examples for sending inference requests
  - [`llm/chat_completions.py`](inference/llm/chat_completions.py): Python example of sending chat completion requests to OpenAI compatible server
  - [`llm/completions.py`](inference/llm/completions.py): Python example of sending completion requests to OpenAI compatible server
  - [`llm/completions.sh`](inference/llm/completions.sh): Bash example of sending completion requests to OpenAI compatible server, supports JSON mode
  - [`text_embedding/embeddings.py`](inference/text_embedding/embeddings.py): Python example of sending text embedding requests to OpenAI compatible server
  - [`vlm/vision_completions.py`](inference/vlm/vision_completions.py): Python example of sending chat completion requests with image attached to prompt to OpenAI compatible server for vision language models
- [`logits`](logits): Example for logits generation
  - [`logits.py`](logits/logits.py): Python example of getting logits from hosted model.
- [`api`](api): Examples for using the Python API
  - [`basic_usage.py`](api/basic_usage.py): Basic Python example demonstrating the Vector Inference API
  - [`advanced_usage.py`](api/advanced_usage.py): Advanced Python example with rich UI for the Vector Inference API
