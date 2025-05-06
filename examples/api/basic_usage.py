#!/usr/bin/env python
"""Basic example of Vector Inference API usage.

This script demonstrates the core features of the Vector Inference API
for launching and interacting with models.
"""

from vec_inf.client import VecInfClient


# Create the API client
client = VecInfClient()

# List available models
print("Listing available models...")
models = client.list_models()
print(f"Found {len(models)} models")
for model in models[:3]:  # Show just the first few
    print(f"- {model.name} ({model.model_type})")

# Launch a model (replace with an actual model name from your environment)
model_name = "Meta-Llama-3.1-8B-Instruct"  # Use an available model from your list
print(f"\nLaunching {model_name}...")
response = client.launch_model(model_name)
job_id = response.slurm_job_id
print(f"Launched with job ID: {job_id}")

# Wait for the model to be ready
print("Waiting for model to be ready...")
status = client.wait_until_ready(job_id)
print(f"Model is ready at: {status.base_url}")

# Get metrics
print("\nRetrieving metrics...")
metrics = client.get_metrics(job_id)
if isinstance(metrics.metrics, dict):
    for key, value in metrics.metrics.items():
        print(f"- {key}: {value}")

# Shutdown when done
print("\nShutting down model...")
client.shutdown_model(job_id)
print("Model shutdown complete")
