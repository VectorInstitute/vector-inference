"""Example script to query a launched model via the OpenAI-compatible API."""

import sys

from openai import OpenAI

from vec_inf.client import VecInfClient


if len(sys.argv) < 2:
    raise ValueError("Expected server job ID as the first argument.")
job_id = sys.argv[1]

vi_client = VecInfClient()
print(f"Waiting for SLURM job {job_id} to be ready...")
status = vi_client.wait_until_ready(slurm_job_id=job_id)
print(f"Server is ready at {status.base_url}")

api_client = OpenAI(base_url=status.base_url, api_key="EMPTY")
resp = api_client.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    prompt="Where is the capital of Canada?",
    max_tokens=20,
)

print(resp)
