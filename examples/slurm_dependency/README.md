# SLURM Dependency Workflow Example

This example demonstrates how to launch a model server using `vec-inf`, and run a downstream SLURM job that waits for the server to become ready before querying it.

## Files

This directory contains the following:

1. [run_workflow.sh](run_workflow.sh)
   Launches the model server and submits the downstream job with a dependency, so it starts only after the server job begins running.

2. [downstream_job.sbatch](downstream_job.sbatch)
   A SLURM job script that runs the downstream logic (e.g., prompting the model).

3. [run_downstream.py](run_downstream.py)
   A Python script that waits until the inference server is ready, then sends a request using the OpenAI-compatible API.

## What to update

Before running this example, update the following in [downstream_job.sbatch](downstream_job.sbatch):

- `--job-name`, `--output`, and `--error` paths
- Virtual environment path in the `source` line
- SLURM resource configuration (e.g., partition, memory, GPU)

Also update the model name in [run_downstream.py](run_downstream.py) to match what you're launching.

## Running the example

First, activate a virtual environment where `vec-inf` is installed. Then, from this directory, run:

```bash
bash run_workflow.sh
