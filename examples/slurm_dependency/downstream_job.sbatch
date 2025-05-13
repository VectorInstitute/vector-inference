#!/bin/bash
#SBATCH --job-name=Meta-Llama-3.1-8B-Instruct-downstream
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=$HOME/.vec-inf-logs/Meta-Llama-3.1-8B-Instruct-downstream.%j.out
#SBATCH --error=$HOME/.vec-inf-logs/Meta-Llama-3.1-8B-Instruct-downstream.%j.err

# Activate your environment
# TODO: update this path to match your venv location
source $HOME/vector-inference/.venv/bin/activate

# Wait for the server to be ready using the job ID passed as CLI arg
python run_downstream.py "$SERVER_JOB_ID"
