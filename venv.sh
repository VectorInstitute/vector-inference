#!/bin/bash

# Load python module if you are on Vector cluster and install poetry
module load python/3.10.12
pip3 install poetry

# Optional: it's recommended to change the cache directory to somewhere in the scratch space to avoid
# running out of space in your home directory, below is an example for the Vector cluster
mkdir -p /scratch/ssd004/scratch/$(whoami)/poetry_cache
export POETRY_CACHE_DIR=/scratch/ssd004/scratch/$(whoami)/poetry_cache

# To see if the cache directory is set correctly, run the following command
# poetry config cache-dir
echo "Cache directory set to: $(poetry config cache-dir)"

echo "📜 Telling Poetry to use Python 3.10..."
poetry env use python3.10

# Install dependencies via poetry
poetry install

# Activate the virtual environment
# poetry shell

# Deactivate the virtual environment
# deactivate

# To check where your virtual environment is located, run the following command
# poetry env info --path

# Alternatively, to activate your virtual environment without running poetry shell, run the following command
# source $(poetry env info --path)/bin/activate
