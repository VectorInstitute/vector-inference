#!bin/bash

# Load python module if you are on Vector cluster and install uv
module load python/3.10.13
module load rust
curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional: it's recommended to change the cache directory to somewhere in the scratch space to avoid
# running out of space in your home directory, below is an example for the Vector cluster
mkdir -p /scratch/$(whoami)/uv_cache
export UV_CACHE_DIR=/scratch/$(whoami)/uv_cache

# To see if the cache directory is set correctly, run the following command
# uv config get cache-dir
echo "Cache directory set to: $(uv config get cache-dir)"

# Install dependencies via uv
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Deactivate the virtual environment
# deactivate

# To check where your virtual environment is located, run the following command
# uv venv --show-path

# Alternatively, to activate your virtual environment without running uv shell, run the following command
# source $(uv venv --show-path)/bin/activate
