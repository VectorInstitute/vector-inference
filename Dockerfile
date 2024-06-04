FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Non-interactive apt-get commands
ARG DEBIAN_FRONTEND=noninteractive

# No GPUs visible during build
ARG CUDA_VISIBLE_DEVICES=none

# Specify CUDA architectures -> 7.5: RTX 6000 & T4, 8.0: A100, 8.6+PTX
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

# Install dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv python3.10-dev git bash && \
    rm -rf /var/lib/apt/lists/*

# Download and install pip using get-pip.py
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Create the scratch directory for Poetry cache
RUN mkdir -p /.cache/poetry_default
ENV POETRY_CACHE_DIR=/.cache/poetry_default

# Ensure pip for Python 3.10 is used
RUN python3.10 -m pip install --upgrade pip

# Install Poetry using Python 3.10
RUN python3.10 -m pip install poetry

# Clone the repository
RUN git clone https://github.com/VectorInstitute/vector-inference /vec-inf

# Set the working directory
WORKDIR /vec-inf

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Update Poetry lock file if necessary
RUN poetry lock

# Install project dependencies via Poetry
RUN poetry install

# Install Flash Attention 2 backend
RUN python3.10 -m pip install flash-attn --no-build-isolation

# Set the default command to start an interactive shell
CMD ["bash"]
