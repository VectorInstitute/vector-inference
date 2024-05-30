FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Non-interactive apt-get commands
ARG DEBIAN_FRONTEND=noninteractive
# No GPUs visible during build
ARG CUDA_VISIBLE_DEVICES=none
# Specify CUDA architectures -> 7.5: RTX 6000 & T4, 8.0: A100, 8.6: A40
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

# Install dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# # Set environment variables for Poetry installation
# ENV POETRY_VERSION=1.8
# ENV POETRY_HOME=/opt/poetry


# Create the scratch directory for Poetry cache
RUN mkdir -p /.cache/poetry_default
ENV POETRY_CACHE_DIR=/.cache/poetry_default

# Install Poetry
# RUN curl -sSL https://install.python-poetry.org | python3 - && \
#     ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry
RUN pip3 install poetry

# Clone the repository
RUN git clone https://github.com/VectorInstitute/vector-inference /vec-inf

# Copy the project files into the container
WORKDIR /vec-inf

# Set poetry to create a virtual env
RUN poetry config virtualenvs.create true

# Update poetry lock just in case
RUN poetry lock

# Install project dependencies via Poetry
RUN poetry install

# Install Flash Attention 2 backend
RUN poetry run pip install wheel
RUN poetry run pip install flash-attn --no-build-isolation
RUN poetry run pip install vllm-flash-attn

# Set the default command to start the virtual environment
CMD ["poetry", "shell"]