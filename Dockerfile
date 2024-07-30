FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Non-interactive apt-get commands
ARG DEBIAN_FRONTEND=noninteractive

# No GPUs visible during build
ARG CUDA_VISIBLE_DEVICES=none

# Specify CUDA architectures -> 7.5: RTX 6000 & T4, 8.0: A100, 8.6+PTX
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

# Set the Python version
ARG PYTHON_VERSION=3.10.12

# Install dependencies for building Python
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python from precompiled binaries
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-$PYTHON_VERSION.tgz Python-$PYTHON_VERSION

# Download and install pip using get-pip.py
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Ensure pip for Python 3.10 is used
RUN python3.10 -m pip install --upgrade pip

# Install Poetry using Python 3.10
RUN python3.10 -m pip install poetry

# Clone the repository
RUN git clone -b develop https://github.com/VectorInstitute/vector-inference /vec-inf

# Set the working directory
WORKDIR /vec-inf

# Don't create venv
RUN poetry config virtualenvs.create false

# Update Poetry lock file if necessary
RUN poetry lock

# Install vec-inf
RUN python3.10 -m pip install .[dev]

# Install Flash Attention 2 backend
RUN python3.10 -m pip install flash-attn --no-build-isolation

# Move nccl to accessible location
RUN mkdir -p /vec-inf/nccl
RUN mv /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1 /vec-inf/nccl/libnccl.so.2.18.1; 

# Set the default command to start an interactive shell
CMD ["bash"]
