FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Non-interactive apt-get commands
ARG DEBIAN_FRONTEND=noninteractive

# No GPUs visible during build
ARG CUDA_VISIBLE_DEVICES=none

# Specify CUDA architectures -> 7.5: Quadro RTX 6000 & T4, 8.0: A100, 8.6: A40, 8.9: L40S, 9.0: H100
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

# Set the Python version
ARG PYTHON_VERSION=3.12.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev liblzma-dev libnuma1 \
    git vim ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-$PYTHON_VERSION.tgz Python-$PYTHON_VERSION

# Install pip and core Python tools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py && \
    python3.12 -m pip install --upgrade pip setuptools wheel uv

# Install RDMA support
RUN apt-get update && apt-get install -y \
    libibverbs1 libibverbs-dev ibverbs-utils \
    librdmacm1 librdmacm-dev rdmacm-utils \
    rdma-core ibverbs-providers infiniband-diags perftest \
    && rm -rf /var/lib/apt/lists/*

# Set up RDMA environment (these will persist in the final container)
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME="^lo,docker0"
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_TIMEOUT=22
ENV NCCL_IB_RETRY_CNT=7
ENV NCCL_DEBUG=INFO

# Set up project
WORKDIR /vec-inf
COPY . /vec-inf

# Install project dependencies pinned to uv.lock.
#
# `uv pip install` does NOT consult uv.lock -- only `uv sync` does, and
# `uv sync` requires a venv (incompatible with --system). Without this,
# every image build does fresh PyPI resolution and may pick a different
# transitive set than what the lockfile records (this is how :0.19.0
# shipped with the pyarrow/datasets ABI mismatch). Instead:
#   1. Export uv.lock to a fully-pinned requirements.txt (no resolver).
#   2. Install transitives with --no-deps so nothing is re-resolved.
#   3. Install the project itself editable, also --no-deps.
RUN uv export --frozen --no-emit-project --no-hashes \
        --extra vllm --group inference \
        -o /tmp/requirements.txt && \
    uv pip install --system --no-cache --no-deps --prerelease=allow \
        -r /tmp/requirements.txt && \
    uv pip install --system --no-cache --no-deps -e . && \
    rm -f /tmp/requirements.txt && \
    rm -rf /root/.cache/uv /tmp/*

# Build-time canary: fail the build if the locked deps cannot be imported
# together. This is the check that would have caught the pyarrow/datasets
# ABI mismatch in :0.19.0 at build time instead of at job start.
RUN python3.12 -c "import vllm, datasets, pyarrow, transformers, torch; \
    print('vllm', vllm.__version__, '/ datasets', datasets.__version__, \
          '/ pyarrow', pyarrow.__version__, '/ torch', torch.__version__)"

# Install a single, system NCCL (from NVIDIA CUDA repo in base image)
RUN apt-get update && apt-get install -y --allow-change-held-packages\
    libnccl2 libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the default command to start an interactive shell
CMD ["bash"]
