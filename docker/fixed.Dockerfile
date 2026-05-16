# Dockerfile.fixed
# Based on the existing tf_torch base, upgraded to Python 3.10 (Ubuntu 20.04 CUDA base).
# This is a best-effort replica: some pinned packages (TensorFlow 1.x, very old wheels)
# may require additional manual adjustments. Use this as a starting point.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    unzip \
    openjdk-8-jdk-headless \
    openssh-client \
    openssh-server \
    libjpeg-dev \
    libpng-dev \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (available in Ubuntu 22.04 main repos)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.10 \
       python3.10-dev \
       python3.10-distutils \
       python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to python3.10 and install pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# Install common Python packages (relaxed versions for Python 3.10 compatibility)
RUN pip install --no-cache-dir \
    tqdm \
    scipy \
    datasets \
    transformers==4.10.3

# Proto/RPC stack (pinned to match requirements.txt for consistency)
RUN pip install --no-cache-dir \
    protobuf==3.20.3 \
    grpcio==1.48.2 \
    grpcio-tools==1.48.2 \
    nvidia-ml-py3 \
    setuptools==65.5.1 \
    wheel \
    matplotlib

# TensorFlow: choose a 3.10-compatible TF; TF 2.11+ supports Python 3.10.
# If your workloads require TF1.15, consider keeping a separate image.
# RUN pip install --no-cache-dir "numpy<2.0" tensorflow==2.11.0
# I dont think we need tflow?

# PyTorch: binary compatibility depends on Python & CUDA; you may need to
# adjust the wheel tags for your environment. This attempts to install
# CUDA 11.x compatible PyTorch wheels from the official index.
RUN echo "[step] Starting PyTorch base install (torch)" \
    && python --version \
    && pip --version \
    && pip install -v --no-cache-dir \
        torch==2.2.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && echo "[step] Finished torch install"

RUN echo "[step] Installing torchvision + torchaudio" \
    && pip install -v --no-cache-dir \
        torchvision==0.17.2+cu118 \
        torchaudio==2.2.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && echo "[step] Finished torchvision/torchaudio install"

RUN echo "[step] Verifying torch imports" \
    && python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__)"

RUN echo "[step] Final setup complete before LABEL/CMD"

# Horovod disabled for single-GPU setup.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     openmpi-bin \
#     libopenmpi-dev \
#     && rm -rf /var/lib/apt/lists/*

# ENV HOROVOD_GPU_OPERATIONS=NCCL \
#     HOROVOD_WITH_TENSORFLOW=1 \
#     HOROVOD_WITH_PYTORCH=1 \
#     HOROVOD_WITH_MXNET=1

# RUN pip install --no-cache-dir --upgrade cython \
#     && pip install --no-cache-dir --no-binary :all: horovod || true

# Minimal niceties and labels
LABEL maintainer="updated-base: py3.10" \
      description="Derived from bingyangwu/tf_torch with Python 3.10 (best-effort)."

CMD ["/bin/bash"]
