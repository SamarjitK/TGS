FROM bingyangwu2000/tf_torch

RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install -y patchelf && \
    apt-get clean

# Install pip if missing, then install HF packages only when not already available.
RUN set -eux; \
    # ensure pip is available
    if ! python3 -c "import pip" >/dev/null 2>&1; then \
        if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y python3-pip; fi; \
    fi; \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel || (curl -sS https://bootstrap.pypa.io/get-pip.py | python3); \
    # install transformers and related packages only if they are not present
    if ! python3 -c "import transformers" >/dev/null 2>&1; then \
        # Install only the core Python packages to avoid building Rust/C extensions
        python3 -m pip install --no-cache-dir transformers huggingface-hub || true; \
    else \
        echo "transformers already present in base image; skipping install"; \
    fi; \
    mkdir -p /cache/huggingface /cache/torch; chown -R root:root /cache; chmod -R 777 /cache

ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/cache/huggingface/datasets
ENV TORCH_HOME=/cache/torch