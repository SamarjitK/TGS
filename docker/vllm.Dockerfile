FROM tf_torch:fixed

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends patchelf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel 'numpy<2'; \
    # these are to avoid installing vllm with deps, which overwrites torch
    python3 -m pip install --no-cache-dir \
        transformers==4.39.1 \
        huggingface-hub \
        protobuf==3.19.6 \
        psutil \
        prometheus-client \
        aioprometheus \
        ray==2.9.3 \
        sentencepiece \
        py-cpuinfo \
        fastapi \
        "uvicorn[standard]" \
        "pydantic>=2.0" \
        pynvml==11.5.0 \
        outlines==0.0.34 \
        tiktoken==0.6.0; \
    python3 -m pip install --no-cache-dir --no-deps vllm==0.3.0; \
    mkdir -p /cache/huggingface /cache/torch; chown -R root:root /cache; chmod -R 777 /cache

ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/cache/huggingface/datasets
ENV TORCH_HOME=/cache/torch