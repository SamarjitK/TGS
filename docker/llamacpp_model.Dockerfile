FROM tf_torch:llamacpp

# You can use the associated bash script instead to run this automatically with your HF_TOKEN!

# Build args for model baking (defaults target Llama-3.2-3B-Instruct GGUF).
ARG MODEL_REPO=bartowski/Llama-3.2-3B-Instruct-GGUF
ARG MODEL_INCLUDE=Llama-3.2-3B-Instruct-f16.gguf
ARG MODEL_FILENAME=Llama-3.2-3B-Instruct-f16.gguf
ARG MODEL_DIR=/opt/models
ARG MODEL_SHA256=
ARG HF_TOKEN=

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir huggingface_hub

# download took around 2 minutes for me, but saving the image takes some time as well.
RUN mkdir -p "${MODEL_DIR}" && \
    if [ -n "${HF_TOKEN}" ]; then \
                HF_TOKEN="${HF_TOKEN}" hf download "${MODEL_REPO}" \
          --include "${MODEL_INCLUDE}" \
          --local-dir "${MODEL_DIR}"; \
    else \
                hf download "${MODEL_REPO}" \
          --include "${MODEL_INCLUDE}" \
          --local-dir "${MODEL_DIR}"; \
    fi && \
    test -s "${MODEL_DIR}/${MODEL_FILENAME}"

RUN if [ -n "${MODEL_SHA256}" ]; then \
        echo "${MODEL_SHA256}  ${MODEL_DIR}/${MODEL_FILENAME}" | sha256sum -c -; \
    fi

ENV LLAMACPP_MODEL=${MODEL_DIR}/${MODEL_FILENAME}

WORKDIR /workspace

CMD ["/bin/bash"]
