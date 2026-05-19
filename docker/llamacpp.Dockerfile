FROM tf_torch:fixed

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir requests

ARG LLAMACPP_REPO=https://github.com/ggerganov/llama.cpp.git
ARG LLAMACPP_REF=master

# Quadro RTX 6000 = compute capability 7.5
# Change this to match your GPU's compute capability.
ARG CUDA_ARCH=75

RUN git clone --depth 1 --branch "${LLAMACPP_REF}" "${LLAMACPP_REPO}" /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build /opt/llama.cpp/build -j"$(nproc)" 2>&1 | grep -i "llama-server\|error:" || true && \
    server_bin="$(find /opt/llama.cpp/build -type f -name 'llama-server' -executable 2>/dev/null | head -n 1)" && \
    if [ -n "${server_bin}" ]; then \
        echo "Found server binary at: ${server_bin}"; \
        ln -sf "${server_bin}" /usr/local/bin/llama-server; \
    else \
        echo "Error: llama-server executable not found in build"; \
        find /opt/llama.cpp/build -type f -executable -not -name "*.so*" -not -name "*CMake*" -not -name "*.o" 2>/dev/null; \
        false; \
    fi

ENV LLAMACPP_PATH=/opt/llama.cpp
ENV LLAMACPP_SERVER_BIN=/usr/local/bin/llama-server

WORKDIR /workspace

CMD ["/bin/bash"]