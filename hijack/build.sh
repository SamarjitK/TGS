#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function build() {
    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
    rm -rf ${ROOT}/build
    mkdir ${ROOT}/build
    cd ${ROOT}/build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

    rm -rf ${ROOT}/high-priority-lib 
    mkdir ${ROOT}/high-priority-lib
    cd ${ROOT}/high-priority-lib
    cp ${ROOT}/build/libcuda-control-high-priority.so ./libcuda-control.so

    touch ./ld.so.preload
    echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
    cp libcuda-control.so ./libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
    cp libcuda-control.so ./libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so

    rm -rf ${ROOT}/low-priority-lib 
    mkdir ${ROOT}/low-priority-lib
    cd ${ROOT}/low-priority-lib
    cp ${ROOT}/build/libcuda-control-low-priority.so ./libcuda-control.so

    touch ./ld.so.preload
    echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
    cp libcuda-control.so ./libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
    cp libcuda-control.so ./libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so

    cd ..
    # BASE_IMAGE=${BASE_IMAGE:-"tf_torch"}
    # docker build -t ${BASE_IMAGE} --network=host -f ./Dockerfile .
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${ROOT}/.." && pwd -P)
BUILDER_IMAGE="${TGS_BUILDER_IMAGE:-bingyangwu2000/tf_torch}"

if [[ "${INSIDE_TGS_BUILD:-0}" == "1" ]]; then
    build
    exit 0
fi

docker run --rm \
    -u root \
    --network host \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace/hijack \
    -e INSIDE_TGS_BUILD=1 \
    "${BUILDER_IMAGE}" \
    bash -lc '
        set -euo pipefail
        # Try toolchain already present in image first.
        if command -v cmake >/dev/null 2>&1 && \
           command -v make >/dev/null 2>&1 && \
           command -v gcc >/dev/null 2>&1 && \
           command -v g++ >/dev/null 2>&1 && \
           command -v patchelf >/dev/null 2>&1; then
            ./build.sh
            exit 0
        fi

        # Some older images ship stale NVIDIA apt sources that break apt update.
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml*.list || true
        apt-get update
        apt-get install -y cmake make gcc g++ patchelf
        ./build.sh
    '