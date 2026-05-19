#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

pushd "$(dirname "$0")/.." >/dev/null

IMAGE_TAG="${1:-tf_torch:llamacpp-model}"
TOKEN_FILE="${HF_TOKEN_FILE:-docker/hf_token.txt}"
MODEL_REPO="${MODEL_REPO:-bartowski/Llama-3.2-3B-Instruct-GGUF}"
MODEL_INCLUDE="${MODEL_INCLUDE:-Llama-3.2-3B-Instruct-f16.gguf}"
MODEL_FILENAME="${MODEL_FILENAME:-Llama-3.2-3B-Instruct-f16.gguf}"
MODEL_SHA256="${MODEL_SHA256:-}"

echo "Building model image ${IMAGE_TAG} using base image tf_torch:llamacpp"

build_args=(
  --build-arg "MODEL_REPO=${MODEL_REPO}"
  --build-arg "MODEL_INCLUDE=${MODEL_INCLUDE}"
  --build-arg "MODEL_FILENAME=${MODEL_FILENAME}"
)

if [[ -n "${MODEL_SHA256}" ]]; then
  build_args+=(--build-arg "MODEL_SHA256=${MODEL_SHA256}")
fi

if [[ -f "${TOKEN_FILE}" ]]; then
  echo "Looking for Hugging Face token at ${TOKEN_FILE}"
  HF_TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
  if [[ -n "${HF_TOKEN}" ]]; then
    build_args+=(--build-arg "HF_TOKEN=${HF_TOKEN}")
    echo "Using Hugging Face token from ${TOKEN_FILE}."
  else
    echo "Token file ${TOKEN_FILE} is empty; building without token."
  fi
else
  echo "No ${TOKEN_FILE} found; building without token."
fi

docker build \
  -f docker/llamacpp_model.Dockerfile \
  -t "${IMAGE_TAG}" \
  "${build_args[@]}" \
  .

popd >/dev/null
