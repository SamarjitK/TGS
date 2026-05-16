#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

# Text inference with vLLM backend
# Runs self-contained in Docker container - no external setup needed

python worker.py --trace config/test_text_inference_vllm.csv --log_path results/test_text_inference_vllm_results.csv --gpus 0 2>&1 | tee backup_logs/test_text_inference_vllm.log
popd >/dev/null
