#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

python worker.py --trace config/test_text_inference_llamacpp.csv --log_path results/test_text_inference_llamacpp_results.csv --gpus 0 2>&1 | tee backup_logs/test_text_inference_llamacpp.log

popd >/dev/null