#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null
# mirror test_tgs.sh: capture stdout/stderr and save to backup_logs/test_tgs.log
# mkdir -p backup_logs
python worker.py --trace config/test_text_inference_tgs.csv --log_path results/test_text_inference_results.csv --gpus 0 2>&1 | tee backup_logs/test_tgs.log

popd >/dev/null
