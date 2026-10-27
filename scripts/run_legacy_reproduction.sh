#!/usr/bin/env bash
# Separate opt-in entry point; no Census receipt, replacement panel, or download.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3}"
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  check)
    [[ $# -eq 0 ]] || { echo "check takes no arguments" >&2; exit 2; }
    exec "$PY" scripts/check_legacy_data.py
    ;;
  train|infer)
    "$PY" scripts/check_legacy_data.py
    if [[ "$mode" == train ]]; then
      exec "$PY" scripts/train.py --legacy-reproduction "$@"
    else
      exec "$PY" scripts/inference.py --legacy-reproduction "$@"
    fi
    ;;
  *) echo "Usage: bash scripts/run_legacy_reproduction.sh [check|train|infer] [arguments]" >&2; exit 2 ;;
esac
