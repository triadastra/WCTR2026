#!/usr/bin/env bash
# Canonical Exp 3 entry point. plan/check do not train; train runs all 144 cells.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PY:-python}" scripts/run_exp3.py "${@:-train}"
