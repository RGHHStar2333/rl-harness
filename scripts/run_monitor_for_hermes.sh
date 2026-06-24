#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python3 scripts/monitor_hermes.py --config configs/pipeline.yaml "$@"

if [ "$#" -eq 0 ] || [ "${1:-}" = "--debug" ]; then
  python3 scripts/l2_check.py --config configs/pipeline.yaml "$@"
fi
