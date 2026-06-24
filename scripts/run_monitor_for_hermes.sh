#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  python scripts/monitor_hermes.py --config configs/pipeline.yaml "$@"
else
  python3 scripts/monitor_hermes.py --config configs/pipeline.yaml "$@"
fi
