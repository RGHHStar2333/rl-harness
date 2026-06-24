#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python scripts/monitor_hermes.py --config configs/pipeline.yaml "$@"
