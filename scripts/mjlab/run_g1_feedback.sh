#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/tasks/mjlab/feedback.yaml"
MJLAB_CONFIG="configs/tasks/mjlab/go1.yaml"

if [ ! -f "$CONFIG" ]; then
  echo "MJLab G1 feedback profile not found: $CONFIG"
  exit 1
fi

if [ "${MJLAB_SKIP_PARSE:-0}" != "1" ]; then
  python3 scripts/mjlab/parse_mjlab_metrics.py --config "$MJLAB_CONFIG" || true
fi

python3 scripts/feedback/monitor_hermes.py --config "$CONFIG" "$@"

if [ "$#" -eq 0 ] || [ "${1:-}" = "--debug" ] || [ "${1:-}" = "--allow-inactive" ]; then
  python3 scripts/feedback/l2_check.py --config "$CONFIG" "$@"
  python3 scripts/feedback/l3_check.py --config "$CONFIG" "$@"
fi
