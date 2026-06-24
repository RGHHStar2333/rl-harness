#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

if [ -f runs/adjustments.jsonl ]; then
  before_lines=$(wc -l < runs/adjustments.jsonl)
else
  before_lines=0
fi

case "${1:-}" in
  --force-l3-test)
    python3 scripts/l3_check.py --config configs/pipeline.yaml --force-l3-test
    ;;
  *)
    python3 scripts/monitor_hermes.py --config configs/pipeline.yaml "$@"
    if [ "$#" -eq 0 ] || [ "${1:-}" = "--debug" ]; then
      python3 scripts/l2_check.py --config configs/pipeline.yaml "$@"
      python3 scripts/l3_check.py --config configs/pipeline.yaml "$@"
    fi
    ;;
esac

python3 scripts/auto_restart_if_needed.py --since-line "$before_lines" || true
python3 scripts/git_auto_commit.py --level AUTO --reason "post monitor harness update with restart" || true
