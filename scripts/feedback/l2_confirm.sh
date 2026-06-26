#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f runs/adjustments.jsonl ]; then
  before_lines=$(wc -l < runs/adjustments.jsonl)
else
  before_lines=0
fi

python3 scripts/feedback/l2_decide.py --decision confirm --token "$1"
python3 scripts/feedback/auto_restart_if_needed.py --since-line "$before_lines" || true
python3 scripts/ops/git_auto_commit.py --level L2 --reason "confirmed L2 token $1 with restart" || true
