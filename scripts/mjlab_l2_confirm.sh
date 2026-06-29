#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TOKEN="${1:-}"

if [ -z "$TOKEN" ]; then
  echo "用法: bash scripts/mjlab_l2_confirm.sh TOKEN"
  exit 1
fi

python3 scripts/mjlab_auto_tune.py --confirm "$TOKEN"
