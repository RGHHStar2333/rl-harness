#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/l2_decide.py --decision confirm --token "$1"
python3 scripts/git_auto_commit.py --level L2 --reason "confirmed L2 token $1" || true
