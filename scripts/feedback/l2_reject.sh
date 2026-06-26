#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 scripts/feedback/l2_decide.py --decision reject --token "$1"
python3 scripts/ops/git_auto_commit.py --level L2 --reason "rejected L2 token $1" || true
