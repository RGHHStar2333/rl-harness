#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/mjlab/run_g1_feedback.sh "$@"
