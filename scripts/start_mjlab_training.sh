#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/mjlab/start_mjlab_training.sh "$@"
