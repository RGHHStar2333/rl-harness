#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PROJECT_DIR="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["project_dir"])')"
TASK="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["task"])')"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "❌ 找不到 MJLab 目录: $PROJECT_DIR"
  exit 1
fi

echo "▶️ 播放 MJLab 机器人"
echo "project_dir: $PROJECT_DIR"
echo "task: $TASK"

cd "$PROJECT_DIR"
uv run play "$TASK"
