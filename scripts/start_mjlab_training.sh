#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/tasks/mjlab/go1.yaml"

RUN_ID="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["run_id"])')"
PROJECT_DIR="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["project_dir"])')"
TASK="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["task"])')"
NUM_ENVS="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["num_envs"])')"
WANDB_PROJECT_NAME="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["wandb_project"])')"
WANDB_RUN_NAME="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["wandb_name"])')"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "❌ 找不到 MJLab 目录: $PROJECT_DIR"
  exit 1
fi

mkdir -p "runs/$RUN_ID"
LOG_PATH="runs/$RUN_ID/training_process.log"

echo "🚀 启动 MJLab 训练"
echo "project_dir: $PROJECT_DIR"
echo "task: $TASK"
echo "num_envs: $NUM_ENVS"
echo "log: $LOG_PATH"

nohup bash -lc "cd '$PROJECT_DIR' && export WANDB_PROJECT='$WANDB_PROJECT_NAME' && export WANDB_NAME='$WANDB_RUN_NAME' && uv run train '$TASK' --env.scene.num-envs '$NUM_ENVS'" > "$LOG_PATH" 2>&1 &

PID=$!

cat > runs/active_training.json <<JSON
{
  "status": "running",
  "kind": "mjlab",
  "run_id": "$RUN_ID",
  "pid": $PID,
  "project_dir": "$PROJECT_DIR",
  "command": "uv run train $TASK --env.scene.num-envs $NUM_ENVS",
  "log": "$LOG_PATH"
}
JSON

echo "✅ MJLab 训练已后台启动"
echo "PID: $PID"
echo "看日志：tail -f $LOG_PATH"
