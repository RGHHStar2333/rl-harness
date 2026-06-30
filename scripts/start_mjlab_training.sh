#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="configs/tasks/mjlab/go1.yaml"

get_yaml_value() {
  local key="$1"
  awk -v k="$key" '$1 == k ":" {print $2; exit}' "$CFG"
}

RUN_ID="$(get_yaml_value run_id)"
PROJECT_DIR="$(get_yaml_value project_dir)"
TASK="$(get_yaml_value task)"
NUM_ENVS="$(get_yaml_value num_envs)"
MAX_ITERATIONS="$(get_yaml_value max_iterations)"
WANDB_PROJECT_NAME="$(get_yaml_value wandb_project)"
WANDB_RUN_NAME="$(get_yaml_value wandb_name)"

mkdir -p "runs/$RUN_ID"
LOG_PATH="runs/$RUN_ID/training_process.log"

echo "🚀 启动 MJLab 训练"
echo "run_id: $RUN_ID"
echo "project_dir: $PROJECT_DIR"
echo "task: $TASK"
echo "num_envs: $NUM_ENVS"
echo "max_iterations: $MAX_ITERATIONS"
echo "wandb_project: $WANDB_PROJECT_NAME"
echo "wandb_name: $WANDB_RUN_NAME"
echo "log: $LOG_PATH"

nohup bash -lc "cd '$PROJECT_DIR' && exec uv run train '$TASK' --env.scene.num-envs '$NUM_ENVS' --agent.max-iterations '$MAX_ITERATIONS' --agent.logger wandb --agent.wandb-project '$WANDB_PROJECT_NAME' --agent.run-name '$WANDB_RUN_NAME'" \
  > "$LOG_PATH" 2>&1 &

PID=$!

cat > runs/active_training.json <<JSON
{
  "status": "running",
  "kind": "mjlab",
  "run_id": "$RUN_ID",
  "pid": $PID,
  "project_dir": "$PROJECT_DIR",
  "command": "uv run train $TASK --env.scene.num-envs $NUM_ENVS --agent.max-iterations $MAX_ITERATIONS --agent.logger wandb --agent.wandb-project $WANDB_PROJECT_NAME --agent.run-name $WANDB_RUN_NAME",
  "log": "$LOG_PATH"
}
JSON

echo "✅ MJLab 训练已后台启动"
echo "PID: $PID"
echo "看日志：tail -f $LOG_PATH"
