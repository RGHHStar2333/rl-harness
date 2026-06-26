#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="configs/tasks/mjlab/go1.yaml"

RUN_ID="$(awk '/run_id:/ {print $2; exit}' "$CFG")"
PROJECT_DIR="$(awk '/project_dir:/ {print $2; exit}' "$CFG")"
TASK="$(awk '/task:/ {print $2; exit}' "$CFG")"
NUM_ENVS="$(awk '/num_envs:/ {print $2; exit}' "$CFG")"
MAX_ITERATIONS="$(awk '/max_iterations:/ {print $2; exit}' "$CFG")"
WANDB_PROJECT_NAME="$(awk '/wandb_project:/ {print $2; exit}' "$CFG")"
WANDB_RUN_NAME="$(awk '/wandb_name:/ {print $2; exit}' "$CFG")"

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
