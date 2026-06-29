#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/tasks/mjlab/go1.yaml"
DRY_RUN=false

case "${1:-}" in
  --dry-run|--print-command)
    DRY_RUN=true
    shift
    ;;
esac

RUN_ID="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["run_id"])')"
PROJECT_DIR="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["project_dir"])')"
TASK="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["task"])')"
NUM_ENVS="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["num_envs"])')"
MAX_ITERATIONS="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["max_iterations"])')"
LEARNING_RATE="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["agent"]["algorithm"]["learning_rate"])')"
WANDB_PROJECT_NAME="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["wandb_project"])')"
WANDB_RUN_NAME="$(python3 -c 'import yaml; c=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml")); print(c["mjlab"]["wandb_name"])')"

TRAIN_CMD=(
  uv run train "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITERATIONS"
  --agent.logger wandb
  --agent.wandb-project "$WANDB_PROJECT_NAME"
  --agent.run-name "$WANDB_RUN_NAME"
  --agent.algorithm.learning-rate "$LEARNING_RATE"
)

printf -v TRAIN_CMD_STR "%q " "${TRAIN_CMD[@]}"
COMMAND_TEXT="${TRAIN_CMD[*]}"

mkdir -p "runs/$RUN_ID"
LOG_PATH="runs/$RUN_ID/training_process.log"

echo "🚀 启动 MJLab 训练"
echo "project_dir: $PROJECT_DIR"
echo "task: $TASK"
echo "num_envs: $NUM_ENVS"
echo "max_iterations: $MAX_ITERATIONS"
echo "learning_rate: $LEARNING_RATE"
echo "log: $LOG_PATH"

if [ "$DRY_RUN" = true ]; then
  echo "dry_run: true"
  echo "command: $COMMAND_TEXT"
  exit 0
fi

if [ ! -d "$PROJECT_DIR" ]; then
  echo "❌ 找不到 MJLab 目录: $PROJECT_DIR"
  exit 1
fi

PROJECT_DIR_Q="$(printf "%q" "$PROJECT_DIR")"
WANDB_PROJECT_Q="$(printf "%q" "$WANDB_PROJECT_NAME")"
WANDB_NAME_Q="$(printf "%q" "$WANDB_RUN_NAME")"
RUN_SHELL_CMD="cd $PROJECT_DIR_Q && export WANDB_PROJECT=$WANDB_PROJECT_Q && export WANDB_NAME=$WANDB_NAME_Q && exec $TRAIN_CMD_STR"

nohup bash -lc "$RUN_SHELL_CMD" > "$LOG_PATH" 2>&1 &

PID=$!

cat > runs/active_training.json <<JSON
{
  "status": "running",
  "kind": "mjlab",
  "run_id": "$RUN_ID",
  "pid": $PID,
  "project_dir": "$PROJECT_DIR",
  "command": "$COMMAND_TEXT",
  "learning_rate": $LEARNING_RATE,
  "log": "$LOG_PATH"
}
JSON

echo "✅ MJLab 训练已后台启动"
echo "PID: $PID"
echo "看日志：tail -f $LOG_PATH"
