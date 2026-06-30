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
AUTO_RESUME="$(get_yaml_value auto_resume || echo true)"
CHECKPOINT_SUBDIR="$(get_yaml_value checkpoint_subdir || echo logs/rsl_rl/g1_velocity)"
LEARNING_RATE="$(
  python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/tasks/mjlab/go1.yaml", "r", encoding="utf-8"))
print(cfg.get("agent", {}).get("learning_rate", ""))
PY
)"

EXTRA_ARGS=()
if [ -n "$LEARNING_RATE" ]; then
  EXTRA_ARGS+=(--agent.algorithm.learning-rate "$LEARNING_RATE")
fi

while IFS= read -r arg; do
  EXTRA_ARGS+=("$arg")
done < <(
  python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/tasks/mjlab/go1.yaml", "r", encoding="utf-8"))
for name, value in cfg.get("reward_weights", {}).items():
    print(f"--env.rewards.{name.replace('_', '-')}.weight")
    print(value)
PY
)

RUN_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

LOG_PATH="$RUN_DIR/training_process.log"
RESTART_LOG="$RUN_DIR/restart_history.jsonl"
CHECKPOINT_INDEX="$RUN_DIR/checkpoint_index.jsonl"

echo "🔎 查找 MJLab checkpoint"
echo "run_id: $RUN_ID"
echo "project_dir: $PROJECT_DIR"
echo "checkpoint_subdir: $CHECKPOINT_SUBDIR"

MJLAB_CKPT_ROOT="$PROJECT_DIR/$CHECKPOINT_SUBDIR"

if [ ! -d "$MJLAB_CKPT_ROOT" ]; then
  echo "❌ checkpoint 目录不存在: $MJLAB_CKPT_ROOT"
  exit 1
fi

LATEST_CKPT="$(
  find "$MJLAB_CKPT_ROOT" -type f -name "model_*.pt" | grep "$RUN_ID" | sort -V | tail -n 1 || true
)"

if [ -z "$LATEST_CKPT" ]; then
  echo "⚠️ 没找到包含 run_id=$RUN_ID 的 checkpoint"
  echo "将退回普通启动，不使用 checkpoint resume"

  RESUME_ARGS=()
  LOAD_RUN=""
  LOAD_CHECKPOINT=""
else
  LOAD_RUN="$(basename "$(dirname "$LATEST_CKPT")")"
  LOAD_CHECKPOINT="$(basename "$LATEST_CKPT")"

  echo "✅ 找到最新 checkpoint:"
  echo "$LATEST_CKPT"
  echo "load_run: $LOAD_RUN"
  echo "load_checkpoint: $LOAD_CHECKPOINT"

  RESUME_ARGS=(
    --agent.resume True
    --agent.load-run "$LOAD_RUN"
    --agent.load-checkpoint "$LOAD_CHECKPOINT"
  )
fi

echo "🧹 停止旧的 MJLab 训练/播放进程"
python3 scripts/pause_training.py || true
pkill -f "uv run train $TASK" 2>/dev/null || true
pkill -f "uv run play $TASK" 2>/dev/null || true
sleep 2

TIMESTAMP="$(date -Iseconds)"

if [ -n "${LATEST_CKPT:-}" ]; then
  cat >> "$CHECKPOINT_INDEX" <<JSON
{"time":"$TIMESTAMP","run_id":"$RUN_ID","checkpoint":"$LATEST_CKPT","load_run":"$LOAD_RUN","load_checkpoint":"$LOAD_CHECKPOINT"}
JSON
fi

cat >> "$RESTART_LOG" <<JSON
{"time":"$TIMESTAMP","event":"restart_from_checkpoint","run_id":"$RUN_ID","checkpoint":"${LATEST_CKPT:-none}","num_envs":"$NUM_ENVS","max_iterations":"$MAX_ITERATIONS"}
JSON

echo "🚀 从 checkpoint 自动重启 MJLab 训练"
echo "task: $TASK"
echo "num_envs: $NUM_ENVS"
echo "max_iterations: $MAX_ITERATIONS"
echo "learning_rate: ${LEARNING_RATE:-none}"
echo "wandb_project: $WANDB_PROJECT_NAME"
echo "wandb_name: $WANDB_RUN_NAME"
echo "log: $LOG_PATH"

cd "$PROJECT_DIR"

nohup uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.logger wandb \
  --agent.wandb-project "$WANDB_PROJECT_NAME" \
  --agent.run-name "$WANDB_RUN_NAME" \
  "${EXTRA_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  > "$LOG_PATH" 2>&1 &

PID=$!

cd "$ROOT"

cat > runs/active_training.json <<JSON
{
  "status": "running",
  "kind": "mjlab",
  "run_id": "$RUN_ID",
  "pid": $PID,
  "project_dir": "$PROJECT_DIR",
  "checkpoint": "${LATEST_CKPT:-none}",
  "load_run": "${LOAD_RUN:-}",
  "load_checkpoint": "${LOAD_CHECKPOINT:-}",
  "command": "uv run train $TASK --env.scene.num-envs $NUM_ENVS --agent.max-iterations $MAX_ITERATIONS --agent.logger wandb --agent.wandb-project $WANDB_PROJECT_NAME --agent.run-name $WANDB_RUN_NAME ${EXTRA_ARGS[*]} ${RESUME_ARGS[*]}",
  "learning_rate": "${LEARNING_RATE:-}",
  "log": "$LOG_PATH"
}
JSON

echo "✅ MJLab 已从 checkpoint 后台重启"
echo "PID: $PID"
echo "看日志：tail -f $LOG_PATH"
