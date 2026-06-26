#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

dry_run="false"

if [ "${1:-}" = "--dry-run" ]; then
  dry_run="true"
fi

checkpoint="$(python3 scripts/training/find_latest_checkpoint.py --config configs/pipeline.yaml)"

echo "🔁 准备从最近 checkpoint 重启训练"
echo "checkpoint: $checkpoint"

if [ "$dry_run" = "true" ]; then
  echo "dry-run：不暂停、不启动，只检查 checkpoint。"
  exit 0
fi

mkdir -p runs

echo "🛑 先暂停旧训练进程，如果没有运行则忽略。"
python3 scripts/training/pause_training.py --reason "restart from latest checkpoint" --force || true

state_path="runs/active_training.json"
log_path="runs/training_process.log"

echo "🚀 从 checkpoint 后台重启训练..."
nohup setsid python3 scripts/training/train.py --config configs/pipeline.yaml --resume-checkpoint "$checkpoint" >> "$log_path" 2>&1 &
pid=$!

printf '{\n  "pid": %s,\n  "status": "running",\n  "started_at": %s,\n  "command": "python3 scripts/training/train.py --config configs/pipeline.yaml --resume-checkpoint %s",\n  "resume_checkpoint": "%s",\n  "cwd": "%s",\n  "log_path": "%s"\n}\n' "$pid" "$(date +%s)" "$checkpoint" "$checkpoint" "$(pwd)" "$log_path" > "$state_path"

echo "✅ 已从 checkpoint 重启训练"
echo "PID: $pid"
echo "checkpoint: $checkpoint"
echo "日志: $log_path"
echo "状态: $state_path"
