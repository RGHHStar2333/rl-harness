#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p runs

state_path="runs/active_training.json"
log_path="runs/training_process.log"

if [ -f "$state_path" ]; then
  old_pid=$(python3 -c "import json; print(json.load(open(\"$state_path\")).get(\"pid\", \"\"))" 2>/dev/null || true)
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "⚠️ 已经有训练进程在运行，PID: $old_pid"
    echo "先运行：python3 scripts/training/pause_training.py --status"
    exit 1
  fi
fi

echo "🚀 启动训练..."
nohup setsid python3 scripts/training/train.py --config configs/pipeline.yaml >> "$log_path" 2>&1 &
pid=$!

printf '{\n  "pid": %s,\n  "status": "running",\n  "started_at": %s,\n  "command": "python3 scripts/training/train.py --config configs/pipeline.yaml",\n  "cwd": "%s",\n  "log_path": "%s"\n}\n' "$pid" "$(date +%s)" "$(pwd)" "$log_path" > "$state_path"

echo "✅ 训练已后台启动"
echo "PID: $pid"
echo "日志: $log_path"
echo "状态: $state_path"
