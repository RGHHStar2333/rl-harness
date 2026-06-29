#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="$(python3 - <<'PY'
import yaml
cfg=yaml.safe_load(open("configs/tasks/mjlab/go1.yaml", "r", encoding="utf-8"))
print(cfg["mjlab"]["run_id"])
PY
)"

mkdir -p "runs/$RUN_ID"
LOG="runs/$RUN_ID/feedback_loop.log"

echo "🚀 启动 MJLab feedback loop"
echo "run_id: $RUN_ID"
echo "log: $LOG"

nohup bash -lc '
while true; do
  date
  python3 scripts/mjlab_parse_log.py
  python3 scripts/mjlab_auto_tune.py
  echo "----"
  sleep 60
done
' > "$LOG" 2>&1 &

PID=$!

echo "$PID" > "runs/$RUN_ID/feedback_loop.pid"

echo "✅ feedback loop 已后台启动"
echo "PID: $PID"
echo "看反馈日志：tail -f $LOG"
