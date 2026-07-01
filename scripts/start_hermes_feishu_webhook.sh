#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p runs

HOST="${HERMES_FEISHU_HOST:-0.0.0.0}"
PORT="${HERMES_FEISHU_PORT:-8765}"
STATE="runs/hermes_feishu_webhook.json"
LOG="runs/hermes_feishu_webhook.log"

if [ -f "$STATE" ]; then
  old_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "$STATE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Hermes Feishu webhook already running: PID=$old_pid"
    echo "status: $STATE"
    echo "log: $LOG"
    exit 0
  fi
fi

PYTHONUNBUFFERED=1 nohup setsid python3 scripts/hermes_feishu_webhook.py --host "$HOST" --port "$PORT" >> "$LOG" 2>&1 &
PID=$!

python3 - "$STATE" "$PID" "$HOST" "$PORT" "$LOG" <<'PY'
import json
import sys
import time

state_path, pid, host, port, log_path = sys.argv[1:]
payload = {
    "status": "running",
    "pid": int(pid),
    "host": host,
    "port": int(port),
    "log": log_path,
    "started_at": time.time(),
    "command": f"python3 scripts/hermes_feishu_webhook.py --host {host} --port {port}",
}
with open(state_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

echo "Hermes Feishu webhook started"
echo "PID: $PID"
echo "URL: http://$HOST:$PORT"
echo "State: $STATE"
echo "Log: $LOG"
