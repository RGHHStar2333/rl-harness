#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p runs

QUIET=0
if [ "${1:-}" = "--quiet" ]; then
  QUIET=1
fi

HOST="${HERMES_FEISHU_HOST:-0.0.0.0}"
PORT="${HERMES_FEISHU_PORT:-8765}"
STATE="runs/hermes_feishu_webhook.json"
LOG="runs/hermes_feishu_webhook.log"

is_webhook_pid() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -F "scripts/hermes_feishu_webhook.py" >/dev/null
}

if [ -f "$STATE" ]; then
  old_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "$STATE" 2>/dev/null || true)"
  if is_webhook_pid "$old_pid"; then
    if [ "$QUIET" -eq 0 ]; then
      echo "Hermes Feishu webhook already running: PID=$old_pid"
      echo "status: $STATE"
      echo "log: $LOG"
    fi
    exit 0
  fi
fi

PYTHONUNBUFFERED=1 nohup setsid python3 scripts/hermes_feishu_webhook.py --host "$HOST" --port "$PORT" >> "$LOG" 2>&1 &
PID=$!
sleep 0.5

if ! is_webhook_pid "$PID"; then
  python3 - "$STATE" "$PID" "$HOST" "$PORT" "$LOG" <<'PY'
import json
import sys
import time

state_path, pid, host, port, log_path = sys.argv[1:]
payload = {
    "status": "failed",
    "pid": int(pid),
    "host": host,
    "port": int(port),
    "log": log_path,
    "failed_at": time.time(),
    "command": f"python3 scripts/hermes_feishu_webhook.py --host {host} --port {port}",
}
with open(state_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
  echo "Hermes Feishu webhook failed to start. Check: $LOG"
  exit 1
fi

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
    "health_url": f"http://127.0.0.1:{port}/health",
    "command": f"python3 scripts/hermes_feishu_webhook.py --host {host} --port {port}",
}
with open(state_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

if [ "$QUIET" -eq 0 ]; then
  echo "Hermes Feishu webhook started"
  echo "PID: $PID"
  echo "URL: http://$HOST:$PORT"
  echo "Health: http://127.0.0.1:$PORT/health"
  echo "State: $STATE"
  echo "Log: $LOG"
fi
