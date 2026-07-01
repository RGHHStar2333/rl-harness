#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "${HERMES_FEISHU_DISABLE_WEBHOOK_AUTOSTART:-0}" = "1" ]; then
  exit 0
fi

STATE="runs/hermes_feishu_webhook.json"

is_webhook_pid() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -F "scripts/hermes_feishu_webhook.py" >/dev/null
}

if [ -f "$STATE" ]; then
  PID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "$STATE" 2>/dev/null || true)"
  if is_webhook_pid "$PID"; then
    exit 0
  fi
fi

bash scripts/start_hermes_feishu_webhook.sh --quiet
