#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STATE="runs/hermes_feishu_webhook.json"

if [ ! -f "$STATE" ]; then
  echo "Hermes Feishu webhook state not found."
  exit 0
fi

PID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "$STATE" 2>/dev/null || true)"
if [ -z "$PID" ]; then
  echo "No PID in $STATE"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Hermes Feishu webhook stopped: PID=$PID"
else
  echo "Hermes Feishu webhook was not running: PID=$PID"
fi

python3 - "$STATE" <<'PY'
import json
import sys
import time

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    data = {}
data["status"] = "stopped"
data["stopped_at"] = time.time()
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
