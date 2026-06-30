#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

state_last_notified_at() {
  local state="$1"
  if [ ! -f "$state" ]; then
    echo 0
    return
  fi

  "$PYTHON_BIN" - "$state" <<'PY'
import json
import sys

try:
    print(int(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("last_notified_at", 0)))
except Exception:
    print(0)
PY
}

state_write() {
  local state="$1"
  local now="$2"
  local reason="$3"
  "$PYTHON_BIN" - "$state" "$now" "$reason" <<'PY'
import json
import sys

path = sys.argv[1]
now = int(sys.argv[2])
reason = sys.argv[3]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"last_notified_at": now, "reason": reason}, f, ensure_ascii=False, indent=2)
PY
}

now_seconds() {
  "$PYTHON_BIN" - <<'PY'
import time
print(int(time.time()))
PY
}

read_cfg_bool() {
  local key="$1"
  local default="$2"
  "$PYTHON_BIN" - "$key" "$default" <<'PY'
import sys
import yaml

key = sys.argv[1]
default = sys.argv[2].lower() == "true"

cfg = yaml.safe_load(open("configs/tasks/mjlab/go1.yaml", "r", encoding="utf-8"))
value = cfg.get("autotune", {}).get(key, default)
print("true" if bool(value) else "false")
PY
}

ENABLED="$(read_cfg_bool wandb_autodecide_enabled false)"
APPLY="$(read_cfg_bool wandb_autodecide_apply false)"

if [ "$ENABLED" != "true" ]; then
  echo "W&B MJLab autodecide disabled."
  exit 0
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import wandb  # noqa: F401
PY
then
  mkdir -p runs
  STATE="runs/_wandb_autodecide_dependency_state.json"
  NOW="$(now_seconds)"
  LAST="$(state_last_notified_at "$STATE")"

  if [ $((NOW - LAST)) -ge 86400 ]; then
    echo "W&B MJLab autodecide skipped: 当前 Python 缺少 wandb；普通训练监控继续运行。"
    echo "安装后会自动恢复：$PYTHON_BIN -m pip install wandb"
    state_write "$STATE" "$NOW" "missing wandb"
  fi
  exit 0
fi

ARGS=()
if [ "$APPLY" = "true" ]; then
  ARGS+=(--apply --no-restart)
fi

OUTPUT="$(mktemp)"
if "$PYTHON_BIN" scripts/wandb_mjlab_autodecide.py "${ARGS[@]}" >"$OUTPUT" 2>&1; then
  cat "$OUTPUT"
  rm -f "$OUTPUT"
  exit 0
fi

mkdir -p runs
STATE="runs/_wandb_autodecide_failure_state.json"
NOW="$(now_seconds)"
LAST="$(state_last_notified_at "$STATE")"

if [ $((NOW - LAST)) -ge 86400 ]; then
  echo "W&B MJLab autodecide skipped: W&B 曲线读取失败；普通训练监控继续运行。"
  echo "最近错误摘要："
  tail -n 8 "$OUTPUT"
  state_write "$STATE" "$NOW" "wandb autodecide failed"
fi

rm -f "$OUTPUT"
exit 0
