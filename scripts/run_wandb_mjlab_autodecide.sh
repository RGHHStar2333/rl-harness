#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

read_cfg_bool() {
  local key="$1"
  local default="$2"
  python3 - "$key" "$default" <<'PY'
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

ARGS=()
if [ "$APPLY" = "true" ]; then
  ARGS+=(--apply --no-restart)
fi

python3 scripts/wandb_mjlab_autodecide.py "${ARGS[@]}"
