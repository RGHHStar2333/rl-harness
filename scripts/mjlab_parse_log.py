import argparse
import json
import re
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

KEY_MAP = {
    "Total steps": "train/total_steps",
    "Steps per second": "time/fps",
    "Collection time": "time/collection_time",
    "Learning time": "time/learning_time",
    "Mean value loss": "train/value_loss",
    "Mean surrogate loss": "train/surrogate_loss",
    "Mean entropy loss": "train/entropy_loss",
    "Mean reward": "train/episode_reward_mean",
    "Mean episode length": "train/episode_length_mean",
    "Mean action std": "train/action_std",
    "Iteration time": "time/iteration_time",
}

def clean(line):
    return ANSI_RE.sub("", line).strip()

def load_cfg():
    return yaml.safe_load((ROOT / "configs/tasks/mjlab/go1.yaml").read_text(encoding="utf-8"))

def parse_float(s):
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    run_id = args.run_id or cfg["mjlab"]["run_id"]

    run_dir = ROOT / "runs" / run_id
    log_path = run_dir / "training_process.log"
    out_path = run_dir / "train.jsonl"

    if not log_path.exists():
        print(f"❌ 找不到日志: {log_path}")
        return

    rows = []
    current = None

    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = clean(raw)

        m_iter = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
        if m_iter:
            if current and "mjlab/iteration" in current:
                rows.append(current)
            current = {
                "timestamp": time.time(),
                "mjlab/iteration": int(m_iter.group(1)),
                "mjlab/max_iterations": int(m_iter.group(2)),
            }
            continue

        if current is None:
            continue

        m = re.match(r"(.+?):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$", line)
        if not m:
            continue

        key = m.group(1).strip()
        val = parse_float(m.group(2).strip())
        if val is None:
            continue

        mapped = KEY_MAP.get(key, key.replace(" ", "_"))
        current[mapped] = val

    if current and "mjlab/iteration" in current:
        rows.append(current)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    print(f"✅ parsed {len(rows)} rows -> {out_path}")
    if rows:
        last = rows[-1]
        print("latest iteration:", last.get("mjlab/iteration"))
        print("latest reward:", last.get("train/episode_reward_mean"))
        print("latest error_vel_xy:", last.get("Metrics/twist/error_vel_xy"))

if __name__ == "__main__":
    main()
