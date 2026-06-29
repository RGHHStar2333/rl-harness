import argparse
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_adjustments():
    path = os.path.join(ROOT, "runs", "adjustments.jsonl")
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def should_restart(row):
    if row.get("trainer_kind") == "mjlab":
        return False

    level = row.get("level")

    if level == "L1":
        return True

    if level == "L2" and row.get("decision") == "confirmed":
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-line", type=int, required=True)
    args = parser.parse_args()

    rows = read_adjustments()
    new_rows = rows[args.since_line:]

    restart_rows = [r for r in new_rows if should_restart(r)]
    mjlab_restart_rows = [
        r
        for r in new_rows
        if r.get("trainer_kind") == "mjlab" and r.get("restart_required")
    ]

    if not restart_rows:
        if mjlab_restart_rows:
            last = mjlab_restart_rows[-1]
            print("ℹ️ 检测到 MJLab 参数调整：不会自动调用 HalfCheetah checkpoint 重启。")
            print(f"profile: {last.get('profile_name')}")
            print(f"target: {last.get('target')}")
            print(f"old_value: {last.get('old_value')}")
            print(f"new_value: {last.get('new_value')}")
            print("说明：MJLab 调整会在重启训练或下次启动时生效。")
            return

        print("✅ 没有新的 L1/L2 confirmed 调整，不需要重启。")
        return

    last = restart_rows[-1]

    print("🔁 检测到新的参数调整，需要从 checkpoint 重启训练。")
    print(f"level: {last.get('level')}")
    print(f"target: {last.get('target')}")
    print(f"old_value: {last.get('old_value')}")
    print(f"new_value: {last.get('new_value')}")

    result = subprocess.run(
        ["bash", "scripts/training/restart_from_checkpoint.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
