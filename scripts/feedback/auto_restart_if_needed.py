import argparse
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_adjustments(path):
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


def is_mjlab_row(row):
    run_id = str(row.get("run_id", ""))
    return (
        row.get("trainer_kind") == "mjlab"
        or row.get("system") == "mjlab"
        or run_id.startswith("mjlab")
    )


def row_requires_restart(row):
    return bool(row.get("restart_required") or row.get("requires_restart"))


def should_restart_sb3(row):
    if is_mjlab_row(row):
        return False

    level = row.get("level")

    if level == "L1":
        return True

    if level == "L2" and row.get("decision") == "confirmed":
        return True

    return False


def should_restart_mjlab(row):
    if not is_mjlab_row(row) or not row_requires_restart(row):
        return False

    if row.get("trainer_kind") == "mjlab":
        decision = row.get("decision")
        if decision and decision != "confirmed":
            return False

    level = row.get("level")

    if level == "L1":
        return True

    if level == "L2" and row.get("decision", "confirmed") == "confirmed":
        return True

    return False


def has_l3_emergency(rows):
    return any(row.get("level") == "L3" for row in rows)


def run_restart(command, dry_run):
    if dry_run:
        print("dry_run: true")
        print("would_run:", " ".join(command))
        return 0

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    print(result.stdout)
    print(result.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-line", type=int, required=True)
    parser.add_argument("--adjustments-path", default="runs/adjustments.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sb3-restart-cmd", default="bash scripts/training/restart_from_checkpoint.sh")
    parser.add_argument("--mjlab-restart-cmd", default="bash scripts/restart_mjlab_from_checkpoint.sh")
    args = parser.parse_args()

    adjustments_path = (
        args.adjustments_path
        if os.path.isabs(args.adjustments_path)
        else os.path.join(ROOT, args.adjustments_path)
    )

    rows = read_adjustments(adjustments_path)
    new_rows = rows[args.since_line:]

    if has_l3_emergency(new_rows):
        print("🛑 检测到新的 L3 紧急记录，本轮跳过自动重启。")
        print("说明：L3 暂停优先级高于 L1/L2 参数调整，请人工检查后再恢复训练。")
        return

    restart_rows = [r for r in new_rows if should_restart_sb3(r)]
    mjlab_restart_rows = [r for r in new_rows if should_restart_mjlab(r)]

    if mjlab_restart_rows:
        last = mjlab_restart_rows[-1]

        print("🔁 检测到新的 MJLab 参数调整，需要从 MJLab checkpoint 自动重启。")
        print(f"level: {last.get('level')}")
        print(f"target: {last.get('target') or last.get('changes')}")
        print(f"old_value: {last.get('old_value')}")
        print(f"new_value: {last.get('new_value')}")

        code = run_restart(args.mjlab_restart_cmd.split(), args.dry_run)
        if code != 0:
            raise SystemExit(code)
        return

    if not restart_rows:
        print("✅ 没有新的 L1/L2 confirmed 调整，不需要重启。")
        return

    last = restart_rows[-1]

    print("🔁 检测到新的参数调整，需要从 checkpoint 重启训练。")
    print(f"level: {last.get('level')}")
    print(f"target: {last.get('target')}")
    print(f"old_value: {last.get('old_value')}")
    print(f"new_value: {last.get('new_value')}")

    code = run_restart(args.sb3_restart_cmd.split(), args.dry_run)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
