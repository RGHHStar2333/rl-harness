import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
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


def load_state(path):
    if not os.path.exists(path):
        return {"paused": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {"paused": {}}

    state.setdefault("paused", {})
    return state


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_metric_points(rows, metric):
    points = []

    for row in rows:
        value = row.get(metric)
        step = row.get("train/step")

        if value is None:
            continue

        try:
            points.append({
                "step": int(step) if step is not None else 0,
                "value": float(value),
            })
        except Exception:
            pass

    return points


def check_rule(rule, points):
    if not points:
        return False, "没有足够数据", None

    condition = rule["condition"]
    ctype = condition["type"]
    latest = points[-1]

    if ctype == "threshold":
        current = latest["value"]
        op = condition["operator"]
        target = float(condition["value"])

        if op == "<" and current < target:
            return True, f"当前值 {current:.3f} < 阈值 {target:.3f}", latest

        if op == ">" and current > target:
            return True, f"当前值 {current:.3f} > 阈值 {target:.3f}", latest

    if ctype == "trend":
        window = int(condition["window"])
        min_delta = float(condition["min_delta"])

        if len(points) < window:
            return False, "窗口数据不足", latest

        recent = points[-window:]
        delta = recent[-1]["value"] - recent[0]["value"]

        if delta < min_delta:
            return True, f"最近 {window} 个记录增长 {delta:.3f}，低于要求 {min_delta:.3f}", latest

    if ctype == "volatility":
        window = int(condition["window"])
        max_std = float(condition["max_std"])

        if len(points) < window:
            return False, "窗口数据不足", latest

        recent_values = [p["value"] for p in points[-window:]]
        std = statistics.pstdev(recent_values)

        if std > max_std:
            return True, f"最近 {window} 个记录波动 {std:.3f}，超过 {max_std:.3f}", latest

    return False, "未触发", latest


def run_pause(reason):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/training/pause_training.py",
            "--reason",
            reason,
            "--force",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    return result.returncode, result.stdout + result.stderr


def format_l3_message(pipeline, rule, reason, latest, pause_output):
    project = pipeline["project"]

    lines = []
    lines.append("🔴 RL Harness L3 紧急告警")
    lines.append("")
    lines.append(f"项目：{project['name']}")
    lines.append(f"任务：{project['task']}")
    lines.append(f"Run ID：{project['run_id']}")
    lines.append(f"规则：{rule['id']}")
    lines.append(f"指标：{rule.get('metric')}")

    if latest:
        lines.append(f"最新 step：{latest.get('step')}")
        lines.append(f"最新值：{latest.get('value'):.3f}")

    lines.append(f"触发原因：{reason}")
    lines.append("")
    lines.append("执行动作：pause_training")
    lines.append("")
    lines.append("暂停结果：")
    lines.append(pause_output.strip())
    lines.append("")
    lines.append("下一步：请人工检查训练日志、checkpoint 和 reward 曲线，再决定是否恢复训练。")
    lines.append(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force-l3-test", action="store_true")
    args = parser.parse_args()

    pipeline = load_yaml(args.config)
    rules_cfg = load_yaml(pipeline["paths"]["detection_rules"])

    state_path = os.path.join(ROOT, "runs", "_l3_pause_state.json")
    state = load_state(state_path)

    l3_rules = [r for r in rules_cfg["rules"] if r.get("response_level") == "L3"]

    if args.force_l3_test:
        if not l3_rules:
            print("❌ detection_rules.yaml 里没有 L3 规则。")
            return

        rule = l3_rules[0]
        latest = {"step": 0, "value": 0.0}
        reason = "force-l3-test 手动测试触发"

        code, pause_output = run_pause(reason)
        print(format_l3_message(pipeline, rule, reason, latest, pause_output))
        return

    run_dir = os.path.join(ROOT, pipeline["paths"]["run_dir"])
    log_path = os.path.join(run_dir, "train.jsonl")
    rows = load_jsonl(log_path)

    if not rows:
        if args.debug:
            print("L3：暂无训练日志。")
        return

    messages = []

    for rule in l3_rules:
        metric = rule["metric"]
        points = get_metric_points(rows, metric)

        triggered, reason, latest = check_rule(rule, points)

        if not triggered:
            continue

        latest_step = int(latest.get("step", 0)) if latest else 0
        pause_key = f"{pipeline['project']['run_id']}::{rule['id']}::{latest_step}"

        if pause_key in state["paused"]:
            continue

        code, pause_output = run_pause(reason)

        state["paused"][pause_key] = {
            "time": time.time(),
            "rule_id": rule["id"],
            "step": latest_step,
            "reason": reason,
            "returncode": code,
        }

        messages.append(format_l3_message(pipeline, rule, reason, latest, pause_output))

    save_state(state_path, state)

    if messages:
        print("\n\n---\n\n".join(messages))
    elif args.debug:
        print("L3：没有触发新的紧急暂停。")


if __name__ == "__main__":
    main()
