import argparse
import json
import os
import statistics
import time

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_state(path):
    if not os.path.exists(path):
        return {"notified": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"notified": {}}


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

        points.append(
            {
                "step": step,
                "value": float(value),
            }
        )

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


def format_alert(pipeline, rule, reason, latest):
    project = pipeline["project"]
    level = rule["response_level"]

    lines = []
    lines.append("🚨 RL Harness 训练反馈")
    lines.append("")
    lines.append(f"项目：{project['name']}")
    lines.append(f"任务：{project['task']}")
    lines.append(f"Run ID：{project['run_id']}")
    lines.append(f"规则：{rule['id']}")
    lines.append(f"级别：{level}")
    lines.append(f"指标：{rule['metric']}")

    if latest:
        lines.append(f"最新 step：{latest.get('step')}")
        lines.append(f"最新值：{latest.get('value'):.3f}")

    lines.append(f"触发原因：{reason}")
    lines.append("")

    if level == "L1":
        lines.append("处理方式：L1 自动级。")
        lines.append("当前 MVP 版本先只发送飞书通知，不自动改配置文件。")

        auto_adjust = rule.get("auto_adjust")
        if auto_adjust:
            lines.append("")
            lines.append("建议自动调整：")
            lines.append(f"- 参数：{auto_adjust.get('target')}")
            lines.append(f"- 操作：{auto_adjust.get('operation')} {auto_adjust.get('value')}")

    elif level == "L2":
        lines.append("处理方式：L2 建议级，需要你确认后再改参数。")

        suggestions = rule.get("suggestions", [])
        if suggestions:
            lines.append("")
            lines.append("建议方案：")
            for suggestion in suggestions:
                lines.append(
                    f"- 调整 {suggestion.get('target')}："
                    f"{suggestion.get('operation')} {suggestion.get('value')}"
                )
                lines.append(f"  原因：{suggestion.get('reason')}")

        lines.append("")
        lines.append("下一步：如果你同意，回复“确认这个 L2 调整”，再把它接成自动修改配置。")

    elif level == "L3":
        lines.append("处理方式：L3 告警级，建议立刻暂停训练并人工检查。")
        lines.append(f"紧急动作：{rule.get('emergency_action', 'pause_training')}")

    lines.append("")
    lines.append(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--state-path")
    args = parser.parse_args()

    pipeline = load_yaml(args.config)

    if args.test:
        print("✅ Hermes / 飞书通知测试")
        print("")
        print("如果你在飞书里看到这条消息，说明 Hermes 已经可以把训练监控脚本的输出转发到飞书。")
        print(f"项目：{pipeline['project']['name']}")
        print(f"Run ID：{pipeline['project']['run_id']}")
        return

    rules_cfg = load_yaml(pipeline["paths"]["detection_rules"])

    run_dir = os.path.join(ROOT, pipeline["paths"]["run_dir"])
    log_path = os.path.join(run_dir, "train.jsonl")

    state_path = args.state_path or os.path.join(ROOT, "runs", "_hermes_notify_state.json")
    state = load_state(state_path)

    rows = load_jsonl(log_path)

    if not rows:
        if args.debug:
            print("暂无训练日志，不发送飞书通知。")
        return

    messages = []

    for rule in rules_cfg["rules"]:
        metric = rule["metric"]
        points = get_metric_points(rows, metric)

        triggered, reason, latest = check_rule(rule, points)

        if not triggered:
            continue

        latest_step = latest.get("step") if latest else "unknown"
        notify_key = f"{pipeline['project']['run_id']}::{rule['id']}::{latest_step}"

        if notify_key in state["notified"]:
            continue

        messages.append(format_alert(pipeline, rule, reason, latest))
        state["notified"][notify_key] = {
            "time": time.time(),
            "rule_id": rule["id"],
            "step": latest_step,
            "reason": reason,
        }

    if messages:
        print("\n\n---\n\n".join(messages))
        save_state(state_path, state)
    else:
        if args.debug:
            print("✅ 没有触发 L1/L2/L3，不发送飞书通知。")


if __name__ == "__main__":
    main()
