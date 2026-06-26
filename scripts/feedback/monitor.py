import argparse
import json
import os
import statistics

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
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


def get_metric_values(rows, metric):
    return [r[metric] for r in rows if metric in r and r[metric] is not None]


def check_rule(rule, values):
    condition = rule["condition"]
    ctype = condition["type"]

    if not values:
        return False, "没有足够数据"

    if ctype == "threshold":
        current = values[-1]
        op = condition["operator"]
        target = condition["value"]

        if op == "<" and current < target:
            return True, f"当前值 {current} < 阈值 {target}"
        if op == ">" and current > target:
            return True, f"当前值 {current} > 阈值 {target}"

    if ctype == "trend":
        window = condition["window"]
        min_delta = condition["min_delta"]

        if len(values) < window:
            return False, "窗口数据不足"

        recent = values[-window:]
        delta = recent[-1] - recent[0]

        if delta < min_delta:
            return True, f"最近 {window} 个窗口增长 {delta}，低于要求 {min_delta}"

    if ctype == "volatility":
        window = condition["window"]
        max_std = condition["max_std"]

        if len(values) < window:
            return False, "窗口数据不足"

        recent = values[-window:]
        std = statistics.pstdev(recent)

        if std > max_std:
            return True, f"最近 {window} 个窗口波动 {std}，超过 {max_std}"

    return False, "未触发"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    pipeline = load_yaml(args.config)
    rules_cfg = load_yaml(pipeline["paths"]["detection_rules"])

    run_dir = os.path.join(ROOT, pipeline["paths"]["run_dir"])
    log_path = os.path.join(run_dir, "train.jsonl")

    rows = load_jsonl(log_path)

    print("🔍 开始反馈飞轮检测")

    for rule in rules_cfg["rules"]:
        metric = rule["metric"]
        values = get_metric_values(rows, metric)

        triggered, reason = check_rule(rule, values)

        if not triggered:
            print(f"✅ {rule['id']} 未触发：{reason}")
            continue

        level = rule["response_level"]

        print("\n==============================")
        print(f"⚠️ 规则触发: {rule['id']}")
        print(f"描述: {rule['description']}")
        print(f"指标: {metric}")
        print(f"原因: {reason}")
        print(f"响应层级: {level}")

        if level == "L1":
            print("动作: 自动微调参数，并通知用户。")
            print("当前 MVP 版本先只打印，不自动改文件。")

        elif level == "L2":
            print("动作: 生成调整建议，等待用户确认。")
            for s in rule.get("suggestions", []):
                print(f"- 建议调整 {s['target']}：{s['operation']} {s['value']}")
                print(f"  原因：{s['reason']}")

        elif level == "L3":
            print("动作: 紧急告警，建议暂停训练。")
            print(f"紧急动作: {rule.get('emergency_action')}")

        print("==============================\n")


if __name__ == "__main__":
    main()
