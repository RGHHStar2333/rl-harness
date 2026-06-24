import argparse
import hashlib
import json
import os
import statistics
import time
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    if ctype == "trend":
        window = int(condition["window"])
        min_delta = float(condition["min_delta"])
        if len(points) < window:
            return False, "窗口数据不足", latest
        recent = points[-window:]
        delta = recent[-1]["value"] - recent[0]["value"]
        if delta < min_delta:
            return True, f"最近 {window} 个记录增长 {delta:.3f}，低于要求 {min_delta:.3f}", latest

    if ctype == "threshold":
        current = latest["value"]
        op = condition["operator"]
        target = float(condition["value"])
        if op == "<" and current < target:
            return True, f"当前值 {current:.3f} < 阈值 {target:.3f}", latest
        if op == ">" and current > target:
            return True, f"当前值 {current:.3f} > 阈值 {target:.3f}", latest

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


def make_token(run_id, rule_id, step):
    raw = f"{run_id}:{rule_id}:{step}:{time.time()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pipeline = load_yaml(args.config)
    rules_cfg = load_yaml(pipeline["paths"]["detection_rules"])

    run_id = pipeline["project"]["run_id"]
    run_dir = os.path.join(ROOT, pipeline["paths"]["run_dir"])
    log_path = os.path.join(run_dir, "train.jsonl")

    pending_dir = os.path.join(ROOT, "runs", "l2_pending")
    os.makedirs(pending_dir, exist_ok=True)

    state_path = os.path.join(ROOT, "runs", "_l2_proposal_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"proposed": {}}

    rows = load_jsonl(log_path)
    if not rows:
        if args.debug:
            print("L2：暂无训练日志。")
        return

    messages = []

    for rule in rules_cfg["rules"]:
        if rule.get("response_level") != "L2":
            continue

        metric = rule["metric"]
        points = get_metric_points(rows, metric)
        triggered, reason, latest = check_rule(rule, points)

        if not triggered:
            continue

        latest_step = latest.get("step", 0) if latest else 0
        proposal_key = f"{run_id}::{rule['id']}::{latest_step}"

        if proposal_key in state["proposed"]:
            continue

        token = make_token(run_id, rule["id"], latest_step)

        proposal = {
            "token": token,
            "status": "pending",
            "created_at": time.time(),
            "run_id": run_id,
            "project": pipeline["project"],
            "rule": rule,
            "reason": reason,
            "latest": latest,
            "hyper_config": pipeline["paths"]["hyper_config"],
            "l2_max_change_ratio": pipeline["feedback_flywheel"].get("l2_max_change_ratio", 0.30),
        }

        proposal_path = os.path.join(pending_dir, f"{token}.json")
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=2)

        state["proposed"][proposal_key] = {
            "token": token,
            "time": time.time(),
            "proposal_path": proposal_path,
        }

        lines = []
        lines.append("🟡 RL Harness L2 调整建议")
        lines.append("")
        lines.append(f"项目：{pipeline['project']['name']}")
        lines.append(f"任务：{pipeline['project']['task']}")
        lines.append(f"Run ID：{run_id}")
        lines.append(f"规则：{rule['id']}")
        lines.append(f"指标：{metric}")
        lines.append(f"最新 step：{latest_step}")
        lines.append(f"最新值：{latest.get('value'):.3f}")
        lines.append(f"触发原因：{reason}")
        lines.append("")
        lines.append("建议调整：")

        for s in rule.get("suggestions", []):
            lines.append(f"- {s.get('target')}: {s.get('operation')} {s.get('value')}")
            lines.append(f"  原因：{s.get('reason')}")

        lines.append("")
        lines.append(f"确认口令 token：{token}")
        lines.append("")
        lines.append("确认执行：")
        lines.append(f"cd {ROOT} && bash scripts/l2_confirm.sh {token}")
        lines.append("")
        lines.append("拒绝执行：")
        lines.append(f"cd {ROOT} && bash scripts/l2_reject.sh {token}")
        lines.append("")
        lines.append("说明：L2 不会自动改参数，必须确认后才执行。")

        messages.append("\n".join(lines))

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    if messages:
        print("\n\n---\n\n".join(messages))
    elif args.debug:
        print("L2：没有触发新的确认建议。")


if __name__ == "__main__":
    main()
