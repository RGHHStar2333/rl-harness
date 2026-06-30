import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import yaml

from feedback_profile import (
    active_training_gate,
    append_jsonl as profile_append_jsonl,
    apply_config_adjustment,
    load_feedback_context,
    load_yaml as profile_load_yaml,
    resolve_path,
)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


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


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_state(path):
    if not os.path.exists(path):
        return {"notified": {}, "l1_auto": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {"notified": {}, "l1_auto": {}}

    state.setdefault("notified", {})
    state.setdefault("l1_auto", {})
    return state


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def nested_get(data, path):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"找不到参数路径: {path}")
        cur = cur[key]
    return cur


def nested_set(data, path, value):
    keys = path.split(".")
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            raise KeyError(f"找不到参数路径: {path}")
        cur = cur[key]
    cur[keys[-1]] = value


def apply_operation(old_value, operation, raw_value):
    old_value = float(old_value)
    raw_value = float(raw_value)

    if operation == "multiply":
        return old_value * raw_value
    if operation == "add":
        return old_value + raw_value
    if operation == "set":
        return raw_value

    raise ValueError(f"不支持的操作: {operation}")


def clamp_l1_change(old_value, proposed_value, max_ratio):
    old_value = float(old_value)
    proposed_value = float(proposed_value)
    max_delta = abs(old_value) * float(max_ratio)

    if max_delta == 0:
        max_delta = float(max_ratio)

    lower = old_value - max_delta
    upper = old_value + max_delta

    return max(lower, min(upper, proposed_value))


def run_lint():
    result = subprocess.run(
        [sys.executable, "scripts/ops/lint_config.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def get_l1_auto_adjustments(rule):
    if rule.get("auto_adjustments"):
        return rule["auto_adjustments"]

    if rule.get("auto_adjust"):
        return [rule["auto_adjust"]]

    return []


def apply_l1_auto_adjust(context, rule, reason, latest, dry_run=False):
    auto_adjustments = get_l1_auto_adjustments(rule)

    if not auto_adjustments:
        return False, "L1 规则没有 auto_adjust 或 auto_adjustments 字段，无法自动修改。"

    changes = []

    for auto_adjust in auto_adjustments:
        target = auto_adjust["target"]
        operation = auto_adjust["operation"]
        raw_value = auto_adjust["value"]

        try:
            change = apply_config_adjustment(
                context=context,
                target=target,
                operation=operation,
                raw_value=raw_value,
                max_ratio=context.l1_max_change_ratio,
                backup_label="l1",
                dry_run=dry_run,
            )
        except Exception as exc:
            return False, str(exc)

        change["reason_detail"] = auto_adjust.get("reason")
        changes.append(change)

    summary = []
    summary.append("L1 自动修改可调配置")
    summary.append(f"- 规则：{rule['id']}")
    summary.append(f"- Profile：{context.profile_name}")
    summary.append(f"- L1 最大改动比例：{context.l1_max_change_ratio}")
    summary.append(f"- 触发原因：{reason}")
    for change in changes:
        summary.append(
            f"- {change['target']}: {change['old_value']} -> {change['new_value']} "
            f"({change['operation']} {change['operation_value']})"
        )

    if dry_run:
        summary.append("- 模式：dry-run，没有真正写入文件")
        return True, "\n".join(summary)

    for change in changes:
        profile_append_jsonl(context.adjustments_path, {
            "timestamp": time.time(),
            "run_id": context.run_id,
            "profile_name": context.profile_name,
            "trainer_kind": context.trainer_kind,
            "level": "L1",
            "rule_id": rule["id"],
            "target": change["target"],
            "old_value": change["old_value"],
            "new_value": change["new_value"],
            "operation": change["operation"],
            "operation_value": change["operation_value"],
            "reason": reason,
            "reason_detail": change.get("reason_detail"),
            "latest_step": latest.get("step") if latest else None,
            "latest_value": latest.get("value") if latest else None,
            "config_path": context.adjust_config_path,
            "hyper_path": context.adjust_config_path,
            "backup_path": change["backup_path"],
            "restart_required": context.restart_required,
        })

    summary.append("")
    summary.append("✅ 修改成功，lint 校验通过。")
    summary.append("备份文件：")
    for change in changes:
        summary.append(f"- {change['backup_path']}")
    summary.append(f"调整日志：{resolve_path(context.adjustments_path)}")
    if context.restart_required:
        summary.append("说明：该修改需要重启训练或下次启动后生效。")
    else:
        summary.append("说明：当前只修改配置，不自动重启训练。")

    return True, "\n".join(summary)


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


def format_alert(context, rule, reason, latest, adjustment_summary=None):
    project = context.project
    level = rule["response_level"]

    lines = []
    lines.append("🚨 RL Harness 训练反馈")
    lines.append("")
    lines.append(f"项目：{project['name']}")
    lines.append(f"任务：{project['task']}")
    lines.append(f"Run ID：{context.run_id}")
    lines.append(f"Profile：{context.profile_name}")
    lines.append(f"规则：{rule['id']}")
    lines.append(f"级别：{level}")
    lines.append(f"指标：{rule.get('metric')}")

    if latest:
        lines.append(f"最新 step：{latest.get('step')}")
        lines.append(f"最新值：{latest.get('value'):.3f}")

    lines.append(f"触发原因：{reason}")
    lines.append("")

    if level == "L1":
        lines.append("处理方式：L1 自动级。")
        lines.append("系统已尝试自动修改可调配置。")
        if adjustment_summary:
            lines.append("")
            lines.append(adjustment_summary)

    elif level == "L2":
        lines.append("处理方式：L2 建议级，需要你确认后再改参数。")
        for s in rule.get("suggestions", []):
            lines.append(f"- 建议调整 {s.get('target')}：{s.get('operation')} {s.get('value')}")
            lines.append(f"  原因：{s.get('reason')}")

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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-l1-test", action="store_true")
    parser.add_argument("--allow-inactive", action="store_true")
    parser.add_argument("--state-path")

    args = parser.parse_args()

    context = load_feedback_context(args.config)
    rules_cfg = profile_load_yaml(context.rules_path)

    if args.test:
        print("✅ Hermes / 飞书通知测试")
        print("")
        print("如果你在飞书里看到这条消息，说明 Hermes 可以转发训练监控脚本输出。")
        print(f"项目：{context.project['name']}")
        print(f"Run ID：{context.run_id}")
        print(f"Profile：{context.profile_name}")
        return

    gate_ok, gate_reason = active_training_gate(context, allow_inactive=args.allow_inactive)
    if not gate_ok:
        if args.debug:
            print(f"{context.profile_name}：{gate_reason}")
        return

    if args.force_l1_test:
        l1_rules = [r for r in rules_cfg["rules"] if r.get("response_level") == "L1"]

        if not l1_rules:
            print("❌ detection_rules.yaml 里没有 L1 规则。")
            return

        rule = l1_rules[0]
        latest = {"step": 0, "value": 0.0}
        reason = "force-l1-test 手动测试触发"

        ok, adjustment_summary = apply_l1_auto_adjust(
            context=context,
            rule=rule,
            reason=reason,
            latest=latest,
            dry_run=args.dry_run,
        )

        print(format_alert(context, rule, reason, latest, adjustment_summary))
        return

    log_path = resolve_path(context.metric_log_path)
    default_state_path = (
        "runs/_hermes_notify_state.json"
        if context.trainer_kind == "sb3"
        else f"{context.state_path_prefix}_notify_state.json"
    )
    state_path = resolve_path(args.state_path or default_state_path)

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

        latest_step = int(latest.get("step", 0)) if latest else 0
        notify_key = f"{context.run_id}::{rule['id']}::{latest_step}"

        if notify_key in state["notified"]:
            continue

        level = rule["response_level"]
        adjustment_summary = None

        if level == "L1":
            auto_key = f"{context.run_id}::{rule['id']}"
            cooldown_steps = int(rule.get("cooldown_steps", 20000))
            last_auto = state["l1_auto"].get(auto_key)

            if last_auto:
                last_step = int(last_auto.get("last_step", 0))
                if latest_step - last_step < cooldown_steps:
                    if args.debug:
                        print(f"⏳ L1 规则 {rule['id']} 在冷却中，不重复自动修改。")
                    continue

            ok, adjustment_summary = apply_l1_auto_adjust(
                context=context,
                rule=rule,
                reason=reason,
                latest=latest,
                dry_run=args.dry_run,
            )

            state["l1_auto"][auto_key] = {
                "last_step": latest_step,
                "last_time": time.time(),
                "ok": ok,
                "dry_run": args.dry_run,
            }

        messages.append(format_alert(context, rule, reason, latest, adjustment_summary))

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
