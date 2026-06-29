import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import yaml

from feedback_profile import (
    append_jsonl as profile_append_jsonl,
    context_from_proposal,
    resolve_path,
    target_allowed,
)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def nested_get(data, path):
    cur = data
    for key in path.split("."):
        if key not in cur:
            raise KeyError(f"找不到参数路径: {path}")
        cur = cur[key]
    return cur


def nested_set(data, path, value):
    keys = path.split(".")
    cur = data
    for key in keys[:-1]:
        if key not in cur:
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


def clamp_change(old_value, proposed_value, max_ratio):
    old_value = float(old_value)
    proposed_value = float(proposed_value)
    max_delta = abs(old_value) * float(max_ratio)

    if max_delta == 0:
        max_delta = float(max_ratio)

    lower = old_value - max_delta
    upper = old_value + max_delta

    return max(lower, min(upper, proposed_value))


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_lint():
    result = subprocess.run(
        [sys.executable, "scripts/ops/lint_config.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--decision", required=True, choices=["confirm", "reject"])
    parser.add_argument("--pending-dir", default="runs/l2_pending")
    args = parser.parse_args()

    proposal_path = os.path.join(resolve_path(args.pending_dir), f"{args.token}.json")

    if not os.path.exists(proposal_path):
        print(f"❌ 找不到 L2 提案 token: {args.token}")
        return

    with open(proposal_path, "r", encoding="utf-8") as f:
        proposal = json.load(f)

    context = context_from_proposal(proposal)

    if proposal.get("status") != "pending":
        print(f"⚠️ 这个 L2 提案已经处理过了，当前状态：{proposal.get('status')}")
        return

    if args.decision == "reject":
        proposal["status"] = "rejected"
        proposal["decided_at"] = time.time()

        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=2)

        profile_append_jsonl(context.adjustments_path, {
            "timestamp": time.time(),
            "run_id": proposal["run_id"],
            "profile_name": context.profile_name,
            "trainer_kind": context.trainer_kind,
            "level": "L2",
            "decision": "rejected",
            "token": args.token,
            "rule_id": proposal["rule"]["id"],
            "reason": proposal["reason"],
            "restart_required": context.restart_required,
        })

        print("✅ 已拒绝 L2 调整。不会修改配置。")
        return

    config_path = context.adjust_config_path
    config_full = resolve_path(config_path)
    config_cfg = load_yaml(config_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{config_full}.bak_l2_{timestamp}"
    shutil.copyfile(config_full, backup_path)

    changes = []
    max_ratio = float(context.l2_max_change_ratio)

    try:
        for s in proposal["rule"].get("suggestions", []):
            target = s["target"]

            if not target_allowed(context, target):
                allowed = ", ".join(context.allowed_target_prefixes)
                raise ValueError(f"安全限制：{context.profile_name} 只允许修改 {allowed}，不允许修改 {target}")

            old_value = nested_get(config_cfg, target)

            if not isinstance(old_value, (int, float)):
                raise ValueError(f"目标参数不是数字：{target}={old_value}")

            proposed_value = apply_operation(old_value, s["operation"], s["value"])
            new_value = clamp_change(old_value, proposed_value, max_ratio)

            if target.endswith("learning_rate") and new_value <= 0:
                raise ValueError("learning_rate 调整后小于等于 0，已阻止。")

            nested_set(config_cfg, target, new_value)

            changes.append({
                "target": target,
                "old_value": old_value,
                "new_value": new_value,
                "operation": s["operation"],
                "operation_value": s["value"],
                "reason": s.get("reason"),
            })

        save_yaml(config_path, config_cfg)

        lint_ok, lint_output = run_lint()

        if not lint_ok:
            shutil.copyfile(backup_path, config_full)
            print("❌ L2 修改后 lint 失败，已经回滚。")
            print(lint_output)
            return

    except Exception as e:
        shutil.copyfile(backup_path, config_full)
        print("❌ L2 执行失败，已经回滚。")
        print(str(e))
        return

    proposal["status"] = "confirmed"
    proposal["decided_at"] = time.time()
    proposal["changes"] = changes
    proposal["backup_path"] = backup_path
    proposal["restart_required"] = context.restart_required

    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)

    for c in changes:
        profile_append_jsonl(context.adjustments_path, {
            "timestamp": time.time(),
            "run_id": proposal["run_id"],
            "profile_name": context.profile_name,
            "trainer_kind": context.trainer_kind,
            "level": "L2",
            "decision": "confirmed",
            "token": args.token,
            "rule_id": proposal["rule"]["id"],
            "target": c["target"],
            "old_value": c["old_value"],
            "new_value": c["new_value"],
            "operation": c["operation"],
            "operation_value": c["operation_value"],
            "reason": proposal["reason"],
            "latest_step": proposal["latest"].get("step"),
            "latest_value": proposal["latest"].get("value"),
            "config_path": config_path,
            "hyper_path": config_path,
            "backup_path": backup_path,
            "restart_required": context.restart_required,
        })

    print("✅ L2 调整已确认并执行。")
    print(f"token: {args.token}")
    print(f"备份文件: {backup_path}")
    print("变更内容：")
    for c in changes:
        print(f"- {c['target']}: {c['old_value']} -> {c['new_value']}")
    print("✅ lint 校验通过。")
    if context.restart_required:
        print("说明：该配置变更需要重启训练或下次启动后生效。")


if __name__ == "__main__":
    main()
