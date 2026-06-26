import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import yaml


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
    args = parser.parse_args()

    proposal_path = os.path.join(ROOT, "runs", "l2_pending", f"{args.token}.json")

    if not os.path.exists(proposal_path):
        print(f"❌ 找不到 L2 提案 token: {args.token}")
        return

    with open(proposal_path, "r", encoding="utf-8") as f:
        proposal = json.load(f)

    if proposal.get("status") != "pending":
        print(f"⚠️ 这个 L2 提案已经处理过了，当前状态：{proposal.get('status')}")
        return

    if args.decision == "reject":
        proposal["status"] = "rejected"
        proposal["decided_at"] = time.time()

        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=2)

        append_jsonl(os.path.join(ROOT, "runs", "adjustments.jsonl"), {
            "timestamp": time.time(),
            "run_id": proposal["run_id"],
            "level": "L2",
            "decision": "rejected",
            "token": args.token,
            "rule_id": proposal["rule"]["id"],
            "reason": proposal["reason"],
        })

        print("✅ 已拒绝 L2 调整。不会修改 hyper.yaml。")
        return

    hyper_path = proposal["hyper_config"]
    hyper_full = os.path.join(ROOT, hyper_path)
    hyper_cfg = load_yaml(hyper_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{hyper_full}.bak_l2_{timestamp}"
    shutil.copyfile(hyper_full, backup_path)

    changes = []
    max_ratio = float(proposal.get("l2_max_change_ratio", 0.30))

    try:
        for s in proposal["rule"].get("suggestions", []):
            target = s["target"]

            if not target.startswith("ppo."):
                raise ValueError(f"安全限制：L2 目前只允许修改 ppo.*，不允许修改 {target}")

            old_value = nested_get(hyper_cfg, target)

            if not isinstance(old_value, (int, float)):
                raise ValueError(f"目标参数不是数字：{target}={old_value}")

            proposed_value = apply_operation(old_value, s["operation"], s["value"])
            new_value = clamp_change(old_value, proposed_value, max_ratio)

            if target.endswith("learning_rate") and new_value <= 0:
                raise ValueError("learning_rate 调整后小于等于 0，已阻止。")

            nested_set(hyper_cfg, target, new_value)

            changes.append({
                "target": target,
                "old_value": old_value,
                "new_value": new_value,
                "operation": s["operation"],
                "operation_value": s["value"],
                "reason": s.get("reason"),
            })

        save_yaml(hyper_path, hyper_cfg)

        lint_ok, lint_output = run_lint()

        if not lint_ok:
            shutil.copyfile(backup_path, hyper_full)
            print("❌ L2 修改后 lint 失败，已经回滚。")
            print(lint_output)
            return

    except Exception as e:
        shutil.copyfile(backup_path, hyper_full)
        print("❌ L2 执行失败，已经回滚。")
        print(str(e))
        return

    proposal["status"] = "confirmed"
    proposal["decided_at"] = time.time()
    proposal["changes"] = changes
    proposal["backup_path"] = backup_path

    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)

    for c in changes:
        append_jsonl(os.path.join(ROOT, "runs", "adjustments.jsonl"), {
            "timestamp": time.time(),
            "run_id": proposal["run_id"],
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
            "hyper_path": hyper_path,
            "backup_path": backup_path,
        })

    print("✅ L2 调整已确认并执行。")
    print(f"token: {args.token}")
    print(f"备份文件: {backup_path}")
    print("变更内容：")
    for c in changes:
        print(f"- {c['target']}: {c['old_value']} -> {c['new_value']}")
    print("✅ lint 校验通过。")


if __name__ == "__main__":
    main()
