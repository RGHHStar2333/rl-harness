import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class FeedbackContext:
    config_path: str
    profile_name: str
    trainer_kind: str
    project: dict[str, Any]
    run_id: str
    task: str
    metric_log_path: str
    rules_path: str
    adjust_config_path: str
    adjustments_path: str
    l2_pending_dir: str
    state_path_prefix: str
    allowed_target_prefixes: list[str]
    l1_max_change_ratio: float
    l2_max_change_ratio: float
    restart_required: bool
    require_active_training: bool
    enabled: bool


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def display_path(path: str) -> str:
    full_path = resolve_path(path)
    try:
        rel = os.path.relpath(full_path, ROOT)
    except ValueError:
        return full_path
    return rel if not rel.startswith("..") else full_path


def load_yaml(path: str) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: Any) -> None:
    with open(resolve_path(path), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_json(path: str) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    full_path = resolve_path(path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def nested_get(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"找不到参数路径: {path}")
        cur = cur[key]
    return cur


def nested_set(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Any = data
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"找不到参数路径: {path}")
        cur = cur[key]
    cur[keys[-1]] = value


def apply_operation(old_value: float, operation: str, raw_value: float) -> float:
    old_value = float(old_value)
    raw_value = float(raw_value)

    if operation == "multiply":
        return old_value * raw_value
    if operation == "add":
        return old_value + raw_value
    if operation == "set":
        return raw_value

    raise ValueError(f"不支持的操作: {operation}")


def clamp_change(old_value: float, proposed_value: float, max_ratio: float) -> float:
    old_value = float(old_value)
    proposed_value = float(proposed_value)
    max_delta = abs(old_value) * float(max_ratio)

    if max_delta == 0:
        max_delta = float(max_ratio)

    lower = old_value - max_delta
    upper = old_value + max_delta
    return max(lower, min(upper, proposed_value))


def target_allowed(context: FeedbackContext, target: str) -> bool:
    return any(target.startswith(prefix) for prefix in context.allowed_target_prefixes)


def run_lint() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "scripts/ops/lint_config.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def load_feedback_context(config_path: str) -> FeedbackContext:
    cfg = load_yaml(config_path)

    if "feedback_profile" in cfg:
        profile = cfg["feedback_profile"]
        project = profile["project"]
        paths = profile["paths"]
        safety = profile.get("safety", {})
        restart = profile.get("restart", {})

        profile_name = profile.get("name", project["run_id"])
        trainer_kind = profile.get("trainer_kind", "mjlab")

        return FeedbackContext(
            config_path=display_path(config_path),
            profile_name=profile_name,
            trainer_kind=trainer_kind,
            project=project,
            run_id=project["run_id"],
            task=project["task"],
            metric_log_path=paths["metric_log"],
            rules_path=paths["rules_config"],
            adjust_config_path=paths["adjustable_config"],
            adjustments_path=paths.get("adjustments_log", "runs/adjustments.jsonl"),
            l2_pending_dir=paths.get("l2_pending_dir", "runs/l2_pending"),
            state_path_prefix=paths.get("state_path_prefix", f"runs/_{profile_name}"),
            allowed_target_prefixes=safety.get("allowed_target_prefixes", []),
            l1_max_change_ratio=float(safety.get("l1_max_change_ratio", 0.10)),
            l2_max_change_ratio=float(safety.get("l2_max_change_ratio", 0.30)),
            restart_required=bool(restart.get("required_after_adjustment", trainer_kind == "mjlab")),
            require_active_training=bool(profile.get("require_active_training", False)),
            enabled=bool(profile.get("enabled", True)),
        )

    project = cfg["project"]
    paths = cfg["paths"]
    feedback = cfg.get("feedback_flywheel", {})

    return FeedbackContext(
        config_path=display_path(config_path),
        profile_name=project["run_id"],
        trainer_kind="sb3",
        project=project,
        run_id=project["run_id"],
        task=project["task"],
        metric_log_path=os.path.join(paths["run_dir"], "train.jsonl"),
        rules_path=paths["detection_rules"],
        adjust_config_path=paths["hyper_config"],
        adjustments_path=paths.get("adjustments_log", "runs/adjustments.jsonl"),
        l2_pending_dir=paths.get("l2_pending_dir", "runs/l2_pending"),
        state_path_prefix="runs/_halfcheetah_feedback",
        allowed_target_prefixes=["ppo."],
        l1_max_change_ratio=float(feedback.get("l1_max_change_ratio", 0.10)),
        l2_max_change_ratio=float(feedback.get("l2_max_change_ratio", 0.30)),
        restart_required=False,
        require_active_training=False,
        enabled=bool(feedback.get("enabled", True)),
    )


def context_public_payload(context: FeedbackContext) -> dict[str, Any]:
    return {
        "feedback_config": context.config_path,
        "profile_name": context.profile_name,
        "trainer_kind": context.trainer_kind,
        "adjust_config": context.adjust_config_path,
        "adjustments_path": context.adjustments_path,
        "l2_pending_dir": context.l2_pending_dir,
        "allowed_target_prefixes": context.allowed_target_prefixes,
        "restart_required": context.restart_required,
    }


def context_from_proposal(proposal: dict[str, Any]) -> FeedbackContext:
    project = proposal.get("project", {})
    run_id = proposal["run_id"]
    trainer_kind = proposal.get("trainer_kind", "sb3")
    adjust_config = proposal.get("adjust_config") or proposal.get("hyper_config")

    return FeedbackContext(
        config_path=proposal.get("feedback_config", ""),
        profile_name=proposal.get("profile_name", run_id),
        trainer_kind=trainer_kind,
        project=project,
        run_id=run_id,
        task=project.get("task", proposal.get("task", "")),
        metric_log_path=proposal.get("metric_log_path", ""),
        rules_path=proposal.get("rules_path", ""),
        adjust_config_path=adjust_config,
        adjustments_path=proposal.get("adjustments_path", "runs/adjustments.jsonl"),
        l2_pending_dir=proposal.get("l2_pending_dir", "runs/l2_pending"),
        state_path_prefix=proposal.get("state_path_prefix", "runs/_feedback"),
        allowed_target_prefixes=proposal.get("allowed_target_prefixes") or ["ppo."],
        l1_max_change_ratio=float(proposal.get("l1_max_change_ratio", 0.10)),
        l2_max_change_ratio=float(proposal.get("l2_max_change_ratio", 0.30)),
        restart_required=bool(proposal.get("restart_required", False)),
        require_active_training=False,
        enabled=True,
    )


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def active_training_gate(context: FeedbackContext, allow_inactive: bool = False) -> tuple[bool, str]:
    if not context.enabled:
        return False, f"{context.profile_name} feedback profile 已禁用。"

    if allow_inactive or not context.require_active_training:
        return True, "active training gate skipped"

    state_path = os.path.join(ROOT, "runs", "active_training.json")
    if not os.path.exists(state_path):
        return False, "没有 runs/active_training.json，跳过需要活跃训练的反馈。"

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        return False, f"读取 active_training.json 失败：{exc}"

    if state.get("kind") != context.trainer_kind:
        return False, f"active training kind={state.get('kind')}，不是 {context.trainer_kind}。"

    if state.get("run_id") != context.run_id:
        return False, f"active run_id={state.get('run_id')}，不是 {context.run_id}。"

    if state.get("status") != "running":
        return False, f"active training status={state.get('status')}，不是 running。"

    pid = state.get("pid")
    if not pid or not pid_alive(pid):
        return False, f"active training PID={pid} 不在运行。"

    return True, "active training is running"


def apply_config_adjustment(
    context: FeedbackContext,
    target: str,
    operation: str,
    raw_value: float,
    max_ratio: float,
    backup_label: str,
    run_lint_after: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not target_allowed(context, target):
        allowed = ", ".join(context.allowed_target_prefixes)
        raise ValueError(f"安全限制：{context.profile_name} 只允许修改 {allowed}，不允许修改 {target}")

    cfg = load_yaml(context.adjust_config_path)
    old_value = nested_get(cfg, target)

    if not isinstance(old_value, (int, float)):
        raise ValueError(f"目标参数不是数字，无法自动调整: {target}={old_value}")

    proposed_value = apply_operation(float(old_value), operation, float(raw_value))
    new_value = clamp_change(float(old_value), proposed_value, float(max_ratio))

    if target.endswith("learning_rate") and new_value <= 0:
        raise ValueError("learning_rate 调整后小于等于 0，已阻止。")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config_full_path = resolve_path(context.adjust_config_path)
    backup_path = f"{config_full_path}.bak_{backup_label}_{timestamp}"

    change = {
        "target": target,
        "old_value": old_value,
        "new_value": new_value,
        "operation": operation,
        "operation_value": raw_value,
        "adjust_config": context.adjust_config_path,
        "backup_path": backup_path,
    }

    if dry_run:
        change["dry_run"] = True
        return change

    shutil.copyfile(config_full_path, backup_path)
    nested_set(cfg, target, new_value)
    save_yaml(context.adjust_config_path, cfg)

    if run_lint_after:
        lint_ok, lint_output = run_lint()
        if not lint_ok:
            shutil.copyfile(backup_path, config_full_path)
            raise RuntimeError(f"修改后 lint 失败，已回滚。\n备份文件：{backup_path}\nlint 输出：\n{lint_output}")

    return change
