import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def exists(path):
    return os.path.exists(os.path.join(ROOT, path))


def fail(message, fix):
    print("❌ 校验失败")
    print("问题:", message)
    print("修复:", fix)
    sys.exit(1)


def nested_get(data, path):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(path)
        cur = cur[key]
    return cur


def validate_rule_targets(rules_cfg, allowed_prefixes, rules_path):
    allowed_levels = {"L1", "L2", "L3"}

    for rule in rules_cfg.get("rules", []):
        level = rule.get("response_level")
        if level not in allowed_levels:
            fail(
                f"{rules_path} 规则 {rule.get('id')} 的 response_level 无效: {level}",
                "请把 response_level 改成 L1、L2 或 L3。",
            )

        targets = []
        if rule.get("auto_adjust"):
            targets.append(rule["auto_adjust"].get("target"))

        for suggestion in rule.get("suggestions", []):
            targets.append(suggestion.get("target"))

        for target in [t for t in targets if t]:
            if not any(target.startswith(prefix) for prefix in allowed_prefixes):
                fail(
                    f"{rules_path} 规则 {rule.get('id')} target 不在安全白名单内: {target}",
                    f"允许的 target 前缀只有: {', '.join(allowed_prefixes)}。",
                )


def validate_mjlab_feedback_profile():
    profile_path = "configs/tasks/mjlab/feedback.yaml"

    if not exists(profile_path):
        return

    cfg = load_yaml(profile_path)
    profile = cfg.get("feedback_profile")

    if not profile:
        fail(
            "configs/tasks/mjlab/feedback.yaml 缺少 feedback_profile",
            "请在文件顶层添加 feedback_profile 配置块。",
        )

    for key in ["name", "trainer_kind", "project", "paths", "safety", "restart"]:
        if key not in profile:
            fail(
                f"MJLab feedback profile 缺少 {key}",
                f"请在 {profile_path} 添加 {key} 配置。",
            )

    if profile["trainer_kind"] != "mjlab":
        fail(
            f"MJLab feedback profile trainer_kind 必须是 mjlab，当前是 {profile['trainer_kind']}",
            "请把 trainer_kind 改成 mjlab。",
        )

    paths = profile["paths"]
    for key in ["metric_log", "rules_config", "adjustable_config"]:
        if key not in paths:
            fail(
                f"MJLab feedback profile paths 缺少 {key}",
                f"请在 {profile_path} 的 paths 下添加 {key}。",
            )

    metric_log = paths["metric_log"]
    if not metric_log.startswith("runs/") or not metric_log.endswith("/train.jsonl"):
        fail(
            f"MJLab metric_log 路径形状不安全: {metric_log}",
            "请使用类似 runs/mjlab_g1_4096_10000/train.jsonl 的路径。",
        )

    for path in [paths["rules_config"], paths["adjustable_config"]]:
        if not exists(path):
            fail(
                f"MJLab feedback profile 引用的文件不存在: {path}",
                "请创建该文件，或者修正 configs/tasks/mjlab/feedback.yaml 中的路径。",
            )

    allowed_prefixes = profile["safety"].get("allowed_target_prefixes", [])
    if not allowed_prefixes:
        fail(
            "MJLab feedback profile 缺少 safety.allowed_target_prefixes",
            "请至少添加 mjlab.agent.algorithm. 作为安全 target 前缀。",
        )

    rules_cfg = load_yaml(paths["rules_config"])
    validate_rule_targets(rules_cfg, allowed_prefixes, paths["rules_config"])

    mjlab_cfg = load_yaml(paths["adjustable_config"])
    try:
        learning_rate = nested_get(mjlab_cfg, "mjlab.agent.algorithm.learning_rate")
    except KeyError:
        fail(
            "MJLab 可调配置缺少 mjlab.agent.algorithm.learning_rate",
            "请在 configs/tasks/mjlab/go1.yaml 添加 Harness 持有的 learning_rate。",
        )

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        fail(
            f"MJLab learning_rate 必须是正数，当前是 {learning_rate}",
            "请把 configs/tasks/mjlab/go1.yaml 里的 mjlab.agent.algorithm.learning_rate 改成正数。",
        )


def main():
    pipeline = load_yaml("configs/pipeline.yaml")

    required_top_keys = [
        "project",
        "environment",
        "paths",
        "checkpoint",
        "feedback_flywheel",
        "mechanical_enforcement",
    ]

    for key in required_top_keys:
        if key not in pipeline:
            fail(
                f"pipeline.yaml 缺少 {key}",
                f"请在 configs/pipeline.yaml 添加 {key} 配置块。",
            )

    hyper_path = pipeline["paths"]["hyper_config"]
    reward_path = pipeline["paths"]["reward_config"]
    rules_path = pipeline["paths"]["detection_rules"]

    for path in [hyper_path, reward_path, rules_path]:
        if not os.path.exists(os.path.join(ROOT, path)):
            fail(
                f"引用的配置文件不存在: {path}",
                "请创建该文件，或者修改 pipeline.yaml 中的路径。",
            )

    save_interval = pipeline["checkpoint"]["save_interval"]
    if save_interval <= 0:
        fail(
            "checkpoint.save_interval 必须大于 0",
            "把 configs/pipeline.yaml 里的 checkpoint.save_interval 改成正整数，例如 5000。",
        )

    reward_cfg = load_yaml(reward_path)
    weights = reward_cfg.get("reward_weights", {})
    total_weight = sum(float(v) for v in weights.values())

    min_w = reward_cfg["constraints"]["total_weight_min"]
    max_w = reward_cfg["constraints"]["total_weight_max"]

    if not (min_w <= total_weight <= max_w):
        fail(
            f"reward 权重总和 {total_weight} 不在范围 [{min_w}, {max_w}] 内",
            "请调整 configs/tasks/cartpole/reward.yaml 里的 reward_weights。",
        )

    validate_mjlab_feedback_profile()

    print("✅ 机械化校验通过，可以启动训练。")


if __name__ == "__main__":
    main()
