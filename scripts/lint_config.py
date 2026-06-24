import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fail(message, fix):
    print("❌ 校验失败")
    print("问题:", message)
    print("修复:", fix)
    sys.exit(1)


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

    print("✅ 机械化校验通过，可以启动训练。")


if __name__ == "__main__":
    main()
