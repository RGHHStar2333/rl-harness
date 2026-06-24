import argparse
import os
import re
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    pipeline = load_yaml(args.config)
    checkpoint_dir = os.path.join(ROOT, pipeline["paths"]["checkpoint_dir"])

    if not os.path.exists(checkpoint_dir):
        raise SystemExit(f"❌ checkpoint 目录不存在: {checkpoint_dir}")

    candidates = []

    for name in os.listdir(checkpoint_dir):
        match = re.match(r"model_step_(\d+)\.zip$", name)
        if match:
            step = int(match.group(1))
            candidates.append((step, os.path.join(checkpoint_dir, name)))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        print(candidates[0][1])
        return

    final_model = os.path.join(checkpoint_dir, "final_model.zip")
    if os.path.exists(final_model):
        print(final_model)
        return

    raise SystemExit(f"❌ 没有找到可恢复的 checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
