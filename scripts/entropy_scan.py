import argparse
import json
import os

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    pipeline = load_yaml(args.config)

    checkpoint_dir = os.path.join(ROOT, pipeline["paths"]["checkpoint_dir"])
    report_dir = os.path.join(ROOT, pipeline["paths"]["report_dir"])

    keep_recent = pipeline["checkpoint"]["keep_recent"]

    os.makedirs(report_dir, exist_ok=True)

    ckpts = []
    if os.path.exists(checkpoint_dir):
        for name in os.listdir(checkpoint_dir):
            if name.endswith(".zip"):
                path = os.path.join(checkpoint_dir, name)
                ckpts.append(
                    {
                        "name": name,
                        "path": path,
                        "size_bytes": os.path.getsize(path),
                        "modified_time": os.path.getmtime(path),
                    }
                )

    ckpts.sort(key=lambda x: x["modified_time"], reverse=True)

    keep = ckpts[:keep_recent]
    cleanup = ckpts[keep_recent:]

    report = {
        "checkpoint_dir": checkpoint_dir,
        "keep_recent": keep_recent,
        "keep": keep,
        "cleanup_candidates": cleanup,
        "cleanup_size_bytes": sum(x["size_bytes"] for x in cleanup),
    }

    report_path = os.path.join(report_dir, "entropy_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("🧹 熵管理扫描完成")
    print(f"保留 checkpoint 数量: {len(keep)}")
    print(f"建议清理 checkpoint 数量: {len(cleanup)}")
    print(f"报告路径: {report_path}")


if __name__ == "__main__":
    main()
