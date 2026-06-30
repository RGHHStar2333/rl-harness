#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"


def load_queue_module():
    spec = importlib.util.spec_from_file_location("hermes_queue_entrypoint", QUEUE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Handle one Hermes/Feishu training request message.")
    parser.add_argument("--text", help="Hermes/Feishu message body.")
    parser.add_argument("--file", help="Read message body from file.")
    parser.add_argument("--source", default="feishu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise SystemExit("请提供 --text 或 --file。")

    queue = load_queue_module()
    jobs = queue.enqueue_text(text, source=args.source, dry_run=args.dry_run)

    if args.json:
        print(json.dumps({"accepted": len(jobs), "jobs": jobs}, ensure_ascii=False, indent=2))
        return

    if not jobs:
        print("没有识别到可入队的训练任务。")
        print("示例：G1 4096并行 8000次 1小时")
        return

    mode = "dry-run，未真正入队" if args.dry_run else "已入队"
    print(f"Hermes 已解析 {len(jobs)} 个训练任务（{mode}）。")
    for index, job in enumerate(jobs, start=1):
        print(
            f"{index}. {job['run_id']} | envs={job['num_envs']} "
            f"| iterations={job['max_iterations']} "
            f"| runtime={job.get('max_runtime_minutes') or 'none'}min"
        )
    print("")
    print("队列会在下一轮 monitor tick 自动启动；也可以手动运行：")
    print("python3 scripts/training_queue/hermes_queue.py tick")


if __name__ == "__main__":
    main()
