#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"
WEBHOOK_PATH = ROOT / "scripts" / "hermes_feishu_webhook.py"


class FakeProc:
    def __init__(self, pid):
        self.pid = pid


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_temp_harness(tmp_root):
    root = Path(tmp_root)
    write_yaml(root / "configs" / "tasks" / "mjlab" / "go1.yaml", {
        "mjlab": {
            "run_id": "mjlab_g1_base",
            "project_dir": "/tmp/mjlab",
            "task": "Mjlab-Velocity-Flat-Unitree-G1",
            "num_envs": 4096,
            "max_iterations": 5000,
            "wandb_project": "rl-harness-mjlab",
            "wandb_name": "mjlab_g1_base",
        },
        "agent": {"learning_rate": 0.001},
        "reward_weights": {"track_linear_velocity": 2.0},
    })
    write_yaml(root / "configs" / "tasks" / "mjlab" / "feedback.yaml", {
        "feedback_profile": {
            "name": "mjlab_g1_feedback",
            "enabled": True,
            "trainer_kind": "mjlab",
            "project": {
                "name": "rl_harness_mjlab",
                "task": "Mjlab-Velocity-Flat-Unitree-G1",
                "run_id": "mjlab_g1_base",
            },
            "paths": {
                "metric_log": "runs/mjlab_g1_base/train.jsonl",
                "rules_config": "configs/tasks/mjlab/detection_rules.yaml",
                "adjustable_config": "configs/tasks/mjlab/go1.yaml",
                "adjustments_log": "runs/adjustments.jsonl",
                "l2_pending_dir": "runs/l2_pending",
                "state_path_prefix": "runs/_mjlab_g1_feedback",
            },
        },
    })


def patch_queue_paths(queue, tmp_root):
    queue.ROOT = Path(tmp_root)
    queue.QUEUE_DIR = queue.ROOT / "runs" / "training_queue"
    queue.QUEUE_PATH = queue.QUEUE_DIR / "queue.json"
    queue.EVENTS_PATH = queue.QUEUE_DIR / "events.jsonl"
    queue.ACTIVE_PATH = queue.ROOT / "runs" / "active_training.json"
    queue.MJLAB_CONFIG_PATH = queue.ROOT / "configs" / "tasks" / "mjlab" / "go1.yaml"
    queue.MJLAB_FEEDBACK_PATH = queue.ROOT / "configs" / "tasks" / "mjlab" / "feedback.yaml"


def latest_report_path():
    return ROOT / "reports" / "hermes_feishu_e2e_validation_20260701.md"


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_temp_harness(tmpdir)
        queue = load_module(f"hermes_queue_e2e_{time.time_ns()}", QUEUE_PATH)
        webhook = load_module(f"hermes_feishu_webhook_e2e_{time.time_ns()}", WEBHOOK_PATH)
        patch_queue_paths(queue, tmpdir)

        next_pid = {"value": 3000}

        def fake_launch(cmd, project_dir, log_path):
            next_pid["value"] += 1
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("fake mjlab process\n", encoding="utf-8")
            return FakeProc(next_pid["value"])

        queue.launch_process = fake_launch
        queue.pid_alive = lambda pid: True
        queue.pause_active_training = lambda reason: (0, f"paused: {reason}")

        payload = {
            "event": {
                "message": {
                    "message_type": "text",
                    "content": json.dumps({
                        "text": "\n".join([
                            "1. G1 128并行 10次 1分钟",
                            "2. G1 256并行 20次",
                            "Hermes 每3个小时自动切换到下一个训练",
                        ])
                    }, ensure_ascii=False),
                }
            }
        }

        response, status = webhook.handle_payload(payload, queue=queue, source="feishu_webhook")
        if status != 200 or response["accepted"] != 2:
            raise SystemExit(f"webhook enqueue failed: status={status} response={response}")

        started = queue.tick()
        queued = queue.load_queue()
        first = queued["jobs"][0]
        if first["status"] != "running":
            raise SystemExit(f"first job did not start: {started}")

        first["started_at_ts"] = time.time() - 120
        queue.save_queue(queued)

        advanced = queue.tick()
        final_queue = queue.load_queue()
        if final_queue["jobs"][0]["status"] != "completed":
            raise SystemExit(f"first job did not complete: {advanced}")
        if final_queue["jobs"][1]["status"] != "running":
            raise SystemExit(f"second job did not start: {advanced}")

        report = latest_report_path()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "\n".join([
                "# Hermes Feishu E2E Validation - 2026-07-01",
                "",
                "## Result",
                "",
                "PASS",
                "",
                "## Covered Flow",
                "",
                "- Feishu-style JSON message payload",
                "- Training text extraction",
                "- Two MJLab jobs enqueued in order",
                "- First job started by queue tick",
                "- Runtime limit stopped first job",
                "- Second job started automatically",
                "",
                "## Commands",
                "",
                "```bash",
                "python3 scripts/validate_hermes_feishu_e2e.py",
                "```",
                "",
                "## Tick Output",
                "",
                "```text",
                "\n".join(started + advanced),
                "```",
                "",
            ]),
            encoding="utf-8",
        )

        print("PASS Hermes Feishu E2E validation")
        print(f"report={report}")


if __name__ == "__main__":
    main()
