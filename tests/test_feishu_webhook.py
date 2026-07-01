import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"
WEBHOOK_PATH = ROOT / "scripts" / "hermes_feishu_webhook.py"


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(f"{name}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_root(tmp_root):
    root = Path(tmp_root)
    write_yaml(root / "configs" / "tasks" / "mjlab" / "go1.yaml", {
        "mjlab": {
            "run_id": "mjlab_g1_base",
            "project_dir": "/tmp/mjlab",
            "task": "Mjlab-Velocity-Flat-Unitree-G1",
            "num_envs": 4096,
            "max_iterations": 5000,
            "wandb_project": "rl-harness-mjlab",
        },
        "agent": {"learning_rate": 0.001},
        "reward_weights": {},
    })
    write_yaml(root / "configs" / "tasks" / "mjlab" / "feedback.yaml", {
        "feedback_profile": {
            "project": {"run_id": "mjlab_g1_base"},
            "paths": {"metric_log": "runs/mjlab_g1_base/train.jsonl"},
        }
    })


def patch_queue(queue, tmp_root):
    queue.ROOT = Path(tmp_root)
    queue.QUEUE_DIR = queue.ROOT / "runs" / "training_queue"
    queue.QUEUE_PATH = queue.QUEUE_DIR / "queue.json"
    queue.EVENTS_PATH = queue.QUEUE_DIR / "events.jsonl"
    queue.ACTIVE_PATH = queue.ROOT / "runs" / "active_training.json"
    queue.MJLAB_CONFIG_PATH = queue.ROOT / "configs" / "tasks" / "mjlab" / "go1.yaml"
    queue.MJLAB_FEEDBACK_PATH = queue.ROOT / "configs" / "tasks" / "mjlab" / "feedback.yaml"


class FeishuWebhookTest(unittest.TestCase):
    def test_challenge_and_token(self):
        webhook = load_module(WEBHOOK_PATH, "webhook")
        response, status = webhook.handle_payload(
            {"challenge": "abc", "token": "secret"},
            expected_token="secret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["challenge"], "abc")

        response, status = webhook.handle_payload(
            {"challenge": "abc", "token": "wrong"},
            expected_token="secret",
        )
        self.assertEqual(status, 403)
        self.assertEqual(response["error"], "invalid_token")

    def test_feishu_message_payload_enqueues_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            queue = load_module(QUEUE_PATH, "queue")
            webhook = load_module(WEBHOOK_PATH, "webhook")
            patch_queue(queue, tmpdir)
            webhook.DELIVERY_LOG_PATH = Path(tmpdir) / "runs" / "hermes_feishu_inbox.jsonl"

            payload = {
                "event": {
                    "message": {
                        "message_type": "text",
                        "content": json.dumps({
                            "text": "1. G1 128并行 10次 1分钟\n2. G1 256并行 20次"
                        }, ensure_ascii=False),
                    }
                }
            }
            response, status = webhook.handle_payload(payload, queue=queue)
            persisted = queue.load_queue()

            self.assertEqual(status, 200)
            self.assertEqual(response["accepted"], 2)
            self.assertEqual(len(persisted["jobs"]), 2)
            self.assertEqual(persisted["jobs"][0]["source"], "feishu_webhook")

    def test_delivery_log_records_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            webhook = load_module(WEBHOOK_PATH, "webhook")
            webhook.DELIVERY_LOG_PATH = Path(tmpdir) / "runs" / "hermes_feishu_inbox.jsonl"

            webhook.append_delivery_log(
                {"text": "G1 128并行 10次"},
                {"ok": True, "accepted": 1, "text": "G1 128并行 10次"},
                200,
            )
            rows = webhook.DELIVERY_LOG_PATH.read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0])

            self.assertEqual(row["status_code"], 200)
            self.assertEqual(row["accepted"], 1)
            self.assertEqual(row["text"], "G1 128并行 10次")

    def test_direct_payload_can_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            queue = load_module(QUEUE_PATH, "queue")
            webhook = load_module(WEBHOOK_PATH, "webhook")
            patch_queue(queue, tmpdir)

            response, status = webhook.handle_payload(
                {"text": "G1 128并行 10次 30分钟"},
                queue=queue,
                dry_run=True,
            )

            self.assertEqual(status, 200)
            self.assertEqual(response["accepted"], 1)
            self.assertFalse(queue.QUEUE_PATH.exists())


if __name__ == "__main__":
    unittest.main()
