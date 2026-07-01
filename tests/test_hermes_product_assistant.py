import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_PATH = ROOT / "scripts" / "hermes_product_assistant.py"
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
            "wandb_entity": "entity",
            "wandb_name": "mjlab_g1_base",
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


def load_assistant(tmp_root):
    spec = importlib.util.spec_from_file_location(f"hermes_product_assistant_{time.time_ns()}", ASSISTANT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.ROOT = Path(tmp_root)
    module.QUEUE_PATH = QUEUE_PATH
    module.STATE_DIR = module.ROOT / "runs" / "hermes_assistant"
    module.CONFIRM_PATH = module.STATE_DIR / "pending_confirmations.json"
    module.WEBHOOK_STATE_PATH = module.ROOT / "runs" / "hermes_feishu_webhook.json"
    module.INBOX_PATH = module.ROOT / "runs" / "hermes_feishu_inbox.jsonl"
    module.EVENTS_PATH = module.ROOT / "runs" / "training_queue" / "events.jsonl"
    return module


class HermesProductAssistantTest(unittest.TestCase):
    def test_ask_explains_plan_and_creates_confirmation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)

            payload = json.loads(assistant.ask("帮我跑 G1 4096并行 8000次 1小时", json_mode=True))

            self.assertEqual(payload["action"], "plan")
            self.assertEqual(len(payload["jobs"]), 1)
            self.assertTrue(payload["start_after_confirm"])
            self.assertIn("4096 并行训练", "\n".join(payload["risks"]))
            self.assertTrue(assistant.CONFIRM_PATH.exists())

    def test_missing_details_are_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)

            payload = json.loads(assistant.ask("帮我跑个 G1", json_mode=True))

            self.assertEqual(payload["action"], "need_more_info")
            self.assertIn("并行数", payload["missing"])

    def test_confirm_enqueues_without_starting_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)

            plan = json.loads(assistant.ask("帮我跑 G1 128并行 10次 30分钟", json_mode=True))
            result = json.loads(assistant.confirm(plan["token"], start=False, json_mode=True))
            queue = assistant.load_queue_module().load_queue()

            self.assertEqual(result["action"], "confirmed")
            self.assertEqual(len(queue["jobs"]), 1)
            self.assertEqual(queue["jobs"][0]["status"], "queued")
            self.assertEqual(assistant.load_pending()["items"], {})

    def test_status_explains_stale_active_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)
            queue = assistant.load_queue_module()
            queue.enqueue_text("G1 128并行 10次 30分钟")
            write_json(queue.ACTIVE_PATH, {
                "status": "running",
                "kind": "mjlab",
                "run_id": "old_run",
                "pid": 99999999,
            })

            message = assistant.status()

            self.assertIn("队列里共有 1 个任务", message)
            self.assertIn("状态残留", message)

    def test_diagnose_explains_challenge_only_delivery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)
            write_json(assistant.WEBHOOK_STATE_PATH, {
                "status": "running",
                "port": 8765,
            })
            assistant.INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
            assistant.INBOX_PATH.write_text(
                json.dumps({"status_code": 200, "ok": True, "accepted": 0, "payload_keys": ["challenge"]}) + "\n",
                encoding="utf-8",
            )

            message = assistant.diagnose(text="G1 128并行 10次 30分钟")

            self.assertIn("只有 challenge", message)
            self.assertIn("可以解析出 1 个训练任务", message)

    def test_shell_like_request_is_refused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)

            payload = json.loads(assistant.ask("cd /tmp\npython3 evil.py", json_mode=True))

            self.assertEqual(payload["action"], "refuse")
            self.assertIn("不能直接执行", payload["message"])

    def test_ask_status_is_not_treated_as_missing_training_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            assistant = load_assistant(tmpdir)

            payload = json.loads(assistant.ask("现在训练状态怎么样", json_mode=True))

            self.assertEqual(payload["action"], "status")
            self.assertIn("训练状态概览", payload["message"])


if __name__ == "__main__":
    unittest.main()
