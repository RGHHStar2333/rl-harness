import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class FakeProc:
    def __init__(self, pid):
        self.pid = pid


def load_module(tmp_root):
    spec = importlib.util.spec_from_file_location(f"hermes_queue_{time.time_ns()}", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.ROOT = Path(tmp_root)
    module.QUEUE_DIR = module.ROOT / "runs" / "training_queue"
    module.QUEUE_PATH = module.QUEUE_DIR / "queue.json"
    module.EVENTS_PATH = module.QUEUE_DIR / "events.jsonl"
    module.ACTIVE_PATH = module.ROOT / "runs" / "active_training.json"
    module.MJLAB_CONFIG_PATH = module.ROOT / "configs" / "tasks" / "mjlab" / "go1.yaml"
    module.MJLAB_FEEDBACK_PATH = module.ROOT / "configs" / "tasks" / "mjlab" / "feedback.yaml"
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
            "wandb_entity": "entity",
            "wandb_name": "mjlab_g1_base",
        },
        "agent": {"learning_rate": 0.001},
        "reward_weights": {
            "track_linear_velocity": 2.0,
            "action_rate_l2": -0.1,
        },
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
            "safety": {
                "allowed_target_prefixes": ["agent.", "reward_weights."],
                "l1_max_change_ratio": 0.1,
                "l2_max_change_ratio": 0.3,
            },
            "restart": {"required_after_adjustment": True},
        },
    })


class HermesTrainingQueueTest(unittest.TestCase):
    def test_parse_single_and_batch_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            module = load_module(tmpdir)

            text = "\n".join([
                "1. G1 并行数 4096 训练次数 8000 1小时",
                "2. G1 2048并行 5000次",
                "Hermes 每3个小时自动切换到下一个训练",
            ])
            jobs = module.parse_requests(text)

            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["num_envs"], 4096)
            self.assertEqual(jobs[0]["max_iterations"], 8000)
            self.assertEqual(jobs[0]["max_runtime_minutes"], 60)
            self.assertEqual(jobs[1]["num_envs"], 2048)
            self.assertEqual(jobs[1]["max_iterations"], 5000)
            self.assertEqual(jobs[1]["max_runtime_minutes"], 180)

    def test_enqueue_persists_jobs_and_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            module = load_module(tmpdir)

            jobs = module.enqueue_text("G1 128并行 10次 30分钟", source="feishu")
            queue = json.loads(module.QUEUE_PATH.read_text(encoding="utf-8"))
            events = module.EVENTS_PATH.read_text(encoding="utf-8")

            self.assertEqual(len(jobs), 1)
            self.assertEqual(queue["jobs"][0]["source"], "feishu")
            self.assertEqual(queue["jobs"][0]["status"], "queued")
            self.assertIn("enqueued", events)

    def test_tick_starts_next_job_and_updates_active_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            module = load_module(tmpdir)
            module.enqueue_text("G1 128并行 10次 30分钟")
            module.launch_process = lambda cmd, project_dir, log_path: FakeProc(4242)
            module.pid_alive = lambda pid: True

            messages = module.tick()
            queue = module.load_queue()
            active = json.loads(module.ACTIVE_PATH.read_text(encoding="utf-8"))
            go1 = read_yaml(module.MJLAB_CONFIG_PATH)
            feedback = read_yaml(module.MJLAB_FEEDBACK_PATH)

            self.assertIn("started", "\n".join(messages))
            self.assertEqual(queue["jobs"][0]["status"], "running")
            self.assertTrue(active["queue_managed"])
            self.assertEqual(go1["mjlab"]["num_envs"], 128)
            self.assertEqual(go1["mjlab"]["max_iterations"], 10)
            self.assertEqual(feedback["feedback_profile"]["project"]["run_id"], queue["jobs"][0]["run_id"])

    def test_tick_stops_runtime_limit_and_starts_next_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            module = load_module(tmpdir)
            job1, job2 = module.parse_requests("G1 128并行 10次 1分钟\nG1 256并行 20次 1分钟")
            job1.update({
                "status": "running",
                "started_at": module.now_iso(),
                "started_at_ts": time.time() - 120,
                "pid": 111,
                "log": str(Path(tmpdir) / "runs" / job1["run_id"] / "training_process.log"),
            })
            module.save_queue({"version": 1, "jobs": [job1, job2]})
            module.save_json(module.ACTIVE_PATH, {
                "status": "running",
                "kind": "mjlab",
                "queue_managed": True,
                "queue_job_id": job1["id"],
                "run_id": job1["run_id"],
                "pid": 111,
            })
            module.pid_alive = lambda pid: True
            module.pause_active_training = lambda reason: (0, "paused")
            module.launch_process = lambda cmd, project_dir, log_path: FakeProc(222)

            messages = module.tick()
            queue = module.load_queue()

            self.assertIn("runtime_limit", "\n".join(messages))
            self.assertEqual(queue["jobs"][0]["status"], "completed")
            self.assertEqual(queue["jobs"][1]["status"], "running")
            self.assertEqual(queue["jobs"][1]["pid"], 222)

    def test_tick_stops_iteration_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root(tmpdir)
            module = load_module(tmpdir)
            job = module.parse_requests("G1 128并行 10次 1小时")[0]
            job.update({
                "status": "running",
                "started_at": module.now_iso(),
                "started_at_ts": time.time(),
                "pid": 111,
                "log": str(Path(tmpdir) / "runs" / job["run_id"] / "training_process.log"),
            })
            train_path = Path(tmpdir) / "runs" / job["run_id"] / "train.jsonl"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.write_text(json.dumps({
                "mjlab/run_id": job["run_id"],
                "mjlab/iteration": 9,
                "train/step": 999,
                "train/episode_reward_mean": 10.0,
            }) + "\n", encoding="utf-8")
            module.save_queue({"version": 1, "jobs": [job]})
            module.save_json(module.ACTIVE_PATH, {
                "status": "running",
                "kind": "mjlab",
                "queue_managed": True,
                "queue_job_id": job["id"],
                "run_id": job["run_id"],
                "pid": 111,
            })
            module.pid_alive = lambda pid: True
            module.pause_active_training = lambda reason: (0, "paused")

            messages = module.tick()
            queue = module.load_queue()

            self.assertIn("iteration_limit", "\n".join(messages))
            self.assertEqual(queue["jobs"][0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
