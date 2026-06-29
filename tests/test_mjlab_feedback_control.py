import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = ROOT / "scripts" / "feedback"
if str(FEEDBACK_DIR) not in sys.path:
    sys.path.insert(0, str(FEEDBACK_DIR))

import feedback_profile  # noqa: E402
import l2_check  # noqa: E402
import l2_decide  # noqa: E402
import l3_check  # noqa: E402
import monitor_hermes  # noqa: E402


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_metrics(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, value in enumerate(values, start=1):
        rows.append({
            "timestamp": time.time(),
            "train/step": i * 10000,
            "train/episode_reward_mean": value,
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_main(module, argv):
    old_argv = sys.argv[:]
    stdout = io.StringIO()
    sys.argv = [str(Path(module.__file__).name)] + argv
    try:
        with contextlib.redirect_stdout(stdout):
            module.main()
    finally:
        sys.argv = old_argv
    return stdout.getvalue()


def make_mjlab_profile(tmpdir, require_active=False, learning_rate=0.001):
    tmp = Path(tmpdir)
    train_jsonl = tmp / "runs" / "mjlab_g1_test" / "train.jsonl"
    rules_path = tmp / "rules.yaml"
    config_path = tmp / "go1.yaml"
    adjustments_path = tmp / "adjustments.jsonl"
    pending_dir = tmp / "l2_pending"
    profile_path = tmp / "feedback.yaml"

    write_yaml(config_path, {
        "mjlab": {
            "agent": {
                "algorithm": {
                    "learning_rate": learning_rate,
                },
            },
        },
    })

    write_yaml(rules_path, {
        "rules": [
            {
                "id": "mjlab_l1_plateau",
                "metric": "train/episode_reward_mean",
                "condition": {"type": "trend", "window": 5, "min_delta": 1.0},
                "response_level": "L1",
                "auto_adjust": {
                    "target": "mjlab.agent.algorithm.learning_rate",
                    "operation": "multiply",
                    "value": 0.95,
                },
            },
            {
                "id": "mjlab_l2_plateau",
                "metric": "train/episode_reward_mean",
                "condition": {"type": "trend", "window": 10, "min_delta": 2.0},
                "response_level": "L2",
                "suggestions": [
                    {
                        "target": "mjlab.agent.algorithm.learning_rate",
                        "operation": "multiply",
                        "value": 0.8,
                        "reason": "test l2 adjustment",
                    },
                ],
            },
            {
                "id": "mjlab_l3_crash",
                "metric": "train/episode_reward_mean",
                "condition": {"type": "threshold", "operator": "<", "value": 5},
                "response_level": "L3",
                "emergency_action": "pause_training",
            },
        ],
    })

    write_yaml(profile_path, {
        "feedback_profile": {
            "name": "mjlab_g1_test_feedback",
            "enabled": True,
            "trainer_kind": "mjlab",
            "require_active_training": require_active,
            "project": {
                "name": "rl_harness_mjlab_test",
                "task": "Mjlab-Velocity-Flat-Unitree-G1",
                "run_id": "mjlab_g1_test",
            },
            "paths": {
                "metric_log": str(train_jsonl),
                "rules_config": str(rules_path),
                "adjustable_config": str(config_path),
                "adjustments_log": str(adjustments_path),
                "l2_pending_dir": str(pending_dir),
                "state_path_prefix": str(tmp / "state" / "mjlab_g1_test"),
            },
            "safety": {
                "allowed_target_prefixes": ["mjlab.agent.algorithm."],
                "l1_max_change_ratio": 0.1,
                "l2_max_change_ratio": 0.3,
            },
            "restart": {
                "required_after_adjustment": True,
            },
        },
    })

    return {
        "profile": profile_path,
        "config": config_path,
        "rules": rules_path,
        "metrics": train_jsonl,
        "adjustments": adjustments_path,
        "pending": pending_dir,
    }


class MjlabFeedbackControlTest(unittest.TestCase):
    def test_mjlab_l1_plateau_adjusts_learning_rate_and_logs_restart_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = make_mjlab_profile(tmpdir)
            write_metrics(paths["metrics"], [100.0, 100.3, 100.2, 100.4, 100.5])

            context = feedback_profile.load_feedback_context(str(paths["profile"]))
            rules = feedback_profile.load_yaml(str(paths["rules"]))["rules"]
            rows = monitor_hermes.load_jsonl(str(paths["metrics"]))
            points = monitor_hermes.get_metric_points(rows, "train/episode_reward_mean")
            triggered, reason, latest = monitor_hermes.check_rule(rules[0], points)

            self.assertTrue(triggered)
            ok, summary = monitor_hermes.apply_l1_auto_adjust(context, rules[0], reason, latest)

            self.assertTrue(ok, summary)
            cfg = read_yaml(paths["config"])
            self.assertAlmostEqual(cfg["mjlab"]["agent"]["algorithm"]["learning_rate"], 0.00095)

            row = read_jsonl(paths["adjustments"])[-1]
            self.assertEqual(row["level"], "L1")
            self.assertEqual(row["trainer_kind"], "mjlab")
            self.assertEqual(row["target"], "mjlab.agent.algorithm.learning_rate")
            self.assertTrue(row["restart_required"])

    def test_mjlab_l2_creates_token_and_confirm_updates_learning_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = make_mjlab_profile(tmpdir)
            write_metrics(paths["metrics"], [200, 201, 200.5, 201.2, 200.8, 201.0, 200.7, 201.1, 200.9, 201.0])

            output = run_main(l2_check, ["--config", str(paths["profile"]), "--debug"])
            proposals = list(paths["pending"].glob("*.json"))

            self.assertIn("确认口令 token", output)
            self.assertEqual(len(proposals), 1)

            token = proposals[0].stem
            confirm_output = run_main(
                l2_decide,
                ["--token", token, "--decision", "confirm", "--pending-dir", str(paths["pending"])],
            )

            self.assertIn("L2 调整已确认并执行", confirm_output)
            cfg = read_yaml(paths["config"])
            self.assertAlmostEqual(cfg["mjlab"]["agent"]["algorithm"]["learning_rate"], 0.0008)

            row = read_jsonl(paths["adjustments"])[-1]
            self.assertEqual(row["level"], "L2")
            self.assertEqual(row["decision"], "confirmed")
            self.assertEqual(row["trainer_kind"], "mjlab")
            self.assertTrue(row["restart_required"])

    def test_mjlab_l3_crash_pauses_dummy_active_training(self):
        active_path = ROOT / "runs" / "active_training.json"
        original_active = active_path.read_bytes() if active_path.exists() else None
        proc = None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = make_mjlab_profile(tmpdir, require_active=True)
                write_metrics(paths["metrics"], [-1500.0])

                proc = subprocess.Popen(["sleep", "9999"], start_new_session=True)
                active_path.parent.mkdir(parents=True, exist_ok=True)
                active_path.write_text(json.dumps({
                    "status": "running",
                    "kind": "mjlab",
                    "run_id": "mjlab_g1_test",
                    "pid": proc.pid,
                    "command": "sleep 9999",
                    "log": str(Path(tmpdir) / "dummy.log"),
                }), encoding="utf-8")

                output = run_main(l3_check, ["--config", str(paths["profile"]), "--debug"])
                proc.wait(timeout=5)

                self.assertIn("L3 紧急告警", output)
                self.assertIsNotNone(proc.poll())

                state = json.loads(active_path.read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "paused")

                row = read_jsonl(paths["adjustments"])[-1]
                self.assertEqual(row["level"], "L3")
                self.assertEqual(row["trainer_kind"], "mjlab")
                self.assertEqual(row["target"], "pause_training")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
            if original_active is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_bytes(original_active)

    def test_halfcheetah_ppo_l1_and_legacy_l2_still_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            hyper_path = tmp / "hyper.yaml"
            rules_path = tmp / "rules.yaml"
            pipeline_path = tmp / "pipeline.yaml"
            adjustments_path = tmp / "adjustments.jsonl"
            pending_dir = tmp / "l2_pending"

            write_yaml(hyper_path, {"ppo": {"learning_rate": 0.01}})
            write_yaml(rules_path, {
                "rules": [
                    {
                        "id": "half_l1",
                        "metric": "train/episode_reward_mean",
                        "condition": {"type": "trend", "window": 5, "min_delta": 10.0},
                        "response_level": "L1",
                        "auto_adjust": {
                            "target": "ppo.learning_rate",
                            "operation": "multiply",
                            "value": 0.95,
                        },
                    },
                ],
            })
            write_yaml(pipeline_path, {
                "project": {"name": "test", "task": "half_cheetah", "run_id": "half_test"},
                "paths": {
                    "run_dir": str(tmp / "run"),
                    "hyper_config": str(hyper_path),
                    "detection_rules": str(rules_path),
                    "adjustments_log": str(adjustments_path),
                    "l2_pending_dir": str(pending_dir),
                },
                "feedback_flywheel": {"enabled": True, "l1_max_change_ratio": 0.1, "l2_max_change_ratio": 0.3},
            })

            context = feedback_profile.load_feedback_context(str(pipeline_path))
            rule = feedback_profile.load_yaml(str(rules_path))["rules"][0]
            ok, summary = monitor_hermes.apply_l1_auto_adjust(
                context,
                rule,
                "halfcheetah regression l1",
                {"step": 100, "value": 1.0},
            )

            self.assertTrue(ok, summary)
            self.assertAlmostEqual(read_yaml(hyper_path)["ppo"]["learning_rate"], 0.0095)

            write_yaml(hyper_path, {"ppo": {"learning_rate": 0.01}})
            pending_dir.mkdir(parents=True, exist_ok=True)
            token = "abcdef12"
            proposal = {
                "token": token,
                "status": "pending",
                "created_at": time.time(),
                "run_id": "half_test",
                "project": {"name": "test", "task": "half_cheetah", "run_id": "half_test"},
                "rule": {
                    "id": "half_l2",
                    "suggestions": [
                        {
                            "target": "ppo.learning_rate",
                            "operation": "multiply",
                            "value": 0.8,
                            "reason": "legacy ppo l2",
                        },
                    ],
                },
                "reason": "legacy halfcheetah l2",
                "latest": {"step": 200, "value": 2.0},
                "hyper_config": str(hyper_path),
                "adjustments_path": str(adjustments_path),
                "l2_max_change_ratio": 0.3,
            }
            (pending_dir / f"{token}.json").write_text(json.dumps(proposal), encoding="utf-8")

            run_main(l2_decide, ["--token", token, "--decision", "confirm", "--pending-dir", str(pending_dir)])

            self.assertAlmostEqual(read_yaml(hyper_path)["ppo"]["learning_rate"], 0.008)
            row = read_jsonl(adjustments_path)[-1]
            self.assertEqual(row["level"], "L2")
            self.assertEqual(row["target"], "ppo.learning_rate")
            self.assertEqual(row["trainer_kind"], "sb3")


if __name__ == "__main__":
    unittest.main()
