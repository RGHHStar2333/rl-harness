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
            "run_id": "mjlab_g1_test",
            "project_dir": "/tmp/mjlab",
            "task": "Mjlab-Velocity-Flat-Unitree-G1",
            "num_envs": 128,
            "max_iterations": 100,
            "wandb_project": "test",
            "wandb_name": "mjlab_g1_test",
        },
        "agent": {
            "learning_rate": learning_rate,
        },
        "reward_weights": {
            "track_linear_velocity": 2.0,
            "action_rate_l2": -0.1,
        },
    })

    write_yaml(rules_path, {
        "rules": [
            {
                "id": "mjlab_l1_plateau",
                "metric": "train/episode_reward_mean",
                "condition": {"type": "trend", "window": 5, "min_delta": 1.0},
                "response_level": "L1",
                "auto_adjustments": [
                    {
                        "target": "agent.learning_rate",
                        "operation": "multiply",
                        "value": 0.95,
                    },
                    {
                        "target": "reward_weights.track_linear_velocity",
                        "operation": "multiply",
                        "value": 1.05,
                    },
                ],
            },
            {
                "id": "mjlab_l2_plateau",
                "metric": "train/episode_reward_mean",
                "condition": {"type": "trend", "window": 10, "min_delta": 2.0},
                "response_level": "L2",
                "suggestions": [
                    {
                        "target": "agent.learning_rate",
                        "operation": "multiply",
                        "value": 0.8,
                        "reason": "test l2 adjustment",
                    },
                    {
                        "target": "reward_weights.track_linear_velocity",
                        "operation": "multiply",
                        "value": 1.2,
                        "reason": "test l2 reward weight adjustment",
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
                "allowed_target_prefixes": ["agent.", "reward_weights."],
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
            self.assertAlmostEqual(cfg["agent"]["learning_rate"], 0.00095)
            self.assertAlmostEqual(cfg["reward_weights"]["track_linear_velocity"], 2.1)

            rows = read_jsonl(paths["adjustments"])
            self.assertEqual([r["target"] for r in rows[-2:]], [
                "agent.learning_rate",
                "reward_weights.track_linear_velocity",
            ])
            for row in rows[-2:]:
                self.assertEqual(row["level"], "L1")
                self.assertEqual(row["trainer_kind"], "mjlab")
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
            self.assertAlmostEqual(cfg["agent"]["learning_rate"], 0.0008)
            self.assertAlmostEqual(cfg["reward_weights"]["track_linear_velocity"], 2.4)

            rows = read_jsonl(paths["adjustments"])
            self.assertEqual([r["target"] for r in rows[-2:]], [
                "agent.learning_rate",
                "reward_weights.track_linear_velocity",
            ])
            for row in rows[-2:]:
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

    def test_mjlab_value_loss_spike_triggers_hermes_l3_alert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_jsonl = tmp / "runs" / "mjlab_loss_spike_test" / "train.jsonl"
            rules_path = tmp / "rules.yaml"
            config_path = tmp / "go1.yaml"
            profile_path = tmp / "feedback.yaml"

            write_yaml(config_path, {
                "mjlab": {
                    "run_id": "mjlab_loss_spike_test",
                    "project_dir": "/tmp/mjlab",
                    "task": "Mjlab-Velocity-Flat-Unitree-G1",
                    "num_envs": 4096,
                    "max_iterations": 100,
                    "wandb_project": "test",
                    "wandb_name": "mjlab_loss_spike_test",
                },
                "agent": {"learning_rate": 0.001},
                "reward_weights": {"track_linear_velocity": 2.0},
            })
            write_yaml(rules_path, {
                "rules": [
                    {
                        "id": "mjlab_g1_value_loss_explosion_l3",
                        "description": "MJLab G1 value loss exploded to an unsafe level.",
                        "metric": "train/value_loss",
                        "condition": {
                            "type": "threshold",
                            "operator": ">",
                            "value": 1000,
                        },
                        "response_level": "L3",
                        "emergency_action": "pause_training",
                    },
                ],
            })
            write_yaml(profile_path, {
                "feedback_profile": {
                    "name": "mjlab_loss_spike_feedback",
                    "enabled": True,
                    "trainer_kind": "mjlab",
                    "require_active_training": False,
                    "project": {
                        "name": "rl_harness_mjlab_test",
                        "task": "Mjlab-Velocity-Flat-Unitree-G1",
                        "run_id": "mjlab_loss_spike_test",
                    },
                    "paths": {
                        "metric_log": str(train_jsonl),
                        "rules_config": str(rules_path),
                        "adjustable_config": str(config_path),
                        "adjustments_log": str(tmp / "adjustments.jsonl"),
                        "l2_pending_dir": str(tmp / "l2_pending"),
                        "state_path_prefix": str(tmp / "state" / "mjlab_loss_spike"),
                    },
                    "safety": {
                        "allowed_target_prefixes": ["agent.", "reward_weights."],
                        "l1_max_change_ratio": 0.1,
                        "l2_max_change_ratio": 0.3,
                    },
                    "restart": {"required_after_adjustment": True},
                },
            })

            train_jsonl.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {"train/step": 1000, "train/episode_reward_mean": 50.0, "train/value_loss": 0.02},
                {"train/step": 2000, "train/episode_reward_mean": 51.0, "train/value_loss": 50000.0},
            ]
            train_jsonl.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            output = run_main(monitor_hermes, ["--config", str(profile_path), "--debug"])

            self.assertIn("mjlab_g1_value_loss_explosion_l3", output)
            self.assertIn("train/value_loss", output)
            self.assertIn("当前值 50000.000 > 阈值 1000.000", output)
            self.assertIn("级别：L3", output)

    def test_mjlab_start_dry_run_consumes_learning_rate_and_reward_weights(self):
        result = subprocess.run(
            ["python3", "scripts/start_mjlab_training.py", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("--agent.algorithm.learning-rate 0.00095", result.stdout)
        self.assertIn("--env.rewards.track-linear-velocity.weight 2.1", result.stdout)
        self.assertIn("dry_run: true", result.stdout)

    def test_mjlab_adjustment_triggers_checkpoint_restart_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adjustments = Path(tmpdir) / "adjustments.jsonl"
            adjustments.write_text(json.dumps({
                "timestamp": time.time(),
                "run_id": "mjlab_g1_test",
                "trainer_kind": "mjlab",
                "level": "L1",
                "target": "agent.learning_rate",
                "old_value": 0.001,
                "new_value": 0.00095,
                "restart_required": True,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python3",
                    "scripts/feedback/auto_restart_if_needed.py",
                    "--since-line",
                    "0",
                    "--adjustments-path",
                    str(adjustments),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("MJLab 参数调整", result.stdout)
            self.assertIn("would_run: bash scripts/restart_mjlab_from_checkpoint.sh", result.stdout)

    def test_l3_emergency_suppresses_auto_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adjustments = Path(tmpdir) / "adjustments.jsonl"
            rows = [
                {
                    "timestamp": time.time(),
                    "run_id": "mjlab_g1_test",
                    "trainer_kind": "mjlab",
                    "level": "L1",
                    "target": "agent.learning_rate",
                    "old_value": 0.001,
                    "new_value": 0.00095,
                    "restart_required": True,
                },
                {
                    "timestamp": time.time(),
                    "run_id": "mjlab_g1_test",
                    "trainer_kind": "mjlab",
                    "level": "L3",
                    "target": "pause_training",
                    "restart_required": False,
                },
            ]
            adjustments.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    "scripts/feedback/auto_restart_if_needed.py",
                    "--since-line",
                    "0",
                    "--adjustments-path",
                    str(adjustments),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("L3 紧急记录", result.stdout)
            self.assertIn("跳过自动重启", result.stdout)
            self.assertNotIn("would_run:", result.stdout)

    def test_wandb_offline_decision_applies_and_logs_mjlab_adjustment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_path = tmp / "go1.yaml"
            runs_dir = tmp / "runs"
            history_path = tmp / "wandb_history.jsonl"
            adjustments_path = tmp / "adjustments.jsonl"

            write_yaml(cfg_path, {
                "mjlab": {
                    "run_id": "mjlab_g1_test",
                    "project_dir": "/tmp/mjlab",
                    "task": "Mjlab-Velocity-Flat-Unitree-G1",
                    "num_envs": 128,
                    "max_iterations": 100,
                    "wandb_project": "test",
                    "wandb_entity": "entity",
                    "wandb_name": "mjlab_g1_test",
                },
                "agent": {"learning_rate": 0.001},
                "reward_weights": {
                    "track_linear_velocity": 2.0,
                    "track_angular_velocity": 2.0,
                },
                "autotune": {
                    "velocity_error_threshold": 0.25,
                    "l3_reward_crash_threshold": -30.0,
                    "l1_cooldown_iterations": 500,
                },
            })

            rows = []
            for step in range(1, 51):
                rows.append({
                    "_step": step,
                    "reward": 50.0,
                    "track_linear_velocity": 1.0,
                    "track_angular_velocity": 1.0,
                    "error_vel_xy": 0.8,
                    "error_vel_yaw": 0.1,
                    "fell_over": 0.0,
                })
            history_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    "scripts/wandb_mjlab_autodecide.py",
                    "--config",
                    str(cfg_path),
                    "--runs-dir",
                    str(runs_dir),
                    "--history-jsonl",
                    str(history_path),
                    "--adjustments-path",
                    str(adjustments_path),
                    "--apply",
                    "--no-restart",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("W&B 自动决策结果", result.stdout)

            cfg = read_yaml(cfg_path)
            self.assertAlmostEqual(cfg["reward_weights"]["track_linear_velocity"], 2.2)

            decision_path = runs_dir / "mjlab_g1_test" / "wandb_decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["severity"], "L1")
            self.assertTrue(decision["changed"])

            row = read_jsonl(adjustments_path)[-1]
            self.assertEqual(row["trainer_kind"], "mjlab")
            self.assertEqual(row["source"], "wandb")
            self.assertEqual(row["target"], "reward_weights.track_linear_velocity")
            self.assertTrue(row["restart_required"])

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
