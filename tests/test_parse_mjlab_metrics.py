import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mjlab" / "parse_mjlab_metrics.py"

spec = importlib.util.spec_from_file_location("parse_mjlab_metrics", MODULE_PATH)
parse_mjlab_metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parse_mjlab_metrics
spec.loader.exec_module(parse_mjlab_metrics)


COMPLETE_BLOCK = """
################################################################################
\x1b[1m                         Learning iteration 9999/10000                          \x1b[0m

                               Run name: mjlab_g1_4096_10000
                            Total steps: 983040000
                       Steps per second: 151347
                        Collection time: 0.582s
                          Learning time: 0.068s
                        Mean value loss: 0.0242
                    Mean surrogate loss: -0.0126
                      Mean entropy loss: 9.8079
                            Mean reward: 57.66
                    Mean episode length: 1000.00
                        Mean action std: 0.39
  Episode_Reward/track_linear_velocity: 1.5061
          Episode_Reward/action_rate_l2: -0.8839
             Metrics/twist/error_vel_xy: 0.6743
--------------------------------------------------------------------------------
"""


PARTIAL_BLOCK = """
################################################################################
\x1b[1m                         Learning iteration 10000/10000                          \x1b[0m

                               Run name: mjlab_g1_4096_10000
                            Total steps: 983138304
"""


RESTARTED_LOG = """
================================================================================
[2026-06-29T16:36:44] START
cmd: uv run train Mjlab-Velocity-Flat-Unitree-G1 --agent.max-iterations 19999

################################################################################
\x1b[1m                         Learning iteration 0/19999                          \x1b[0m

                               Run name: mjlab_g1_4096_10000
                            Total steps: 98304
                            Mean reward: 1.00

================================================================================
[2026-06-30T10:04:29] START
cmd: uv run train Mjlab-Velocity-Flat-Unitree-G1 --agent.max-iterations 10000

################################################################################
\x1b[1m                         Learning iteration 0/10000                          \x1b[0m

                               Run name: mjlab_g1_4096_10000
                            Total steps: 98304
                            Mean reward: -6.32
"""


class ParseMjlabMetricsTest(unittest.TestCase):
    def test_parse_complete_block(self):
        rows = parse_mjlab_metrics.parse_mjlab_log_lines(
            COMPLETE_BLOCK.splitlines(),
            run_id="mjlab_g1_4096_10000",
            task="Mjlab-Velocity-Flat-Unitree-G1",
            log_path="/tmp/training_process.log",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["mjlab/iteration"], 9999)
        self.assertEqual(row["mjlab/max_iterations"], 10000)
        self.assertEqual(row["train/step"], 983040000)
        self.assertEqual(row["train/episode_reward_mean"], 57.66)
        self.assertEqual(row["train/action_std_mean"], 0.39)
        self.assertEqual(row["mjlab/reward/track_linear_velocity"], 1.5061)
        self.assertEqual(row["mjlab/reward/action_rate_l2"], -0.8839)
        self.assertEqual(row["mjlab/metric/twist/error_vel_xy"], 0.6743)

    def test_sync_is_idempotent_and_ignores_partial_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "training_process.log"
            output_path = Path(tmpdir) / "train.jsonl"
            log_path.write_text(COMPLETE_BLOCK + PARTIAL_BLOCK, encoding="utf-8")

            first = parse_mjlab_metrics.sync_metrics(
                str(log_path),
                str(output_path),
                run_id="mjlab_g1_4096_10000",
                task="Mjlab-Velocity-Flat-Unitree-G1",
            )
            second = parse_mjlab_metrics.sync_metrics(
                str(log_path),
                str(output_path),
                run_id="mjlab_g1_4096_10000",
                task="Mjlab-Velocity-Flat-Unitree-G1",
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(first["parsed"], 1)
            self.assertEqual(first["appended"], 1)
            self.assertEqual(second["parsed"], 1)
            self.assertEqual(second["appended"], 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mjlab/iteration"], 9999)

    def test_missing_log_is_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "train.jsonl"

            summary = parse_mjlab_metrics.sync_metrics(
                str(Path(tmpdir) / "missing.log"),
                str(output_path),
                run_id="missing",
                task=None,
            )

            self.assertEqual(summary["reason"], "log_missing")
            self.assertFalse(output_path.exists())

    def test_sync_appends_latest_session_after_iteration_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "training_process.log"
            output_path = Path(tmpdir) / "train.jsonl"
            log_path.write_text(RESTARTED_LOG, encoding="utf-8")
            output_path.write_text(
                json.dumps(
                    {
                        "mjlab/iteration": 0,
                        "mjlab/max_iterations": 19999,
                        "train/step": 98304,
                        "train/episode_reward_mean": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            first = parse_mjlab_metrics.sync_metrics(
                str(log_path),
                str(output_path),
                run_id="mjlab_g1_4096_10000",
                task="Mjlab-Velocity-Flat-Unitree-G1",
            )
            second = parse_mjlab_metrics.sync_metrics(
                str(log_path),
                str(output_path),
                run_id="mjlab_g1_4096_10000",
                task="Mjlab-Velocity-Flat-Unitree-G1",
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(first["parsed"], 2)
            self.assertEqual(first["appended"], 1)
            self.assertEqual(second["appended"], 0)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["mjlab/session_index"], 1)
            self.assertEqual(rows[-1]["mjlab/iteration"], 0)
            self.assertEqual(rows[-1]["mjlab/max_iterations"], 10000)
            self.assertEqual(rows[-1]["train/episode_reward_mean"], -6.32)


if __name__ == "__main__":
    unittest.main()
