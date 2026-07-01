# MJLab G1 L1/L2/L3 And Reward Tuning Completion Report

Date: 2026-06-30

## Completed

- Unified MJLab feedback targets with the active `configs/tasks/mjlab/go1.yaml` structure:
  - `agent.learning_rate`
  - `reward_weights.*`
- Updated MJLab G1 L1 rules to automatically adjust both learning rate and reward weight.
- Updated MJLab G1 L2 rules so confirmed tokens can adjust both learning rate and reward weights.
- Kept L3 on the MJLab profile path, pausing the process from `runs/active_training.json`.
- Updated mechanical lint to validate `agent.learning_rate`, `reward_weights.*`, and multi-target `auto_adjustments`.
- Updated MJLab launch and checkpoint restart paths to pass adjusted values into MJLab CLI overrides:
  - `--agent.algorithm.learning-rate`
  - `--env.rewards.*.weight`

## Verification

Commands run:

```bash
python3 scripts/ops/lint_config.py
python3 -m unittest tests.test_mjlab_feedback_control tests.test_parse_mjlab_metrics
python3 -m py_compile scripts/start_mjlab_training.py scripts/feedback/monitor_hermes.py scripts/ops/lint_config.py tests/test_mjlab_feedback_control.py
bash scripts/mjlab/start_mjlab_training.sh --dry-run
bash -n scripts/mjlab/start_mjlab_training.sh scripts/start_mjlab_training.sh scripts/restart_mjlab_from_checkpoint.sh
```

Results:

- Config lint passed.
- Feedback and parser tests passed: 8 tests.
- Python syntax checks passed.
- Bash syntax checks passed.
- MJLab dry-run rendered learning-rate and reward-weight CLI overrides.

## Current Status

These two items can now be marked complete:

- L1/L2/L3 truly act on MJLab G1 through the MJLab feedback profile.
- MJLab learning rate and reward weights are modified by Harness and consumed by MJLab launch/restart commands.
