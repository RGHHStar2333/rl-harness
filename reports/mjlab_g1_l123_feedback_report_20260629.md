# MJLab G1 L1/L2/L3 Feedback Implementation Report

Date: 2026-06-29

## Summary

Harness now has a MJLab G1 feedback profile that reads `runs/mjlab_g1_4096_10000/train.jsonl` and applies L1/L2/L3 decisions against MJLab-owned configuration.

The MJLab adjustable target is:

```yaml
mjlab.agent.algorithm.learning_rate
```

MJLab L1/L2 adjustments are written to `configs/tasks/mjlab/go1.yaml` and logged with `trainer_kind: mjlab` plus `restart_required: true`. They take effect after restarting MJLab training or on the next launch. They do not trigger the HalfCheetah checkpoint restart path.

## Implemented Behavior

- L1 mild plateau: automatically multiplies `mjlab.agent.algorithm.learning_rate` by `0.95`, within the configured max-change ratio.
- L2 longer plateau: creates a confirmation token under `runs/l2_pending`; confirmation applies the MJLab learning-rate change.
- L3 reward crash: pauses the process recorded in `runs/active_training.json` when it matches the MJLab profile and is still running.
- Hermes entry point: keeps the existing HalfCheetah pass, refreshes MJLab metrics, then runs MJLab G1 feedback checks.
- MJLab start dry-run: renders the `uv run train` command with `--agent.algorithm.learning-rate`.

## Safety Controls

- MJLab targets are restricted to `mjlab.agent.algorithm.` by profile whitelist.
- Mechanical lint validates the MJLab feedback profile, rules file, metric path shape, allowed target prefixes, and positive MJLab learning rate.
- MJLab profile requires matching active training by default before automated feedback actions.
- `auto_restart_if_needed.py` skips MJLab rows and only auto-restarts SB3/HalfCheetah rows.

## Verification

Commands run:

```bash
python3 scripts/ops/lint_config.py
python3 -m py_compile scripts/feedback/feedback_profile.py scripts/feedback/monitor_hermes.py scripts/feedback/l2_check.py scripts/feedback/l2_decide.py scripts/feedback/l3_check.py scripts/feedback/auto_restart_if_needed.py scripts/ops/lint_config.py
bash scripts/mjlab/start_mjlab_training.sh --dry-run
python3 -m unittest tests.test_mjlab_feedback_control
python3 -m unittest tests.test_parse_mjlab_metrics
```

Results:

- Config lint passed.
- Python syntax checks passed.
- MJLab dry-run rendered a command containing `--agent.algorithm.learning-rate 0.001`.
- MJLab feedback tests passed: L1 adjustment, L2 token/confirm, L3 dummy pause.
- Parser regression tests passed.

## Operator Notes

Manual MJLab feedback pass:

```bash
bash scripts/run_mjlab_g1_feedback.sh --debug
```

MJLab launch dry-run:

```bash
bash scripts/mjlab/start_mjlab_training.sh --dry-run
```

The current `runs/active_training.json` marks `mjlab_g1_4096_10000` as running, but its recorded PID was not alive when checked during implementation. The feedback gate will skip MJLab actions when the recorded PID is not running.
