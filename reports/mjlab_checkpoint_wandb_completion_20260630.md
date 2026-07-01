# MJLab Checkpoint Restart And W&B Autodecision Completion Report

Date: 2026-06-30

## Completed

- MJLab adjustment rows now trigger `scripts/restart_mjlab_from_checkpoint.sh` through `scripts/feedback/auto_restart_if_needed.py`.
- The restart detector supports both current profile rows (`trainer_kind: mjlab`, `restart_required: true`) and legacy MJLab autotune rows (`system: mjlab`, `requires_restart: true`).
- `scripts/start_mjlab_feedback_loop.sh` now runs the shared auto-restart step after `mjlab_auto_tune.py`.
- W&B curve analysis is connected through `scripts/run_wandb_mjlab_autodecide.sh`.
- `scripts/run_monitor_for_hermes.sh` now invokes W&B MJLab autodecision after the MJLab L1/L2/L3 feedback pass.
- `scripts/wandb_mjlab_autodecide.py` now supports:
  - offline history fixtures for tests,
  - configurable config/runs/adjustments paths,
  - decision JSON output,
  - adjustment logging when `--apply` is used,
  - central checkpoint restart handoff through `--no-restart`.

## Operating Mode

Current config:

```yaml
autotune:
  wandb_autodecide_enabled: true
  wandb_autodecide_apply: false
```

This means scheduled Hermes runs will analyze W&B and write decision output without mutating config. To allow W&B decisions to apply config and trigger checkpoint restart, set:

```yaml
autotune:
  wandb_autodecide_apply: true
```

When apply is enabled, the wrapper runs W&B autodecision with `--apply --no-restart`; it writes MJLab adjustment rows, then the shared auto-restart step calls `scripts/restart_mjlab_from_checkpoint.sh`.

## Verification

Commands run:

```bash
python3 scripts/ops/lint_config.py
python3 -m unittest tests.test_mjlab_feedback_control tests.test_parse_mjlab_metrics
python3 -m py_compile scripts/feedback/auto_restart_if_needed.py scripts/wandb_mjlab_autodecide.py tests/test_mjlab_feedback_control.py
bash -n scripts/run_monitor_for_hermes.sh scripts/run_wandb_mjlab_autodecide.sh scripts/start_mjlab_feedback_loop.sh scripts/restart_mjlab_from_checkpoint.sh
```

Results:

- Config lint passed.
- Feedback/parser/W&B offline tests passed: 10 tests.
- Python syntax checks passed.
- Bash syntax checks passed.
- Auto-restart dry-run test confirmed MJLab adjustment rows would call `bash scripts/restart_mjlab_from_checkpoint.sh`.
- W&B offline test confirmed a curve-derived L1 decision writes `wandb_decision.json`, updates reward weight when `--apply` is used, and appends a MJLab restart-required adjustment row.

## Current Status

These two items can now be marked complete:

- MJLab checkpoint is connected to automatic restart after Harness-driven MJLab adjustments.
- W&B curve analysis is connected to automatic decision output and optional apply/restart flow.
