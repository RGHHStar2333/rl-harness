# Scripts

The script folder is organized by responsibility. Top-level files are compatibility wrappers for existing commands and scheduled jobs; implementation files live in subdirectories.

## Training

- `training/train.py`: Stable-Baselines3 PPO training entry point.
- `training/start_training.sh`: managed background training launcher.
- `training/restart_from_checkpoint.sh`: pause and resume from the latest checkpoint.
- `training/pause_training.py`: pause the managed training process.
- `training/find_latest_checkpoint.py`: locate the newest resumable checkpoint.
- `training/watch_robot.py`: render the latest MuJoCo checkpoint.

## Feedback Flywheel

- `feedback/monitor_hermes.py`: L1/L2/L3 scan and Hermes/Feishu message formatter.
- `run_monitor_for_hermes.sh`: top-level scheduled Hermes entry point.
- `feedback/l2_check.py`: create pending L2 proposals and confirmation tokens.
- `feedback/l2_decide.py`: apply or reject an L2 proposal.
- `feedback/l2_confirm.sh`: confirm an L2 token and restart if needed.
- `feedback/l2_reject.sh`: reject an L2 token.
- `feedback/l3_check.py`: trigger emergency pause on L3 rules.
- `feedback/auto_restart_if_needed.py`: restart SB3 or MJLab training after L1 or confirmed L2 changes.

## Hermes Training Queue

- `hermes_training_request.py`: Feishu/Hermes-facing text entry point for adding one or more MJLab jobs to the queue.
- `training_queue/hermes_queue.py`: queue engine with `enqueue`, `status`, `tick`, `cancel`, and `clear-completed`.

Examples:

```bash
python3 scripts/hermes_training_request.py --text "G1 4096并行 8000次 1小时"
python3 scripts/training_queue/hermes_queue.py status
python3 scripts/training_queue/hermes_queue.py tick
```

The scheduled Hermes monitor calls `training_queue/hermes_queue.py tick` before the normal feedback pass, so queued jobs can start, stop at runtime or iteration limits, and advance to the next job automatically.

## Maintenance

- `ops/lint_config.py`: mechanical config validation.
- `ops/entropy_scan.py`: checkpoint entropy report generator.
- `ops/run_entropy_scan.sh`: scheduled entropy scan wrapper.
- `ops/git_auto_commit.py`: commit allowed code/config/adjustment changes.

## MJLab

- `mjlab/start_mjlab_training.sh`: launch MJLab Go1 training.
- `mjlab/play_mjlab.sh`: play the MJLab task.
- `mjlab/parse_mjlab_metrics.py`: parse MJLab `training_process.log` into Harness-compatible `train.jsonl`.
- `mjlab/run_g1_feedback.sh`: refresh MJLab G1 metrics and run the MJLab G1 L1/L2/L3 feedback pass.
- `run_mjlab_g1_feedback.sh`: compatibility wrapper for the MJLab G1 feedback pass.
- `run_wandb_mjlab_autodecide.sh`: run the configured W&B curve analysis and write MJLab decision output.
- `wandb_mjlab_autodecide.py`: read W&B curves, decide MJLab adjustments, and optionally apply them.

Sync MJLab metrics for the configured run:

```bash
python scripts/parse_mjlab_metrics.py --config configs/tasks/mjlab/go1.yaml
```

Run the MJLab G1 feedback pass manually:

```bash
bash scripts/run_mjlab_g1_feedback.sh --debug
```

The scheduled Hermes entry point also refreshes MJLab metrics and runs the MJLab G1 feedback profile after the existing HalfCheetah pass:

```bash
bash scripts/run_monitor_for_hermes.sh --debug
```

MJLab G1 L1/L2 changes update `configs/tasks/mjlab/go1.yaml` at `agent.learning_rate` and `reward_weights.*`. They are recorded with `trainer_kind: mjlab` and `restart_required: true`, so they take effect after restarting MJLab training or on the next launch. The MJLab launch/restart commands translate those values into `--agent.algorithm.learning-rate` and `--env.rewards.*.weight` CLI overrides.

When MJLab adjustment rows are appended to `runs/adjustments.jsonl`, `feedback/auto_restart_if_needed.py` calls `scripts/restart_mjlab_from_checkpoint.sh` and resumes from the latest MJLab checkpoint.

Run W&B curve analysis manually:

```bash
bash scripts/run_wandb_mjlab_autodecide.sh
```

The scheduled Hermes entry point calls this wrapper after the MJLab feedback pass. With `autotune.wandb_autodecide_apply: false`, it writes `runs/<run_id>/wandb_decision.json` and history without changing config. With `autotune.wandb_autodecide_apply: true`, it applies config changes, appends MJLab adjustment records, and lets the shared auto-restart step resume from checkpoint.

Verify the MJLab launch command without starting training:

```bash
bash scripts/mjlab/start_mjlab_training.sh --dry-run
```

## Compatibility

Existing commands like `python scripts/train.py` and `bash scripts/start_training.sh` still work through top-level wrappers.
