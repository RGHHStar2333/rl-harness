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
- `feedback/auto_restart_if_needed.py`: restart training after L1 or confirmed L2 changes.

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

MJLab G1 L1/L2 changes update `configs/tasks/mjlab/go1.yaml` at `mjlab.agent.algorithm.learning_rate`. They are recorded with `trainer_kind: mjlab` and `restart_required: true`, so they take effect after restarting MJLab training or on the next launch. They do not call the HalfCheetah checkpoint restart script.

Verify the MJLab launch command without starting training:

```bash
bash scripts/mjlab/start_mjlab_training.sh --dry-run
```

## Compatibility

Existing commands like `python scripts/train.py` and `bash scripts/start_training.sh` still work through top-level wrappers.
