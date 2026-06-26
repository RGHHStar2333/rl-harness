# Scripts

The script folder is organized by responsibility even though files currently stay flat for compatibility with existing commands and Hermes scheduled jobs.

## Training

- `train.py`: Stable-Baselines3 PPO training entry point.
- `start_training.sh`: managed background training launcher.
- `restart_from_checkpoint.sh`: pause and resume from the latest checkpoint.
- `pause_training.py`: pause the managed training process.
- `find_latest_checkpoint.py`: locate the newest resumable checkpoint.
- `watch_robot.py`: render the latest MuJoCo checkpoint.

## Feedback Flywheel

- `monitor_hermes.py`: L1/L2/L3 scan and Hermes/Feishu message formatter.
- `run_monitor_for_hermes.sh`: scheduled Hermes entry point.
- `l2_check.py`: create pending L2 proposals and confirmation tokens.
- `l2_decide.py`: apply or reject an L2 proposal.
- `l2_confirm.sh`: confirm an L2 token and restart if needed.
- `l2_reject.sh`: reject an L2 token.
- `l3_check.py`: trigger emergency pause on L3 rules.
- `auto_restart_if_needed.py`: restart training after L1 or confirmed L2 changes.

## Maintenance

- `lint_config.py`: mechanical config validation.
- `entropy_scan.py`: checkpoint entropy report generator.
- `run_entropy_scan.sh`: scheduled entropy scan wrapper.
- `git_auto_commit.py`: commit allowed code/config/adjustment changes.

## MJLab

- `start_mjlab_training.sh`: launch MJLab Go1 training.
- `play_mjlab.sh`: play the MJLab task.

Future cleanup can move these into subfolders once compatibility wrappers are added.
