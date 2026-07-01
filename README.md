# RL Harness

This repository is the system of record for reinforcement-learning experiments. It keeps the durable parts of the system in Git: configuration, governance rules, training scripts, feedback automation, and operational entry points.

Runtime state stays on disk under `runs/`, `checkpoints/`, and `reports/`.

## Active Setup

- Active task: `HalfCheetah-v5` / MuJoCo
- Top-level config: `configs/pipeline.yaml`
- Hyperparameters: `configs/tasks/halfcheetah/hyper.yaml`
- Reward and environment kwargs: `configs/tasks/halfcheetah/reward.yaml`
- Notification mode: Hermes / Feishu via `scripts/run_monitor_for_hermes.sh`

## Main Workflows

Validate configuration:

```bash
python scripts/lint_config.py
```

Start managed training:

```bash
bash scripts/start_training.sh
```

Run one feedback pass for Hermes / Feishu:

```bash
bash scripts/run_monitor_for_hermes.sh
```

Start the Hermes/Feishu training request webhook:

```bash
bash scripts/start_hermes_feishu_webhook.sh
```

Run entropy scan:

```bash
bash scripts/run_entropy_scan.sh
```

Pause managed training:

```bash
python scripts/pause_training.py --force --reason "manual pause"
```

Restart from latest checkpoint:

```bash
bash scripts/restart_from_checkpoint.sh
```

## Directory Guide

- `configs/`: versioned training, task, reward, and feedback rules.
- `scripts/`: compatibility wrappers plus training, feedback, ops, and MJLab implementation folders.
- `runs/`: local runtime state, logs, adjustment history, and Hermes state.
- `checkpoints/`: local model checkpoint artifacts.
- `reports/`: generated reports such as entropy scan output.

See `AGENTS.md` for the compact project map used by agents.
