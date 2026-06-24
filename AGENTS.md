# RL Harness Project Map

This repository is the system of record for RL training.

## Entry points

- configs/pipeline.yaml: top-level training and governance config
- configs/tasks/cartpole/hyper.yaml: algorithm hyperparameters
- configs/tasks/cartpole/reward.yaml: reward config
- configs/detection_rules.yaml: feedback flywheel rules
- configs/reward_skills.yaml: adjustment skill library

## Commands

Validate configs:

```bash
python scripts/lint_config.py
```

Run training:

```bash
python scripts/train.py --config configs/pipeline.yaml
```

Run monitor:

```bash
python scripts/monitor.py --config configs/pipeline.yaml
```

Run entropy scan:

```bash
python scripts/entropy_scan.py --config configs/pipeline.yaml
```

## Rules

All configs must be versioned.
Training logs must be JSONL.
Checkpoints must be indexed.
Parameter changes must be recorded.
