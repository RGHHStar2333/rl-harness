# RL Harness Project Map

This repository is the system of record for RL training.

## Entry points

- configs/pipeline.yaml: top-level training and governance config
- configs/tasks/halfcheetah/hyper.yaml: algorithm hyperparameters
- configs/tasks/halfcheetah/reward.yaml: reward and MuJoCo environment config
- configs/detection_rules.yaml: feedback flywheel rules
- configs/reward_skills.yaml: adjustment skill library
- scripts/monitor_hermes.py: Hermes/Feishu notification monitor

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
bash scripts/run_monitor_for_hermes.sh
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
