# Configs

Configuration is split into layers so the top-level pipeline stays readable.

## Files

- `pipeline.yaml`: active experiment entry point and governance settings.
- `detection_rules.yaml`: L1/L2/L3 feedback flywheel rules.
- `reward_skills.yaml`: reusable adjustment skill library.
- `tasks/cartpole/`: preserved CartPole baseline configs.
- `tasks/halfcheetah/`: active MuJoCo HalfCheetah configs.
- `tasks/mjlab/`: MJLab / Go1 launch config.

## Change Rule

Every config change should be validated before training:

```bash
python scripts/lint_config.py
```

Config changes are Git memory. Commit them separately from runtime artifacts.
