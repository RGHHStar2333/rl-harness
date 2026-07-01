# Hermes Feishu E2E Validation - 2026-07-01

## Result

PASS

## Covered Flow

- Feishu-style JSON message payload
- Training text extraction
- Two MJLab jobs enqueued in order
- First job started by queue tick
- Runtime limit stopped first job
- Second job started automatically

## Commands

```bash
python3 scripts/validate_hermes_feishu_e2e.py
```

## Tick Output

```text
started job_20260701_095651_01_dfe7: mjlab_g1_128_10_20260701_095651_01
stopped job_20260701_095651_01_dfe7: runtime_limit
started job_20260701_095651_02_202b: mjlab_g1_256_20_20260701_095651_02
```
