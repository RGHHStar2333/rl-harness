# Hermes Product Assistant Completion - 2026-07-01

## Result

Harness now has a controlled Hermes product-assistant layer. It does not let the LLM execute arbitrary shell commands. Instead, Hermes can call one safe CLI that interprets training requests, explains the plan in plain Chinese, asks for confirmation, and then executes through the existing queue.

## New Entry Point

```bash
python3 scripts/hermes_product_assistant.py ask --text "帮我跑 G1 4096并行 8000次 1小时"
python3 scripts/hermes_product_assistant.py confirm <token> --start
python3 scripts/hermes_product_assistant.py status
python3 scripts/hermes_product_assistant.py diagnose --text "G1 4096并行 8000次 1小时"
```

## Conversation Behavior

- Natural training request -> interpreted plan.
- Missing env count / iterations / runtime -> asks for missing details.
- 4096 parallel or duplicate jobs -> explains risk and asks for confirmation.
- Confirmation -> enqueues through `training_queue/hermes_queue.py`.
- Optional `--start` -> runs queue tick to start the next job if safe.
- Status -> summarizes queue and active training in plain Chinese.
- Diagnose -> explains whether webhook delivery, inbox, queue, or active state is the likely issue.
- Shell-like requests -> refused with a safe Harness command path.

## Recommended Hermes Prompt

Ask Hermes to run:

```bash
cd /home/leju/桌面/rrr/Harness_RL/rl-harness
python3 scripts/hermes_product_assistant.py ask --text '<your request>'
```

Then confirm using the command it returns.

## Validation

```bash
python3 -m unittest tests.test_hermes_product_assistant
python3 -m py_compile scripts/hermes_product_assistant.py tests/test_hermes_product_assistant.py
```

Both passed during implementation.
