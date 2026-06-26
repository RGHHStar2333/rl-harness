# Runs

This directory is local runtime state. Most contents are intentionally ignored by Git.

Typical files include:

- `training_process.log`: managed training stdout/stderr.
- `active_training.json`: current managed training process state.
- `_hermes_notify_state.json`: notification de-duplication state.
- `_l2_proposal_state.json`: L2 proposal de-duplication state.
- `_l3_pause_state.json`: L3 pause de-duplication state.
- `adjustments.jsonl`: accepted or automatic parameter adjustment history.
- `<run_id>/train.jsonl`: training metrics.

Runtime files are disk state. Durable rules and code belong in Git.
