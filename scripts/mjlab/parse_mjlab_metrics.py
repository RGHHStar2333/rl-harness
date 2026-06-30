#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
START_RE = re.compile(r"\[([^\]]+)\]\s+START")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

AGGREGATE_METRICS = {
    "Total steps": ("train/step", int),
    "Steps per second": ("mjlab/steps_per_second", float),
    "Collection time": ("mjlab/collection_time_seconds", float),
    "Learning time": ("mjlab/learning_time_seconds", float),
    "Mean value loss": ("train/value_loss", float),
    "Mean surrogate loss": ("train/surrogate_loss", float),
    "Mean entropy loss": ("train/entropy_loss", float),
    "Mean reward": ("train/episode_reward_mean", float),
    "Mean episode length": ("train/episode_length_mean", float),
    "Mean action std": ("train/action_std_mean", float),
}

PREFIX_METRICS = {
    "Episode_Reward/": "mjlab/reward/",
    "Episode_Metrics/": "mjlab/episode_metric/",
    "Episode_Termination/": "mjlab/termination/",
    "Metrics/": "mjlab/metric/",
    "Curriculum/": "mjlab/curriculum/",
}


@dataclass
class ResolvedPaths:
    run_id: str
    task: str | None
    log_path: str
    output_path: str


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def parse_number(text: str, caster=float):
    match = NUMBER_RE.search(text)
    if not match:
        return None

    value = float(match.group(0))
    if caster is int:
        return int(value)
    return value


def load_yaml(path: str) -> dict:
    full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_paths(config_path: str, log_path: str | None, output_path: str | None) -> ResolvedPaths:
    cfg = load_yaml(config_path)
    mjlab_cfg = cfg.get("mjlab", {})

    run_id = str(mjlab_cfg.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"MJLab config missing mjlab.run_id: {config_path}")

    task = mjlab_cfg.get("task")
    default_run_dir = os.path.join(ROOT, "runs", run_id)

    return ResolvedPaths(
        run_id=run_id,
        task=str(task) if task is not None else None,
        log_path=os.path.abspath(log_path or os.path.join(default_run_dir, "training_process.log")),
        output_path=os.path.abspath(output_path or os.path.join(default_run_dir, "train.jsonl")),
    )


def finalize_block(block: dict | None, run_id: str, task: str | None, log_path: str) -> dict | None:
    if not block:
        return None

    required = ["mjlab/iteration", "mjlab/max_iterations", "train/step", "train/episode_reward_mean"]
    if any(block.get(key) is None for key in required):
        return None

    record = {
        "timestamp": time.time(),
        "source": "mjlab_training_process_log",
        "mjlab/run_id": run_id,
        "mjlab/task": task,
        "mjlab/log_path": log_path,
    }
    record.update(block)
    return record


def update_block_from_line(block: dict, line: str) -> None:
    if ":" not in line:
        return

    key, raw_value = line.split(":", 1)
    key = key.strip()
    raw_value = raw_value.strip()

    if key == "Run name":
        block["mjlab/run_name"] = raw_value
        return

    metric = AGGREGATE_METRICS.get(key)
    if metric:
        out_key, caster = metric
        value = parse_number(raw_value, caster)
        if value is not None:
            block[out_key] = value
        return

    for prefix, out_prefix in PREFIX_METRICS.items():
        if key.startswith(prefix):
            metric_name = key[len(prefix):].strip()
            value = parse_number(raw_value, float)
            if metric_name and value is not None:
                block[f"{out_prefix}{metric_name}"] = value
            return


def parse_mjlab_log_lines(lines: Iterable[str], run_id: str, task: str | None, log_path: str) -> list[dict]:
    records = []
    current = None
    session_index = -1
    session_started_at = None
    last_iteration = None

    def flush_current() -> None:
        nonlocal current
        record = finalize_block(current, run_id, task, log_path)
        if record:
            records.append(record)
        current = None

    for raw_line in lines:
        line = clean_line(raw_line)
        if not line:
            continue

        start_match = START_RE.search(line)
        if start_match:
            flush_current()
            session_index += 1
            session_started_at = start_match.group(1)
            last_iteration = None
            continue

        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            iteration = int(iteration_match.group(1))
            max_iterations = int(iteration_match.group(2))

            flush_current()

            if session_index < 0:
                session_index = 0
            elif last_iteration is not None and iteration <= last_iteration:
                session_index += 1
                session_started_at = None

            current = {
                "mjlab/session_index": session_index,
                "mjlab/iteration": iteration,
                "mjlab/max_iterations": max_iterations,
            }
            if session_started_at is not None:
                current["mjlab/session_started_at"] = session_started_at

            last_iteration = iteration
            continue

        if current is not None:
            update_block_from_line(current, line)

    flush_current()

    return records


def int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_existing_keys(output_path: str) -> tuple[int, set[tuple[int, int, int]], set[tuple[int | None, int, int]]]:
    row_count = 0
    session_keys = set()
    legacy_keys = set()

    if not os.path.exists(output_path):
        return row_count, session_keys, legacy_keys

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_count += 1
            iteration = int_or_none(row.get("mjlab/iteration"))
            step = int_or_none(row.get("train/step"))
            if iteration is None or step is None:
                continue

            session_index = int_or_none(row.get("mjlab/session_index"))
            max_iterations = int_or_none(row.get("mjlab/max_iterations"))

            if session_index is None:
                legacy_keys.add((max_iterations, iteration, step))
            else:
                session_keys.add((session_index, iteration, step))

    return row_count, session_keys, legacy_keys


def filter_new_records(records: list[dict], output_path: str) -> list[dict]:
    existing_count, existing_session_keys, existing_legacy_keys = load_existing_keys(output_path)
    parsed_sessions = {
        int_or_none(record.get("mjlab/session_index"))
        for record in records
        if int_or_none(record.get("mjlab/session_index")) is not None
    }
    latest_session = max(parsed_sessions) if parsed_sessions else None
    only_latest_session = existing_count > 0 and len(parsed_sessions) > 1
    new_records = []

    for record in records:
        session_index = int_or_none(record.get("mjlab/session_index"))
        iteration = int_or_none(record.get("mjlab/iteration"))
        step = int_or_none(record.get("train/step"))
        max_iterations = int_or_none(record.get("mjlab/max_iterations"))

        if iteration is None or step is None:
            continue

        if only_latest_session and session_index != latest_session:
            continue

        session_key = (session_index, iteration, step) if session_index is not None else None
        legacy_key = (max_iterations, iteration, step)

        if session_key is not None and session_key in existing_session_keys:
            continue

        if legacy_key in existing_legacy_keys and (session_index is None or not only_latest_session):
            continue

        new_records.append(record)
        if session_key is not None:
            existing_session_keys.add(session_key)
        else:
            existing_legacy_keys.add(legacy_key)

    return new_records


def sync_metrics(
    log_path: str,
    output_path: str,
    run_id: str,
    task: str | None = None,
    dry_run: bool = False,
) -> dict:
    if not os.path.exists(log_path):
        return {
            "ok": True,
            "log_path": log_path,
            "output_path": output_path,
            "parsed": 0,
            "appended": 0,
            "reason": "log_missing",
        }

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        records = parse_mjlab_log_lines(f, run_id=run_id, task=task, log_path=log_path)

    new_records = filter_new_records(records, output_path)

    if not dry_run and new_records:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            for record in new_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "log_path": log_path,
        "output_path": output_path,
        "parsed": len(records),
        "appended": len(new_records),
        "dry_run": dry_run,
        "last_iteration": records[-1].get("mjlab/iteration") if records else None,
        "last_reward": records[-1].get("train/episode_reward_mean") if records else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse MJLab training logs into Harness train.jsonl.")
    parser.add_argument("--config", default="configs/tasks/mjlab/go1.yaml")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    paths = resolve_paths(args.config, args.log_path, args.output_path)
    summary = sync_metrics(
        log_path=paths.log_path,
        output_path=paths.output_path,
        run_id=paths.run_id,
        task=paths.task,
        dry_run=args.dry_run,
    )

    if summary.get("reason") == "log_missing":
        print(f"⚠️ MJLab log not found, skipped: {summary['log_path']}")
        return

    mode = "dry-run" if args.dry_run else "sync"
    print(
        "✅ MJLab metrics "
        f"{mode}: parsed={summary['parsed']} appended={summary['appended']} "
        f"last_iteration={summary['last_iteration']} last_reward={summary['last_reward']} "
        f"output={summary['output_path']}"
    )


if __name__ == "__main__":
    main()
