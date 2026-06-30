#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(os.environ.get("HARNESS_ROOT", Path(__file__).resolve().parents[2]))
QUEUE_DIR = ROOT / "runs" / "training_queue"
QUEUE_PATH = QUEUE_DIR / "queue.json"
EVENTS_PATH = QUEUE_DIR / "events.jsonl"
ACTIVE_PATH = ROOT / "runs" / "active_training.json"
MJLAB_CONFIG_PATH = ROOT / "configs" / "tasks" / "mjlab" / "go1.yaml"
MJLAB_FEEDBACK_PATH = ROOT / "configs" / "tasks" / "mjlab" / "feedback.yaml"


ENV_RE = re.compile(r"(?<![A-Za-z])(\d+)\s*(?:并行(?:数)?|envs?|parallel|num[_-]?envs?)", re.I)
ITER_RE = re.compile(r"(?<![A-Za-z])(\d+)\s*(?:次|轮|iterations?|iters?)", re.I)
ENV_AFTER_RE = re.compile(r"(?:并行(?:数)?|envs?|parallel|num[_-]?envs?)\s*[:=：]?\s*(\d+)", re.I)
ITER_AFTER_RE = re.compile(r"(?:训练次数|次数|迭代(?:次数)?|iterations?|iters?)\s*[:=：]?\s*(\d+)", re.I)
HOUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|h|hrs?|hours?)", re.I)
MINUTE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:分钟|mins?|minutes?)", re.I)
GLOBAL_SWITCH_RE = re.compile(r"每\s*(\d+(?:\.\d+)?)\s*(?:个)?\s*(小时|h|hrs?|hours?|分钟|mins?|minutes?)", re.I)
RUN_ID_RE = re.compile(r"(?:run[_-]?id|运行名|名称)\s*[:=：]\s*([A-Za-z0-9_.-]+)", re.I)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_ts() -> float:
    return time.time()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queue() -> dict:
    queue = load_json(QUEUE_PATH, {"version": 1, "jobs": []})
    queue.setdefault("version", 1)
    queue.setdefault("jobs", [])
    return queue


def save_queue(queue: dict) -> None:
    save_json(QUEUE_PATH, queue)


def append_event(job: dict | None, event: str, reason: str = "", extra: dict | None = None) -> None:
    row = {
        "timestamp": now_ts(),
        "time": now_iso(),
        "event": event,
        "reason": reason,
    }
    if job:
        row.update({
            "job_id": job.get("id"),
            "run_id": job.get("run_id"),
            "status": job.get("status"),
            "num_envs": job.get("num_envs"),
            "max_iterations": job.get("max_iterations"),
            "max_runtime_minutes": job.get("max_runtime_minutes"),
        })
    if extra:
        row.update(extra)
    append_jsonl(EVENTS_PATH, row)


def load_defaults() -> dict:
    cfg = load_yaml(MJLAB_CONFIG_PATH)
    mjlab = cfg.get("mjlab", {})
    return {
        "task": mjlab.get("task", "Mjlab-Velocity-Flat-Unitree-G1"),
        "project_dir": mjlab.get("project_dir", "/home/leju/mjlab/src"),
        "num_envs": int(mjlab.get("num_envs", 4096)),
        "max_iterations": int(mjlab.get("max_iterations", 5000)),
        "wandb_project": mjlab.get("wandb_project", "rl-harness-mjlab"),
        "wandb_entity": mjlab.get("wandb_entity"),
        "learning_rate": cfg.get("agent", {}).get("learning_rate"),
        "reward_weights": cfg.get("reward_weights", {}),
    }


def clean_request_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*(?:[-*•]|\d+[.、)])\s*", "", line)
    return line.strip()


def duration_minutes_from_match(value: str, unit: str) -> int:
    minutes = float(value)
    if unit.lower() in {"h", "hr", "hrs", "hour", "hours"} or "小时" in unit:
        minutes *= 60
    return max(1, int(round(minutes)))


def parse_duration_minutes(text: str) -> int | None:
    hour = HOUR_RE.search(text)
    if hour:
        return duration_minutes_from_match(hour.group(1), hour.group(0))

    minute = MINUTE_RE.search(text)
    if minute:
        return duration_minutes_from_match(minute.group(1), minute.group(0))

    return None


def parse_global_runtime(text: str) -> int | None:
    match = GLOBAL_SWITCH_RE.search(text)
    if not match:
        return None
    return duration_minutes_from_match(match.group(1), match.group(2))


def make_job_id(index: int) -> str:
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:02d}_{secrets.token_hex(2)}"


def make_run_id(num_envs: int, max_iterations: int, index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mjlab_g1_{num_envs}_{max_iterations}_{stamp}_{index:02d}"


def parse_requests(text: str, source: str = "hermes") -> list[dict]:
    defaults = load_defaults()
    global_runtime = parse_global_runtime(text)
    raw_lines = re.split(r"[\n;；]+", text)
    jobs = []

    for raw in raw_lines:
        line = clean_request_line(raw)
        if not line:
            continue
        if not re.search(r"(训练|G1|MJLab|并行|env|iteration|次|小时|分钟)", line, re.I):
            continue

        env_match = ENV_RE.search(line) or ENV_AFTER_RE.search(line)
        iter_match = ITER_RE.search(line) or ITER_AFTER_RE.search(line)
        run_id_match = RUN_ID_RE.search(line)
        has_job_anchor = bool(env_match or iter_match or re.search(r"\bG1\b|MJLab", line, re.I))

        if not has_job_anchor:
            continue

        num_envs = int(env_match.group(1)) if env_match else defaults["num_envs"]
        max_iterations = int(iter_match.group(1)) if iter_match else defaults["max_iterations"]
        runtime_minutes = parse_duration_minutes(line) or global_runtime

        if runtime_minutes is None and not env_match and not iter_match:
            continue

        index = len(jobs) + 1
        run_id = run_id_match.group(1) if run_id_match else make_run_id(num_envs, max_iterations, index)
        job = {
            "id": make_job_id(index),
            "status": "queued",
            "created_at": now_iso(),
            "created_at_ts": now_ts(),
            "source": source,
            "source_text": line,
            "task": defaults["task"],
            "project_dir": defaults["project_dir"],
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "max_runtime_minutes": runtime_minutes,
            "run_id": run_id,
            "wandb_project": defaults["wandb_project"],
            "wandb_entity": defaults["wandb_entity"],
            "auto_advance": True,
        }
        jobs.append(job)

    return jobs


def enqueue_text(text: str, source: str = "hermes", dry_run: bool = False) -> list[dict]:
    jobs = parse_requests(text, source=source)
    if dry_run:
        return jobs

    queue = load_queue()
    queue["jobs"].extend(jobs)
    save_queue(queue)

    for job in jobs:
        append_event(job, "enqueued", "created from Hermes text")

    return jobs


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def load_active() -> dict | None:
    return load_json(ACTIVE_PATH, None)


def active_process_running(active: dict | None) -> bool:
    if not active or active.get("status") != "running":
        return False
    pid = active.get("pid")
    return bool(pid and pid_alive(pid))


def command_for_job(job: dict, cfg: dict) -> list[str]:
    mjlab = cfg.get("mjlab", {})
    agent = cfg.get("agent", {})
    reward_weights = cfg.get("reward_weights", {})

    cmd = [
        "uv", "run", "train", str(job["task"]),
        "--env.scene.num-envs", str(job["num_envs"]),
        "--agent.max-iterations", str(job["max_iterations"]),
        "--agent.logger", "wandb",
        "--agent.wandb-project", str(job.get("wandb_project") or mjlab.get("wandb_project", "rl-harness-mjlab")),
        "--agent.run-name", str(job["run_id"]),
    ]

    if agent.get("learning_rate") is not None:
        cmd += ["--agent.algorithm.learning-rate", str(agent["learning_rate"])]

    for name, value in reward_weights.items():
        cmd += [f"--env.rewards.{name.replace('_', '-')}.weight", str(value)]

    return cmd


def sync_active_configs_for_job(job: dict) -> dict:
    cfg = load_yaml(MJLAB_CONFIG_PATH)
    cfg.setdefault("mjlab", {})
    cfg["mjlab"]["run_id"] = job["run_id"]
    cfg["mjlab"]["project_dir"] = job["project_dir"]
    cfg["mjlab"]["task"] = job["task"]
    cfg["mjlab"]["num_envs"] = int(job["num_envs"])
    cfg["mjlab"]["max_iterations"] = int(job["max_iterations"])
    cfg["mjlab"]["wandb_project"] = job.get("wandb_project") or cfg["mjlab"].get("wandb_project", "rl-harness-mjlab")
    if job.get("wandb_entity"):
        cfg["mjlab"]["wandb_entity"] = job["wandb_entity"]
    cfg["mjlab"]["wandb_name"] = job["run_id"]
    save_yaml(MJLAB_CONFIG_PATH, cfg)

    feedback = load_yaml(MJLAB_FEEDBACK_PATH)
    profile = feedback.get("feedback_profile")
    if profile:
        profile.setdefault("project", {})
        profile.setdefault("paths", {})
        profile["project"]["run_id"] = job["run_id"]
        profile["paths"]["metric_log"] = f"runs/{job['run_id']}/train.jsonl"
        profile["paths"]["state_path_prefix"] = f"runs/_{job['run_id']}_feedback"
        save_yaml(MJLAB_FEEDBACK_PATH, feedback)

    return cfg


def launch_process(cmd: list[str], project_dir: str, log_path: Path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f"[{now_iso()}] START\n")
    log_file.write("cmd: " + " ".join(cmd) + "\n")
    log_file.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(project_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_job(job: dict, dry_run: bool = False) -> dict:
    cfg = sync_active_configs_for_job(job)
    run_dir = ROOT / "runs" / job["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training_process.log"
    cmd = command_for_job(job, cfg)

    if dry_run:
        job["status"] = "dry_run"
        job["command"] = " ".join(cmd)
        job["log"] = str(log_path)
        return job

    proc = launch_process(cmd, job["project_dir"], log_path)

    job.update({
        "status": "running",
        "started_at": now_iso(),
        "started_at_ts": now_ts(),
        "pid": proc.pid,
        "command": " ".join(cmd),
        "log": str(log_path),
        "train_jsonl": str(run_dir / "train.jsonl"),
    })

    active = {
        "status": "running",
        "kind": "mjlab",
        "queue_managed": True,
        "queue_job_id": job["id"],
        "run_id": job["run_id"],
        "pid": proc.pid,
        "project_dir": job["project_dir"],
        "command": " ".join(cmd),
        "log": str(log_path),
        "started_at": job["started_at"],
        "max_iterations": job["max_iterations"],
        "max_runtime_minutes": job.get("max_runtime_minutes"),
    }
    save_json(ACTIVE_PATH, active)
    append_event(job, "started", "queue runner started job", {"pid": proc.pid})
    return job


def parser_module():
    path = ROOT / "scripts" / "mjlab" / "parse_mjlab_metrics.py"
    spec = importlib.util.spec_from_file_location("parse_mjlab_metrics_for_queue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sync_job_metrics(job: dict) -> dict | None:
    log_path = Path(job.get("log") or ROOT / "runs" / job["run_id"] / "training_process.log")
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    train_path = ROOT / "runs" / job["run_id"] / "train.jsonl"

    if log_path.exists():
        try:
            parser = parser_module()
            parser.sync_metrics(
                log_path=str(log_path),
                output_path=str(train_path),
                run_id=job["run_id"],
                task=job.get("task"),
            )
        except Exception as exc:
            append_event(job, "metric_sync_failed", str(exc))

    if not train_path.exists():
        return None

    last = None
    for line in train_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("mjlab/run_id") == job["run_id"] or row.get("source") == "wandb":
            last = row

    if last:
        job["last_metric_at"] = now_iso()
        if last.get("mjlab/iteration") is not None:
            job["last_iteration"] = int(last["mjlab/iteration"])
        if last.get("train/step") is not None:
            job["last_step"] = int(last["train/step"])
        if last.get("train/episode_reward_mean") is not None:
            job["last_reward"] = float(last["train/episode_reward_mean"])
    return last


def iteration_limit_reached(job: dict) -> bool:
    iteration = job.get("last_iteration")
    if iteration is None:
        return False
    return int(iteration) + 1 >= int(job["max_iterations"])


def runtime_limit_reached(job: dict) -> bool:
    limit = job.get("max_runtime_minutes")
    started = job.get("started_at_ts")
    if not limit or not started:
        return False
    return now_ts() - float(started) >= float(limit) * 60


def pause_active_training(reason: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "scripts/training/pause_training.py", "--reason", reason, "--force"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode, result.stdout + result.stderr


def complete_job(job: dict, reason: str, dry_run: bool = False) -> None:
    if dry_run:
        append_event(job, "would_stop", reason)
        return

    code, output = pause_active_training(reason)
    job.update({
        "status": "completed" if code == 0 else "stop_failed",
        "stopped_at": now_iso(),
        "stopped_at_ts": now_ts(),
        "stop_reason": reason,
        "pause_returncode": code,
    })
    active = load_active()
    if active and active.get("queue_job_id") == job.get("id"):
        active["status"] = "paused" if code == 0 else "pause_failed"
        active["stopped_at"] = job["stopped_at_ts"]
        active["stop_reason"] = reason
        save_json(ACTIVE_PATH, active)
    append_event(job, "stopped" if code == 0 else "stop_failed", reason, {"pause_output": output[-2000:]})


def first_job(queue: dict, status: str) -> dict | None:
    return next((job for job in queue["jobs"] if job.get("status") == status), None)


def running_job(queue: dict) -> dict | None:
    return first_job(queue, "running")


def tick(dry_run: bool = False, no_start: bool = False) -> list[str]:
    queue = load_queue()
    messages = []

    job = running_job(queue)
    active = load_active()

    if job:
        sync_job_metrics(job)
        active_running = active_process_running(active)

        if iteration_limit_reached(job):
            complete_job(job, "iteration_limit", dry_run=dry_run)
            messages.append(f"stopped {job['id']}: iteration_limit")
        elif runtime_limit_reached(job):
            complete_job(job, "runtime_limit", dry_run=dry_run)
            messages.append(f"stopped {job['id']}: runtime_limit")
        elif not active_running:
            reason = "process_exited"
            if iteration_limit_reached(job):
                reason = "iteration_limit"
            job.update({
                "status": "completed",
                "stopped_at": now_iso(),
                "stopped_at_ts": now_ts(),
                "stop_reason": reason,
            })
            append_event(job, "stopped", reason)
            messages.append(f"completed {job['id']}: {reason}")
        else:
            messages.append(f"running {job['id']}: {job.get('run_id')}")

    active = load_active()
    if not active_process_running(active):
        pending = first_job(queue, "queued")
        if pending and not no_start:
            if dry_run:
                messages.append(f"would start {pending['id']}: {pending['run_id']}")
            else:
                start_job(pending)
                messages.append(f"started {pending['id']}: {pending['run_id']}")
    elif not job:
        messages.append(f"external active training is running: {active.get('run_id') if active else 'unknown'}")

    if not dry_run:
        save_queue(queue)

    if not messages:
        messages.append("queue idle")
    return messages


def cancel_job(job_id: str, reason: str) -> bool:
    queue = load_queue()
    for job in queue["jobs"]:
        if job.get("id") == job_id:
            if job.get("status") == "running":
                complete_job(job, reason)
            else:
                job["status"] = "cancelled"
                job["cancelled_at"] = now_iso()
                job["stop_reason"] = reason
                append_event(job, "cancelled", reason)
            save_queue(queue)
            return True
    return False


def clear_completed() -> int:
    queue = load_queue()
    before = len(queue["jobs"])
    queue["jobs"] = [
        job for job in queue["jobs"]
        if job.get("status") not in {"completed", "cancelled", "failed", "stop_failed"}
    ]
    removed = before - len(queue["jobs"])
    save_queue(queue)
    append_event(None, "clear_completed", f"removed {removed} jobs")
    return removed


def print_jobs(jobs: list[dict]) -> None:
    if not jobs:
        print("没有训练任务。")
        return
    for job in jobs:
        print(
            f"{job['id']} [{job['status']}] run_id={job['run_id']} "
            f"envs={job['num_envs']} iterations={job['max_iterations']} "
            f"runtime={job.get('max_runtime_minutes') or 'none'}min"
        )


def cmd_enqueue(args) -> None:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise SystemExit("请通过 --text 或 --file 提供训练要求。")

    jobs = enqueue_text(text, source=args.source, dry_run=args.dry_run)
    if args.json:
        print(json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2))
    else:
        print(f"已解析 {len(jobs)} 个训练任务。")
        print_jobs(jobs)


def cmd_status(args) -> None:
    queue = load_queue()
    active = load_active()
    payload = {"queue": queue, "active_training": active}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print_jobs(queue["jobs"])
    if active:
        print("")
        print(f"active: status={active.get('status')} run_id={active.get('run_id')} pid={active.get('pid')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes/Feishu MJLab training queue.")
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--text")
    enqueue.add_argument("--file")
    enqueue.add_argument("--source", default="hermes")
    enqueue.add_argument("--dry-run", action="store_true")
    enqueue.add_argument("--json", action="store_true")
    enqueue.set_defaults(func=cmd_enqueue)

    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    tick_cmd = sub.add_parser("tick")
    tick_cmd.add_argument("--dry-run", action="store_true")
    tick_cmd.add_argument("--no-start", action="store_true")
    tick_cmd.add_argument("--quiet", action="store_true")
    tick_cmd.set_defaults(func=cmd_tick)

    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    cancel.add_argument("--reason", default="manual_cancel")
    cancel.set_defaults(func=lambda args: print("cancelled" if cancel_job(args.job_id, args.reason) else "job not found"))

    clear = sub.add_parser("clear-completed")
    clear.set_defaults(func=lambda args: print(f"removed {clear_completed()} jobs"))

    return parser


def cmd_tick(args) -> None:
    messages = tick(dry_run=args.dry_run, no_start=args.no_start)
    if args.quiet and messages == ["queue idle"]:
        return
    print("\n".join(messages))


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
