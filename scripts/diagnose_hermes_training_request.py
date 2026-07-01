#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"
STATE_PATH = ROOT / "runs" / "hermes_feishu_webhook.json"
WEBHOOK_LOG = ROOT / "runs" / "hermes_feishu_webhook.log"
INBOX_LOG = ROOT / "runs" / "hermes_feishu_inbox.jsonl"
QUEUE_FILE = ROOT / "runs" / "training_queue" / "queue.json"
EVENTS_FILE = ROOT / "runs" / "training_queue" / "events.jsonl"
ACTIVE_FILE = ROOT / "runs" / "active_training.json"


def load_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def tail_lines(path, limit=8):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def parse_jsonl_lines(lines):
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def pid_command(pid):
    if not pid:
        return ""
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def is_webhook_pid(pid):
    return "scripts/hermes_feishu_webhook.py" in pid_command(pid)


def health_check(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return True, resp.read().decode("utf-8")
    except Exception as exc:
        return False, str(exc)


def load_queue_module():
    spec = importlib.util.spec_from_file_location("hermes_queue_diagnose", QUEUE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def print_section(title):
    print("")
    print(f"## {title}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Hermes/Feishu training request delivery.")
    parser.add_argument("--text", help="Dry-parse a training request without enqueueing it.")
    args = parser.parse_args()

    state = load_json(STATE_PATH, {})
    queue = load_json(QUEUE_FILE, {"jobs": []})
    active = load_json(ACTIVE_FILE, None)
    pid = state.get("pid")
    process_alive = pid_alive(pid)
    alive = process_alive and is_webhook_pid(pid)
    command = pid_command(pid)
    port = state.get("port") or int(os.environ.get("HERMES_FEISHU_PORT", "8765"))
    health_ok, health_detail = health_check(port) if alive else (False, "webhook process is not running")
    inbox_tail = tail_lines(INBOX_LOG)
    inbox_rows = parse_jsonl_lines(inbox_tail)
    training_deliveries = [
        row for row in inbox_rows
        if row.get("text") or row.get("accepted") or row.get("error")
    ]
    events_tail = tail_lines(EVENTS_FILE)

    print("# Hermes Training Request Diagnosis")

    print_section("Webhook")
    print(f"state: {state.get('status', 'missing')}")
    print(f"pid: {pid or 'none'}")
    print(f"pid_alive: {process_alive}")
    print(f"pid_is_webhook: {alive}")
    if command:
        print(f"pid_command: {command}")
    print(f"port: {port}")
    print(f"health_ok: {health_ok}")
    if not health_ok:
        print(f"health_detail: {health_detail}")

    print_section("Delivery")
    print(f"inbox_log_exists: {INBOX_LOG.exists()}")
    print(f"inbox_recent_rows: {len(inbox_tail)}")
    for line in inbox_tail:
        print(line)

    print_section("Queue")
    queue_jobs = queue.get("jobs", [])
    print(f"jobs: {len(queue_jobs)}")
    for job in queue_jobs[-8:]:
        print(
            f"- {job.get('id')} [{job.get('status')}] run_id={job.get('run_id')} "
            f"envs={job.get('num_envs')} iterations={job.get('max_iterations')} "
            f"runtime={job.get('max_runtime_minutes')}min"
        )

    print_section("Queue Events")
    print(f"events_log_exists: {EVENTS_FILE.exists()}")
    for line in events_tail:
        print(line)

    print_section("Active Training")
    if active:
        print(json.dumps({
            "status": active.get("status"),
            "run_id": active.get("run_id"),
            "pid": active.get("pid"),
            "queue_managed": active.get("queue_managed"),
            "stop_reason": active.get("stop_reason"),
        }, ensure_ascii=False, indent=2))
    else:
        print("none")

    if args.text:
        print_section("Dry Parse")
        queue_module = load_queue_module()
        parsed_jobs = queue_module.parse_requests(args.text, source="diagnose")
        print(json.dumps({"accepted": len(parsed_jobs), "jobs": parsed_jobs}, ensure_ascii=False, indent=2))

    print_section("Next Action")
    if not alive or not health_ok:
        print("Webhook 没有常驻。先执行：bash scripts/start_hermes_feishu_webhook.sh")
    elif not inbox_tail:
        print("Webhook 已启动，但没有收到飞书/Hermes POST。检查飞书回调 URL 或 Hermes agent 是否真的调用了本机 webhook/脚本。")
    elif not training_deliveries:
        print("Webhook 只收到过 challenge/健康类请求，还没有收到训练文本。重新发送训练消息，并确认飞书事件订阅或 Hermes agent 会 POST 到该 webhook。")
    elif not queue_jobs:
        print("Webhook 收到过请求，但没有入队。查看 inbox 中 error/text，确认消息格式和 verification token。")
    elif not any(job.get("status") == "running" for job in queue_jobs):
        print("已有队列但未运行。执行：python3 scripts/training_queue/hermes_queue.py tick")
    else:
        print("队列中已有 running 任务。查看 runs/<run_id>/training_process.log 和 train.jsonl。")


if __name__ == "__main__":
    main()
