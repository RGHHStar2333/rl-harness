#!/usr/bin/env python3
import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time


ROOT = Path(os.environ.get("HARNESS_ROOT", Path(__file__).resolve().parents[1]))
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"
STATE_DIR = ROOT / "runs" / "hermes_assistant"
CONFIRM_PATH = STATE_DIR / "pending_confirmations.json"
WEBHOOK_STATE_PATH = ROOT / "runs" / "hermes_feishu_webhook.json"
INBOX_PATH = ROOT / "runs" / "hermes_feishu_inbox.jsonl"
EVENTS_PATH = ROOT / "runs" / "training_queue" / "events.jsonl"

TRAINING_INTENT_RE = re.compile(r"(G1|MJLab|训练|并行|env|iteration|迭代|小时|分钟|跑|启动|开始)", re.I)
START_INTENT_RE = re.compile(r"(启动|开始|开跑|跑|执行|马上|立即)")
STATUS_INTENT_RE = re.compile(r"(状态|进度|队列|现在|跑到|情况|status)", re.I)
DIAGNOSE_INTENT_RE = re.compile(r"(诊断|排查|没收到|没有收到|回调|webhook|飞书|Hermes)", re.I)
SHELL_LINE_RE = re.compile(
    r"^\s*(?:cd|python3?|bash|sh|sudo|rm|mv|cp|git|curl|wget|nohup|uv|source|conda)\b"
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def load_jsonl_tail(path: Path, limit: int = 10) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def load_queue_module():
    spec = importlib.util.spec_from_file_location("hermes_product_assistant_queue", QUEUE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    configure_queue_module(module)
    return module


def configure_queue_module(queue) -> None:
    queue.ROOT = ROOT
    queue.QUEUE_DIR = ROOT / "runs" / "training_queue"
    queue.QUEUE_PATH = queue.QUEUE_DIR / "queue.json"
    queue.EVENTS_PATH = queue.QUEUE_DIR / "events.jsonl"
    queue.ACTIVE_PATH = ROOT / "runs" / "active_training.json"
    queue.MJLAB_CONFIG_PATH = ROOT / "configs" / "tasks" / "mjlab" / "go1.yaml"
    queue.MJLAB_FEEDBACK_PATH = ROOT / "configs" / "tasks" / "mjlab" / "feedback.yaml"


def load_pending() -> dict:
    data = load_json(CONFIRM_PATH, {"version": 1, "items": {}})
    data.setdefault("version", 1)
    data.setdefault("items", {})
    return data


def save_pending(data: dict) -> None:
    save_json(CONFIRM_PATH, data)


def shell_like(text: str) -> bool:
    if "```" in text:
        return True
    return any(SHELL_LINE_RE.search(line) for line in text.splitlines())


def has_training_intent(text: str) -> bool:
    return bool(TRAINING_INTENT_RE.search(text))


def has_start_intent(text: str) -> bool:
    return bool(START_INTENT_RE.search(text))


def has_status_intent(text: str) -> bool:
    return bool(STATUS_INTENT_RE.search(text))


def has_diagnose_intent(text: str) -> bool:
    return bool(DIAGNOSE_INTENT_RE.search(text))


def has_job_parameters(text: str, queue) -> bool:
    return bool(
        queue.ENV_RE.search(text)
        or queue.ENV_AFTER_RE.search(text)
        or queue.ITER_RE.search(text)
        or queue.ITER_AFTER_RE.search(text)
        or queue.parse_duration_minutes(text)
    )


def format_minutes(minutes) -> str:
    if not minutes:
        return "未设置"
    minutes = int(minutes)
    if minutes % 60 == 0:
        return f"{minutes // 60} 小时"
    return f"{minutes} 分钟"


def summarize_job(job: dict, index: int | None = None) -> str:
    prefix = f"{index}. " if index is not None else "- "
    return (
        f"{prefix}G1，{job['num_envs']} 并行，{job['max_iterations']} 次，"
        f"最多 {format_minutes(job.get('max_runtime_minutes'))}"
    )


def missing_details(text: str, jobs: list[dict], queue) -> list[str]:
    if jobs:
        missing = set()
        for job in jobs:
            line = job.get("source_text", "")
            if not (queue.ENV_RE.search(line) or queue.ENV_AFTER_RE.search(line)):
                missing.add("并行数")
            if not (queue.ITER_RE.search(line) or queue.ITER_AFTER_RE.search(line)):
                missing.add("训练次数")
            if job.get("max_runtime_minutes") is None:
                missing.add("最长运行时间")
        return sorted(missing)

    if has_training_intent(text):
        return ["并行数", "训练次数", "最长运行时间"]
    return []


def find_duplicate_jobs(jobs: list[dict], queue_state: dict) -> list[dict]:
    duplicates = []
    existing = [
        job for job in queue_state.get("jobs", [])
        if job.get("status") in {"queued", "running"}
    ]
    for candidate in jobs:
        for job in existing:
            same_shape = (
                int(job.get("num_envs", -1)) == int(candidate.get("num_envs", -2))
                and int(job.get("max_iterations", -1)) == int(candidate.get("max_iterations", -2))
                and (job.get("max_runtime_minutes") or None) == (candidate.get("max_runtime_minutes") or None)
            )
            if same_shape:
                duplicates.append(job)
                break
    return duplicates


def plan_risks(jobs: list[dict], duplicates: list[dict]) -> list[str]:
    risks = []
    if any(int(job.get("num_envs", 0)) >= 4096 for job in jobs):
        risks.append("包含 4096 并行训练，资源占用较高。")
    if len(jobs) > 1:
        risks.append(f"这是 {len(jobs)} 个顺序任务，我会按队列一个个执行，不会同时跑。")
    if duplicates:
        risks.append(f"当前队列里已有 {len(duplicates)} 个形状相同的任务，可能是重复提交。")
    if any((job.get("max_runtime_minutes") or 0) >= 180 for job in jobs):
        risks.append("存在 3 小时或更长的任务，建议确认机器资源和散热。")
    return risks


def make_confirmation(text: str, jobs: list[dict], risks: list[str], start_after_confirm: bool) -> str:
    pending = load_pending()
    token = secrets.token_hex(4)
    pending["items"][token] = {
        "created_at": now_iso(),
        "created_at_ts": time.time(),
        "text": text,
        "jobs": jobs,
        "risks": risks,
        "start_after_confirm": start_after_confirm,
    }
    save_pending(pending)
    return token


def refuse_shell() -> str:
    return "\n".join([
        "我不能直接执行这段 shell 命令。",
        "",
        "为了训练安全，我只接受受控的 Harness 操作，例如：",
        "",
        "```bash",
        "python3 scripts/hermes_product_assistant.py ask --text 'G1 4096并行 8000次 1小时'",
        "python3 scripts/hermes_product_assistant.py status",
        "```",
        "",
        "你可以把训练目标用自然语言发给我，我会先解释计划，再让你确认。",
    ])


def explain_missing(missing: list[str]) -> str:
    return "\n".join([
        "我理解你想让我安排训练，但信息还不够完整。",
        "",
        "还缺这些信息：",
        *[f"- {item}" for item in missing],
        "",
        "你可以这样说：",
        "",
        "```text",
        "G1 4096并行 8000次 1小时",
        "```",
    ])


def explain_plan(text: str, jobs: list[dict], duplicates: list[dict], risks: list[str], token: str, start_after_confirm: bool) -> str:
    lines = []
    lines.append("我理解你想创建下面的 MJLab G1 训练计划：")
    lines.append("")
    for index, job in enumerate(jobs, start=1):
        lines.append(summarize_job(job, index))
    lines.append("")
    lines.append("执行方式：")
    lines.append("- 这些任务会按顺序进入训练队列。")
    lines.append("- 每个任务会在达到训练次数或时间上限时停止。")
    lines.append("- 任务结束后，队列会切到下一个任务。")

    if risks:
        lines.append("")
        lines.append("我注意到几个需要确认的点：")
        for risk in risks:
            lines.append(f"- {risk}")

    if duplicates:
        lines.append("")
        lines.append("可能重复的现有任务：")
        for job in duplicates[:5]:
            lines.append(
                f"- {job.get('run_id')} [{job.get('status')}] "
                f"{job.get('num_envs')}并行/{job.get('max_iterations')}次/{format_minutes(job.get('max_runtime_minutes'))}"
            )

    lines.append("")
    action = "入队并尝试启动第 1 个任务" if start_after_confirm else "只入队，等待下一轮 monitor 或你手动 tick 启动"
    lines.append(f"确认后我会：{action}。")
    lines.append("")
    lines.append("确认命令：")
    lines.append("")
    lines.append("```bash")
    start_flag = " --start" if start_after_confirm else ""
    lines.append(f"python3 scripts/hermes_product_assistant.py confirm {token}{start_flag}")
    lines.append("```")
    return "\n".join(lines)


def ask(text: str, json_mode: bool = False) -> str:
    if shell_like(text):
        response = {"action": "refuse", "message": refuse_shell()}
        return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else response["message"]

    queue = load_queue_module()

    if has_status_intent(text) and not has_job_parameters(text, queue):
        return status(json_mode=json_mode)

    if has_diagnose_intent(text) and not has_job_parameters(text, queue):
        return diagnose(json_mode=json_mode)

    jobs = queue.parse_requests(text, source="hermes_assistant")
    missing = missing_details(text, jobs, queue)
    if missing:
        response = {"action": "need_more_info", "missing": missing, "message": explain_missing(missing)}
        return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else response["message"]

    if not jobs:
        response = {
            "action": "unknown",
            "message": "我还没有理解你的训练意图。你可以说：G1 4096并行 8000次 1小时",
        }
        return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else response["message"]

    queue_state = queue.load_queue()
    duplicates = find_duplicate_jobs(jobs, queue_state)
    risks = plan_risks(jobs, duplicates)
    start_after_confirm = has_start_intent(text)
    token = make_confirmation(text, jobs, risks, start_after_confirm)
    message = explain_plan(text, jobs, duplicates, risks, token, start_after_confirm)
    response = {
        "action": "plan",
        "token": token,
        "start_after_confirm": start_after_confirm,
        "jobs": jobs,
        "risks": risks,
        "duplicate_count": len(duplicates),
        "message": message,
    }
    return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else message


def confirm(token: str, start: bool | None = None, json_mode: bool = False) -> str:
    pending = load_pending()
    item = pending["items"].get(token)
    if not item:
        message = "我没有找到这个确认 token。可能已经执行过，或者 token 写错了。"
        response = {"action": "confirm_missing", "token": token, "message": message}
        return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else message

    queue = load_queue_module()
    jobs = queue.enqueue_text(item["text"], source="hermes_assistant")
    start_after_confirm = item.get("start_after_confirm", False) if start is None else bool(start)
    tick_messages = []
    if start_after_confirm:
        tick_messages = queue.tick()

    del pending["items"][token]
    save_pending(pending)

    lines = []
    lines.append(f"已确认，{len(jobs)} 个训练任务已经进入队列。")
    lines.append("")
    for index, job in enumerate(jobs, start=1):
        lines.append(summarize_job(job, index))
    if tick_messages:
        lines.append("")
        lines.append("启动检查结果：")
        for message in tick_messages:
            lines.append(f"- {translate_tick_message(message)}")
    else:
        lines.append("")
        lines.append("我还没有立即启动训练；下一轮 monitor tick 会自动处理，或者你可以让我启动队列。")

    response = {
        "action": "confirmed",
        "token": token,
        "jobs": jobs,
        "tick": tick_messages,
        "message": "\n".join(lines),
    }
    return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else response["message"]


def translate_tick_message(message: str) -> str:
    if message.startswith("started "):
        return "已启动队首训练任务。"
    if message.startswith("running "):
        return "当前训练仍在运行。"
    if "runtime_limit" in message:
        return "有任务达到时间上限，已停止并准备切换。"
    if "iteration_limit" in message:
        return "有任务达到训练次数上限，已停止并准备切换。"
    if "external active training" in message:
        return "检测到外部训练正在运行，队列暂时不会抢占。"
    if message == "queue idle":
        return "队列现在空闲。"
    return message


def pid_command(pid) -> str:
    if not pid:
        return ""
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def queue_status_summary(queue, active) -> tuple[list[str], dict]:
    jobs = queue.get("jobs", [])
    counts = Counter(job.get("status", "unknown") for job in jobs)
    queued = [job for job in jobs if job.get("status") == "queued"]
    running = [job for job in jobs if job.get("status") == "running"]

    lines = []
    lines.append("训练状态概览：")
    lines.append("")
    if jobs:
        parts = [f"{status} {count} 个" for status, count in sorted(counts.items())]
        lines.append(f"- 队列里共有 {len(jobs)} 个任务：" + "，".join(parts))
    else:
        lines.append("- 队列现在是空的。")

    if queued:
        lines.append(f"- 下一个等待任务：{summarize_job(queued[0]).lstrip('- ')}。")
    if running:
        lines.append(f"- 队列认为正在运行：{running[0].get('run_id')}。")

    active_summary = {}
    if active:
        queue_module = load_queue_module()
        active_running = queue_module.active_process_running(active)
        active_summary = {
            "status": active.get("status"),
            "run_id": active.get("run_id"),
            "pid": active.get("pid"),
            "pid_alive": active_running,
            "command": pid_command(active.get("pid")),
        }
        if active.get("status") == "running" and active_running:
            lines.append(f"- 当前训练进程还在运行：{active.get('run_id')}，PID {active.get('pid')}。")
        elif active.get("status") == "running" and not active_running:
            lines.append("- active_training 里还写着 running，但 PID 已经不在了；这通常是训练已经结束或进程退出后的状态残留。")
            lines.append("- 建议先运行一次状态修正或 monitor tick，再启动下一个队列任务。")
        else:
            lines.append(f"- 当前 active_training 状态是 {active.get('status')}。")
    else:
        lines.append("- 没有 active_training 状态文件。")

    if not jobs and (not active or active.get("status") != "running"):
        lines.append("- 现在没有待执行训练，也没有正在运行的训练。")

    data = {"job_count": len(jobs), "counts": dict(counts), "active": active_summary}
    return lines, data


def status(json_mode: bool = False) -> str:
    queue = load_queue_module()
    queue_state = queue.load_queue()
    active = queue.load_active()
    lines, data = queue_status_summary(queue_state, active)
    response = {"action": "status", "summary": data, "message": "\n".join(lines)}
    return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else response["message"]


def diagnose(json_mode: bool = False, text: str | None = None) -> str:
    webhook_state = load_json(WEBHOOK_STATE_PATH, {})
    inbox = load_jsonl_tail(INBOX_PATH, limit=10)
    events = load_jsonl_tail(EVENTS_PATH, limit=10)
    training_deliveries = [row for row in inbox if row.get("text") or row.get("accepted") or row.get("error")]
    queue = load_queue_module()
    queue_state = queue.load_queue()

    lines = []
    lines.append("我帮你看了一下 Hermes/飞书到 Harness 的链路：")
    lines.append("")
    if webhook_state.get("status") == "running":
        lines.append(f"- webhook 显示正在运行，端口是 {webhook_state.get('port')}。")
    else:
        lines.append("- webhook 没有显示为 running，飞书消息可能进不来。")

    if not inbox:
        lines.append("- inbox 里没有任何 POST 记录，说明飞书/Hermes 还没有打到 Harness。")
    elif not training_deliveries:
        lines.append("- inbox 里只有 challenge/健康检查，还没有收到训练文本。")
    else:
        last = training_deliveries[-1]
        if last.get("accepted"):
            lines.append(f"- 最近一次训练文本投递成功，入队 {last.get('accepted')} 个任务。")
        elif last.get("error"):
            lines.append(f"- 最近一次投递失败：{last.get('error')}。")
        else:
            lines.append("- 最近收到过训练文本，但没有成功入队。")

    jobs = queue_state.get("jobs", [])
    lines.append(f"- 当前队列有 {len(jobs)} 个任务。")
    if events:
        last_event = events[-1]
        lines.append(f"- 最近队列事件：{last_event.get('event')}，原因：{last_event.get('reason') or '未写明'}。")

    parsed_jobs = []
    if text:
        parsed_jobs = queue.parse_requests(text, source="hermes_assistant_diagnose")
        lines.append("")
        lines.append(f"你给的文本可以解析出 {len(parsed_jobs)} 个训练任务。")
        if parsed_jobs and not jobs and not training_deliveries:
            lines.append("这说明训练文本本身没问题，问题更可能在飞书/Hermes 没有把消息投递过来。")

    response = {
        "action": "diagnose",
        "webhook": webhook_state,
        "inbox_recent": inbox,
        "queue_jobs": len(jobs),
        "parsed_jobs": parsed_jobs,
        "message": "\n".join(lines),
    }
    return json.dumps(response, ensure_ascii=False, indent=2) if json_mode else response["message"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes product assistant for Harness training workflows.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_cmd = sub.add_parser("ask")
    ask_cmd.add_argument("--text", required=True)
    ask_cmd.add_argument("--json", action="store_true")
    ask_cmd.set_defaults(func=lambda args: print(ask(args.text, json_mode=args.json)))

    confirm_cmd = sub.add_parser("confirm")
    confirm_cmd.add_argument("token")
    start_group = confirm_cmd.add_mutually_exclusive_group()
    start_group.add_argument("--start", action="store_true")
    start_group.add_argument("--no-start", action="store_true")
    confirm_cmd.add_argument("--json", action="store_true")

    def run_confirm(args):
        start = None
        if args.start:
            start = True
        if args.no_start:
            start = False
        print(confirm(args.token, start=start, json_mode=args.json))

    confirm_cmd.set_defaults(func=run_confirm)

    status_cmd = sub.add_parser("status")
    status_cmd.add_argument("--json", action="store_true")
    status_cmd.set_defaults(func=lambda args: print(status(json_mode=args.json)))

    diagnose_cmd = sub.add_parser("diagnose")
    diagnose_cmd.add_argument("--text")
    diagnose_cmd.add_argument("--json", action="store_true")
    diagnose_cmd.set_defaults(func=lambda args: print(diagnose(json_mode=args.json, text=args.text)))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
