#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "scripts" / "training_queue" / "hermes_queue.py"
DELIVERY_LOG_PATH = ROOT / "runs" / "hermes_feishu_inbox.jsonl"


def load_queue_module():
    spec = importlib.util.spec_from_file_location("hermes_queue_webhook", QUEUE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_json_content(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"text": value}
    return parsed if isinstance(parsed, dict) else {"text": str(parsed)}


def extract_text(payload):
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""

    for key in ("text", "message_text"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            return payload[key].strip()

    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if isinstance(message, dict):
        content = parse_json_content(message.get("content", message))
        if isinstance(content.get("text"), str):
            return content["text"].strip()

    event = payload.get("event")
    if isinstance(event, dict):
        event_message = event.get("message", {})
        if isinstance(event_message, dict):
            content = parse_json_content(event_message.get("content", event_message))
            if isinstance(content.get("text"), str):
                return content["text"].strip()
            if isinstance(event_message.get("text"), str):
                return event_message["text"].strip()

        for key in ("text", "message"):
            if isinstance(event.get(key), str) and event[key].strip():
                return event[key].strip()

    body = payload.get("body")
    if isinstance(body, dict):
        return extract_text(body)

    return ""


def verify_payload_token(payload, expected_token):
    if not expected_token:
        return True
    if not isinstance(payload, dict):
        return False

    candidates = [
        payload.get("token"),
        payload.get("verify_token"),
    ]
    header = payload.get("header")
    if isinstance(header, dict):
        candidates += [header.get("token"), header.get("verify_token")]
    event = payload.get("event")
    if isinstance(event, dict):
        candidates += [event.get("token"), event.get("verify_token")]

    return expected_token in {str(item) for item in candidates if item is not None}


def summarize_jobs(jobs):
    return [
        {
            "id": job.get("id"),
            "run_id": job.get("run_id"),
            "status": job.get("status"),
            "num_envs": job.get("num_envs"),
            "max_iterations": job.get("max_iterations"),
            "max_runtime_minutes": job.get("max_runtime_minutes"),
        }
        for job in jobs
    ]


def append_delivery_log(payload, response, status):
    DELIVERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.time(),
        "status_code": status,
        "ok": response.get("ok"),
        "accepted": response.get("accepted", 0),
        "error": response.get("error"),
        "text": response.get("text"),
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }
    with DELIVERY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def handle_payload(payload, queue=None, source="feishu_webhook", dry_run=False, auto_tick=False, expected_token=None):
    if not verify_payload_token(payload, expected_token):
        return {"ok": False, "error": "invalid_token"}, 403

    if isinstance(payload, dict) and payload.get("challenge"):
        return {"ok": True, "challenge": payload["challenge"]}, 200

    text = extract_text(payload)
    if not text:
        return {"ok": False, "error": "no_training_text_found"}, 400

    queue = queue or load_queue_module()
    jobs = queue.enqueue_text(text, source=source, dry_run=dry_run)
    response = {
        "ok": True,
        "accepted": len(jobs),
        "dry_run": dry_run,
        "source": source,
        "text": text,
        "jobs": summarize_jobs(jobs),
    }
    if auto_tick and not dry_run:
        response["tick"] = queue.tick()
    return response, 200


def queue_status(queue=None):
    queue = queue or load_queue_module()
    return {
        "queue": queue.load_queue(),
        "active_training": queue.load_active(),
    }


class FeishuWebhookHandler(BaseHTTPRequestHandler):
    queue = None
    dry_run = False
    auto_tick = False
    expected_token = None

    def write_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in {"/health", "/"}:
            self.write_json(200, {"ok": True, "service": "hermes_feishu_webhook"})
            return
        if self.path == "/status":
            self.write_json(200, queue_status(self.queue))
            return
        self.write_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            response = {"ok": False, "error": "invalid_json"}
            append_delivery_log({}, response, 400)
            self.write_json(400, response)
            return

        response, status = handle_payload(
            payload,
            queue=self.queue,
            dry_run=self.dry_run,
            auto_tick=self.auto_tick,
            expected_token=self.expected_token,
        )
        append_delivery_log(payload, response, status)
        self.write_json(status, response)


def build_parser():
    parser = argparse.ArgumentParser(description="HTTP webhook entry for Feishu/Hermes MJLab training requests.")
    parser.add_argument("--host", default=os.environ.get("HERMES_FEISHU_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HERMES_FEISHU_PORT", "8765")))
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("HERMES_FEISHU_DRY_RUN") == "1")
    parser.add_argument("--auto-tick", action="store_true", default=os.environ.get("HERMES_FEISHU_AUTO_TICK") == "1")
    parser.add_argument("--verify-token", default=os.environ.get("FEISHU_VERIFY_TOKEN"))
    return parser


def main():
    args = build_parser().parse_args()
    handler = FeishuWebhookHandler
    handler.queue = load_queue_module()
    handler.dry_run = args.dry_run
    handler.auto_tick = args.auto_tick
    handler.expected_token = args.verify_token

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Hermes Feishu webhook listening on http://{args.host}:{args.port}")
    print("POST Feishu events to /feishu, /webhook, or any POST path.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
