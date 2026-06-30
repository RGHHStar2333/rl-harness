#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/tasks/mjlab/go1.yaml"


def read_cfg_text():
    return CFG_PATH.read_text()


def simple_get(text, key, default=None):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip()
    return default


def section_get(text, section, key, default=None):
    lines = text.splitlines()
    inside = False
    for line in lines:
        if line.strip() == section + ":" and not line.startswith(" "):
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            inside = False
        if inside and line.startswith("  ") and line.strip().startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    return default


def parse_float(v, default):
    try:
        return float(str(v))
    except Exception:
        return default


def set_section_value(text, section, key, value):
    lines = text.splitlines()
    out = []
    inside = False
    changed = False
    saw_section = False

    for line in lines:
        if line.strip() == section + ":" and not line.startswith(" "):
            inside = True
            saw_section = True
            out.append(line)
            continue

        if inside and line and not line.startswith(" "):
            if not changed:
                out.append(f"  {key}: {value}")
                changed = True
            inside = False

        if inside and line.startswith("  ") and line.strip().startswith(key + ":"):
            out.append(f"  {key}: {value}")
            changed = True
        else:
            out.append(line)

    if saw_section and inside and not changed:
        out.append(f"  {key}: {value}")
        changed = True

    if not saw_section:
        out.append(f"{section}:")
        out.append(f"  {key}: {value}")

    return "\n".join(out) + "\n"


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def latest_decision_step(history_path):
    if not history_path.exists():
        return None
    last = None
    for line in history_path.read_text(errors="ignore").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("changed"):
            last = obj.get("step")
    return last


def discover_latest_local_wandb_run_id(run_id):
    log_path = ROOT / "runs" / str(run_id) / "training_process.log"
    if not log_path.exists():
        return None

    patterns = [
        re.compile(r"wandb\.ai/[^/]+/[^/]+/runs/([A-Za-z0-9_-]+)"),
        re.compile(r"/wandb/run-\d{8}_\d{6}-([A-Za-z0-9_-]+)"),
    ]
    found = []

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                found.append(match.group(1))

    return found[-1] if found else None


def find_latest_wandb_run(entity, project, run_name, wandb_run_id=None):
    import wandb

    api = wandb.Api()

    if wandb_run_id:
        try:
            return api.run(f"{entity}/{project}/{wandb_run_id}")
        except Exception as exc:
            print(f"⚠️ W&B run_id={wandb_run_id} 直连失败，回退到 run name 搜索：{exc}")

    runs = list(api.runs(f"{entity}/{project}"))
    matched = [r for r in runs if r.name == run_name or getattr(r, "display_name", None) == run_name]

    if not matched:
        if wandb_run_id:
            raise RuntimeError(f"没有在 W&B 找到 run_id = {wandb_run_id} 或 run name = {run_name}")
        raise RuntimeError(f"没有在 W&B 找到 run name = {run_name}")

    matched.sort(key=lambda r: getattr(r, "created_at", "") or "", reverse=True)
    return matched[0]


def collect_history(run):
    wanted = {
        "reward": [
            "Train/mean_reward",
            "train/mean_reward",
            "mean_reward",
            "Mean reward",
            "Episode/reward",
        ],
        "episode_length": [
            "Train/mean_episode_length",
            "train/mean_episode_length",
            "Mean episode length",
            "mean_episode_length",
        ],
        "track_linear_velocity": [
            "Episode_Reward/track_linear_velocity",
            "Episode Reward/track_linear_velocity",
        ],
        "track_angular_velocity": [
            "Episode_Reward/track_angular_velocity",
            "Episode Reward/track_angular_velocity",
        ],
        "upright": [
            "Episode_Reward/upright",
            "Episode Reward/upright",
        ],
        "pose": [
            "Episode_Reward/pose",
            "Episode Reward/pose",
        ],
        "error_vel_xy": [
            "Metrics/twist/error_vel_xy",
            "Metrics/error_vel_xy",
            "twist/error_vel_xy",
        ],
        "error_vel_yaw": [
            "Metrics/twist/error_vel_yaw",
            "Metrics/error_vel_yaw",
            "twist/error_vel_yaw",
        ],
        "fell_over": [
            "Episode_Termination/fell_over",
            "Episode Termination/fell_over",
            "fell_over",
        ],
    }

    rows = []
    max_rows = 20000

    for row in run.scan_history(page_size=1000):
        item = {}
        step = row.get("_step", None)
        if step is not None:
            item["_step"] = step

        for out_key, candidates in wanted.items():
            for k in candidates:
                v = row.get(k, None)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    item[out_key] = float(v)
                    break

        if len(item) > 1:
            rows.append(item)

        if len(rows) >= max_rows:
            rows = rows[-max_rows:]

    return rows


def load_history_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def mean_last(rows, key, n=20):
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    if not vals:
        return None
    vals = vals[-n:]
    return sum(vals) / len(vals)


def slope_last(rows, key, n=40):
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    if len(vals) < 5:
        return None
    vals = vals[-n:]
    mid = max(1, len(vals) // 2)
    a = sum(vals[:mid]) / len(vals[:mid])
    b = sum(vals[mid:]) / len(vals[mid:])
    return b - a


def latest_step(rows):
    steps = [r.get("_step") for r in rows if isinstance(r.get("_step"), (int, float))]
    return int(max(steps)) if steps else 0


def write_train_jsonl(run_dir, rows, decision):
    train_path = run_dir / "train.jsonl"
    if not rows:
        return

    existing_steps = set()
    if train_path.exists():
        for line in train_path.read_text(errors="ignore").splitlines()[-5000:]:
            try:
                obj = json.loads(line)
                if "step" in obj:
                    existing_steps.add(obj["step"])
            except Exception:
                pass

    new_count = 0
    with train_path.open("a") as f:
        for r in rows[-200:]:
            step = int(r.get("_step", 0))
            if step in existing_steps:
                continue
            obj = {
                "source": "wandb",
                "kind": "mjlab",
                "step": step,
                "time": datetime.now(timezone.utc).isoformat(),
            }
            for k, v in r.items():
                if k != "_step":
                    obj[k] = v
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            new_count += 1

    print(f"✅ W&B history 已写入 train.jsonl，新增 {new_count} 行")


def log_adjustments(adjustments_path, run_id, severity, step, actions, old_text, reasons):
    reason = "; ".join(reasons) if reasons else "W&B automatic decision"

    for sec, key, value in actions:
        old_raw = section_get(old_text, sec, key, None)
        old_value = parse_float(old_raw, old_raw)
        target = f"{sec}.{key}"
        append_jsonl(adjustments_path, {
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "run_id": run_id,
            "profile_name": "mjlab_wandb_autodecide",
            "trainer_kind": "mjlab",
            "source": "wandb",
            "level": severity,
            "decision": "auto_applied",
            "target": target,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
            "latest_step": step,
            "restart_required": True,
        })


def main():
    global CFG_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CFG_PATH))
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    parser.add_argument("--adjustments-path", default=str(ROOT / "runs" / "adjustments.jsonl"))
    parser.add_argument("--history-jsonl", default=None, help="离线测试用：从本地 JSONL 读取 W&B history 行")
    parser.add_argument("--restart-cmd", default="bash scripts/restart_mjlab_from_checkpoint.sh")
    parser.add_argument("--apply", action="store_true", help="真的修改配置并自动重启")
    parser.add_argument("--no-restart", action="store_true", help="只修改配置，不重启")
    args = parser.parse_args()

    CFG_PATH = Path(args.config)

    text = read_cfg_text()

    run_id = section_get(text, "mjlab", "run_id", simple_get(text, "run_id"))
    project = section_get(text, "mjlab", "wandb_project", simple_get(text, "wandb_project"))
    entity = section_get(text, "mjlab", "wandb_entity", os.environ.get("WANDB_ENTITY", "rghhstar-leju"))
    run_name = section_get(text, "mjlab", "wandb_name", simple_get(text, "wandb_name", run_id))
    configured_wandb_run_id = section_get(text, "mjlab", "wandb_run_id", simple_get(text, "wandb_run_id"))
    local_wandb_run_id = discover_latest_local_wandb_run_id(run_id)
    wandb_run_id = local_wandb_run_id or configured_wandb_run_id

    lr = parse_float(section_get(text, "agent", "learning_rate", "0.001"), 0.001)

    lin_w = parse_float(section_get(text, "reward_weights", "track_linear_velocity", "2.0"), 2.0)
    yaw_w = parse_float(section_get(text, "reward_weights", "track_angular_velocity", "2.0"), 2.0)
    upright_w = parse_float(section_get(text, "reward_weights", "upright", "1.0"), 1.0)
    pose_w = parse_float(section_get(text, "reward_weights", "pose", "1.0"), 1.0)

    vel_err_threshold = parse_float(section_get(text, "autotune", "velocity_error_threshold", "0.25"), 0.25)
    l3_crash_threshold = parse_float(section_get(text, "autotune", "l3_reward_crash_threshold", "-30.0"), -30.0)
    cooldown = int(parse_float(section_get(text, "autotune", "l1_cooldown_iterations", "500"), 500))

    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.history_jsonl:
        print("🔎 正在读取本地 W&B history fixture")
        print(f"history_jsonl: {args.history_jsonl}")
        run = None
        wandb_path = f"offline/{run_id}"
        rows = load_history_jsonl(args.history_jsonl)
    else:
        try:
            import wandb  # noqa: F401
        except Exception:
            print("❌ 当前 python3 没有 wandb。请先执行：python3 -m pip install --user wandb")
            sys.exit(1)

        print("🔎 正在读取 W&B 曲线")
        print(f"entity: {entity}")
        print(f"project: {project}")
        if wandb_run_id:
            print(f"run_id: {wandb_run_id}")
            if configured_wandb_run_id and local_wandb_run_id and configured_wandb_run_id != local_wandb_run_id:
                print(f"说明：使用训练日志里的最新 W&B run_id，覆盖配置中的旧值 {configured_wandb_run_id}")
        print(f"run_name: {run_name}")

        run = find_latest_wandb_run(entity, project, run_name, wandb_run_id)
        wandb_path = "/".join(run.path)
        print(f"✅ 找到 W&B run: {run.path}")
        rows = collect_history(run)

    if not rows:
        print("❌ 没有从 W&B 读到可分析的曲线数据")
        sys.exit(1)

    step = latest_step(rows)

    metrics = {
        "step": step,
        "reward_last20": mean_last(rows, "reward", 20),
        "reward_slope40": slope_last(rows, "reward", 40),
        "episode_length_last20": mean_last(rows, "episode_length", 20),
        "track_linear_velocity_last20": mean_last(rows, "track_linear_velocity", 20),
        "track_linear_velocity_slope40": slope_last(rows, "track_linear_velocity", 40),
        "track_angular_velocity_last20": mean_last(rows, "track_angular_velocity", 20),
        "error_vel_xy_last20": mean_last(rows, "error_vel_xy", 20),
        "error_vel_yaw_last20": mean_last(rows, "error_vel_yaw", 20),
        "fell_over_last20": mean_last(rows, "fell_over", 20),
        "upright_last20": mean_last(rows, "upright", 20),
        "pose_last20": mean_last(rows, "pose", 20),
    }

    write_train_jsonl(run_dir, rows, metrics)

    history_path = run_dir / "wandb_decision_history.jsonl"
    last_changed_step = latest_decision_step(history_path)

    reasons = []
    actions = []
    severity = "hold"
    changed = False

    reward_last = metrics["reward_last20"]
    reward_slope = metrics["reward_slope40"]
    lin_last = metrics["track_linear_velocity_last20"]
    lin_slope = metrics["track_linear_velocity_slope40"]
    err_xy = metrics["error_vel_xy_last20"]
    err_yaw = metrics["error_vel_yaw_last20"]
    fell = metrics["fell_over_last20"]

    if reward_last is not None and reward_last < l3_crash_threshold:
        severity = "L3"
        reasons.append(f"mean reward 过低: {reward_last:.4f} < {l3_crash_threshold}")

    if fell is not None and fell > 0.25:
        severity = "L3"
        reasons.append(f"fell_over 太高: {fell:.4f}")

    in_cooldown = (
        last_changed_step is not None
        and step > 0
        and step - int(last_changed_step) < cooldown
    )

    if severity != "L3" and not in_cooldown:
        if err_xy is not None and err_xy > vel_err_threshold:
            severity = "L1"
            new_lin_w = min(lin_w * 1.10, 5.0)
            actions.append(("reward_weights", "track_linear_velocity", new_lin_w))
            reasons.append(f"线速度误差偏高: error_vel_xy={err_xy:.4f} > {vel_err_threshold}")

        if lin_last is not None and lin_last < 0.35 and step > 500:
            severity = "L1"
            new_lin_w = min(lin_w * 1.10, 5.0)
            actions.append(("reward_weights", "track_linear_velocity", new_lin_w))
            reasons.append(f"线速度奖励偏低: track_linear_velocity={lin_last:.4f}")

        if lin_slope is not None and lin_slope < -0.03:
            severity = "L1"
            new_lr = max(lr * 0.90, 0.00005)
            actions.append(("agent", "learning_rate", new_lr))
            reasons.append(f"线速度奖励最近下降: slope={lin_slope:.4f}")

        if err_yaw is not None and err_yaw > 0.25:
            severity = "L1"
            new_yaw_w = min(yaw_w * 1.08, 5.0)
            actions.append(("reward_weights", "track_angular_velocity", new_yaw_w))
            reasons.append(f"转向误差偏高: error_vel_yaw={err_yaw:.4f}")

    elif in_cooldown:
        reasons.append(f"还在 cooldown 内：当前 step={step}, 上次修改 step={last_changed_step}, cooldown={cooldown}")

    # 去重：同一个 section/key 只保留最后一个
    dedup = {}
    for sec, key, value in actions:
        dedup[(sec, key)] = value
    actions = [(sec, key, value) for (sec, key), value in dedup.items()]

    decision = {
        "time": datetime.now(timezone.utc).isoformat(),
        "source": "wandb",
        "run_id": run_id,
        "wandb_path": wandb_path,
        "step": step,
        "severity": severity,
        "metrics": metrics,
        "reasons": reasons,
        "actions": [
            {"section": sec, "key": key, "value": value}
            for sec, key, value in actions
        ],
        "changed": False,
        "applied": args.apply,
    }

    print("\n====== W&B 自动决策结果 ======")
    print(json.dumps(decision, ensure_ascii=False, indent=2))

    (run_dir / "wandb_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")

    if args.apply and severity == "L3":
        print("🛑 L3：自动暂停训练")
        subprocess.run(["python3", "scripts/pause_training.py"], cwd=ROOT)
        decision["changed"] = True
        append_jsonl(Path(args.adjustments_path), {
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "run_id": run_id,
            "profile_name": "mjlab_wandb_autodecide",
            "trainer_kind": "mjlab",
            "source": "wandb",
            "level": "L3",
            "decision": "auto_applied",
            "target": "pause_training",
            "reason": "; ".join(reasons),
            "latest_step": step,
            "restart_required": False,
        })
        append_jsonl(history_path, decision)
        sys.exit(0)

    if args.apply and actions:
        new_text = read_cfg_text()
        for sec, key, value in actions:
            if isinstance(value, float):
                value_s = f"{value:.8g}"
            else:
                value_s = str(value)
            print(f"✏️ 修改配置: {sec}.{key} = {value_s}")
            new_text = set_section_value(new_text, sec, key, value_s)

        CFG_PATH.write_text(new_text, encoding="utf-8")
        decision["changed"] = True
        changed = True
        log_adjustments(Path(args.adjustments_path), run_id, severity, step, actions, text, reasons)

        (run_dir / "wandb_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
        append_jsonl(history_path, decision)

        if not args.no_restart:
            print("🔁 配置已修改，开始从 checkpoint 自动重启 MJLab")
            subprocess.run(args.restart_cmd.split(), cwd=ROOT)
    else:
        append_jsonl(history_path, decision)
        if severity == "hold":
            print("✅ 当前 W&B 曲线不需要调参")
        elif not args.apply:
            print("⚠️ 只是 dry-run，没有真正修改。要自动执行请加 --apply")

    print("\n✅ W&B 曲线自动决策流程完成")


if __name__ == "__main__":
    main()
