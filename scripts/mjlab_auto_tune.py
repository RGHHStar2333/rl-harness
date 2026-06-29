import argparse
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/tasks/mjlab/go1.yaml"
ADJ_PATH = ROOT / "runs/adjustments.jsonl"

def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def save_yaml(path, data):
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

def get_path(d, path):
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur

def set_path(d, path, value):
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_rows(run_id):
    p = ROOT / "runs" / run_id / "train.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def load_state(run_id):
    p = ROOT / "runs" / run_id / "mjlab_autotune_state.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_state(run_id, state):
    p = ROOT / "runs" / run_id / "mjlab_autotune_state.json"
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def make_change(cfg, path, new_value, reason):
    old_value = get_path(cfg, path)
    return {
        "path": path,
        "old": old_value,
        "new": new_value,
        "reason": reason,
    }

def apply_changes(cfg, changes):
    for c in changes:
        set_path(cfg, c["path"], c["new"])

def log_adjustment(level, run_id, changes, reason, requires_restart=True):
    append_jsonl(ADJ_PATH, {
        "time": time.time(),
        "system": "mjlab",
        "level": level,
        "run_id": run_id,
        "reason": reason,
        "requires_restart": requires_restart,
        "changes": changes,
    })

def confirm_l2(token, dry_run=False):
    pending = ROOT / "runs/mjlab_l2_pending" / f"{token}.json"
    if not pending.exists():
        print(f"❌ 找不到 L2 token: {token}")
        sys.exit(1)

    data = json.loads(pending.read_text(encoding="utf-8"))
    cfg = load_yaml(CFG_PATH)

    print("L2 changes:")
    for c in data["changes"]:
        print(f"- {c['path']}: {c['old']} -> {c['new']} | {c['reason']}")

    if dry_run:
        print("dry-run: 不写入")
        return

    apply_changes(cfg, data["changes"])
    save_yaml(CFG_PATH, cfg)

    log_adjustment("L2", data["run_id"], data["changes"], data["reason"], requires_restart=True)
    pending.unlink()

    print("✅ L2 已确认并写入配置")
    print("⚠️ 需要重启 MJLab 训练才会生效")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.confirm:
        confirm_l2(args.confirm, dry_run=args.dry_run)
        return

    cfg = load_yaml(CFG_PATH)
    mj = cfg["mjlab"]
    autotune = cfg.get("autotune", {})
    run_id = mj["run_id"]

    if not autotune.get("enabled", True):
        print("autotune disabled")
        return

    rows = load_rows(run_id)
    if not rows:
        print(f"❌ 没有指标文件，请先运行: python3 scripts/mjlab_parse_log.py --run-id {run_id}")
        return

    last = rows[-1]
    latest_iter = int(last.get("mjlab/iteration", len(rows)))
    latest_reward = last.get("train/episode_reward_mean")

    if latest_reward is None:
        print("❌ train/episode_reward_mean 不存在")
        return

    state = load_state(run_id)

    # L3
    l3_threshold = float(autotune.get("l3_reward_crash_threshold", -30.0))
    if latest_reward < l3_threshold:
        print(f"🚨 L3: reward {latest_reward} < {l3_threshold}, pause training")
        if not args.dry_run:
            subprocess.run([sys.executable, "scripts/pause_training.py"], cwd=str(ROOT))
            log_adjustment("L3", run_id, [], f"reward crash: {latest_reward}", requires_restart=False)
        return

    changes = []

    # L1
    l1_window = int(autotune.get("l1_window", 5))
    l1_min_improvement = float(autotune.get("l1_min_improvement", 0.2))
    l1_cooldown = int(autotune.get("l1_cooldown_iterations", 500))
    last_l1_iter = int(state.get("last_l1_iter", -10**9))

    if len(rows) >= l1_window and latest_iter - last_l1_iter >= l1_cooldown:
        old_reward = rows[-l1_window].get("train/episode_reward_mean")
        if old_reward is not None:
            improvement = latest_reward - old_reward
            if improvement < l1_min_improvement:
                lr = float(get_path(cfg, "agent.learning_rate"))
                changes.append(make_change(
                    cfg,
                    "agent.learning_rate",
                    round(lr * 0.95, 10),
                    f"L1 plateau: improvement {improvement:.4f} < {l1_min_improvement}",
                ))

                vel_err = float(last.get("Metrics/twist/error_vel_xy", 0.0))
                err_threshold = float(autotune.get("velocity_error_threshold", 0.25))
                if vel_err > err_threshold:
                    w = float(get_path(cfg, "reward_weights.track_linear_velocity"))
                    changes.append(make_change(
                        cfg,
                        "reward_weights.track_linear_velocity",
                        round(w * 1.05, 6),
                        f"L1 velocity error high: error_vel_xy {vel_err:.4f} > {err_threshold}",
                    ))

    if changes:
        print("✅ L1 will apply:")
        for c in changes:
            print(f"- {c['path']}: {c['old']} -> {c['new']} | {c['reason']}")

        if not args.dry_run:
            apply_changes(cfg, changes)
            save_yaml(CFG_PATH, cfg)
            state["last_l1_iter"] = latest_iter
            save_state(run_id, state)
            log_adjustment("L1", run_id, changes, "MJLab L1 auto tune", requires_restart=True)

        print("⚠️ L1 已写入配置；需要重启 MJLab 训练才会生效")
        return

    # L2
    l2_window = int(autotune.get("l2_window", 10))
    l2_min_improvement = float(autotune.get("l2_min_improvement", 0.5))
    l2_cooldown = int(autotune.get("l2_cooldown_iterations", 1000))
    last_l2_iter = int(state.get("last_l2_iter", -10**9))

    if len(rows) >= l2_window and latest_iter - last_l2_iter >= l2_cooldown:
        old_reward = rows[-l2_window].get("train/episode_reward_mean")
        if old_reward is not None:
            improvement = latest_reward - old_reward
            if improvement < l2_min_improvement:
                lr = float(get_path(cfg, "agent.learning_rate"))
                lin_w = float(get_path(cfg, "reward_weights.track_linear_velocity"))
                act_w = float(get_path(cfg, "reward_weights.action_rate_l2"))
                slip_w = float(get_path(cfg, "reward_weights.foot_slip"))

                l2_changes = [
                    make_change(cfg, "agent.learning_rate", round(lr * 0.8, 10), "L2 lower learning rate"),
                    make_change(cfg, "reward_weights.track_linear_velocity", round(lin_w * 1.2, 6), "L2 strengthen linear velocity tracking"),
                    make_change(cfg, "reward_weights.action_rate_l2", round(act_w * 0.85, 6), "L2 reduce action-rate penalty magnitude"),
                    make_change(cfg, "reward_weights.foot_slip", round(slip_w * 0.85, 6), "L2 reduce foot-slip penalty magnitude"),
                ]

                token = secrets.token_hex(4)
                pending_dir = ROOT / "runs/mjlab_l2_pending"
                pending_dir.mkdir(parents=True, exist_ok=True)
                pending = {
                    "token": token,
                    "run_id": run_id,
                    "reason": f"MJLab L2 plateau: improvement {improvement:.4f} < {l2_min_improvement}",
                    "changes": l2_changes,
                }

                if not args.dry_run:
                    (pending_dir / f"{token}.json").write_text(
                        json.dumps(pending, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    state["last_l2_iter"] = latest_iter
                    save_state(run_id, state)

                print("🟡 L2 needs confirmation")
                print("TOKEN:", token)
                print("确认命令：")
                print(f"bash scripts/mjlab_l2_confirm.sh {token}")
                for c in l2_changes:
                    print(f"- {c['path']}: {c['old']} -> {c['new']} | {c['reason']}")
                return

    print("✅ no MJLab L1/L2/L3 action")

if __name__ == "__main__":
    main()
