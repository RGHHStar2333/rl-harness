#!/usr/bin/env python3
"""
MJLab 实验调度器

按预设的三个实验顺序执行，每个实验执行固定时长后自动切换下一组。
被 cron 每 5 分钟轮询，通过状态文件追踪当前实验进度。

实验计划 (G1):
  1. 4096 并行, 8000 次迭代, 1 小时
  2. 2048 并行, 12000 次迭代, 2 小时
  3. 4096 并行, 5000 次迭代, 30 分钟
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/tasks/mjlab/go1.yaml"
FEEDBACK_PATH = ROOT / "configs/tasks/mjlab/feedback.yaml"
STATE_PATH = ROOT / "runs/experiment_schedule.json"
ACTIVE_PATH = ROOT / "runs/active_training.json"
START_TRAINING = ROOT / "scripts/start_mjlab_training.py"
PAUSE_TRAINING = ROOT / "scripts/pause_training.py"
RUN_MONITOR = ROOT / "scripts/run_monitor_for_hermes.sh"

EXPERIMENTS = [
    {
        "name": "G1-4096x8000-1h",
        "num_envs": 4096,
        "max_iterations": 8000,
        "run_duration_seconds": 3600,  # 1 小时
        "run_id": "mjlab_g1_4096_8000",
        "wandb_name": "mjlab_g1_4096_8000",
    },
    {
        "name": "G1-2048x12000-2h",
        "num_envs": 2048,
        "max_iterations": 12000,
        "run_duration_seconds": 7200,  # 2 小时
        "run_id": "mjlab_g1_2048_12000",
        "wandb_name": "mjlab_g1_2048_12000",
    },
    {
        "name": "G1-4096x5000-30min",
        "num_envs": 4096,
        "max_iterations": 5000,
        "run_duration_seconds": 1800,  # 30 分钟
        "run_id": "mjlab_g1_4096_5000",
        "wandb_name": "mjlab_g1_4096_5000",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_ts() -> float:
    return time.time()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, data: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def update_go1_config(exp: dict) -> None:
    """更新 go1.yaml 配置为当前实验参数"""
    cfg = load_yaml(CFG_PATH)
    cfg.setdefault("mjlab", {}).update({
        "run_id": exp["run_id"],
        "num_envs": exp["num_envs"],
        "max_iterations": exp["max_iterations"],
        "wandb_name": exp["wandb_name"],
    })
    dump_yaml(CFG_PATH, cfg)
    print(f"  go1.yaml 已更新: run_id={exp['run_id']}, num_envs={exp['num_envs']}, max_iterations={exp['max_iterations']}")


def update_feedback_run_id(run_id: str) -> None:
    """更新 feedback.yaml 中的 run_id 和 metric_log 路径"""
    cfg = load_yaml(FEEDBACK_PATH)
    cfg.setdefault("feedback_profile", {}).setdefault("project", {})["run_id"] = run_id
    cfg["feedback_profile"]["paths"]["metric_log"] = f"runs/{run_id}/train.jsonl"
    dump_yaml(FEEDBACK_PATH, cfg)
    print(f"  feedback.yaml 已更新: run_id={run_id}")


def start_training() -> bool:
    """调用 start_mjlab_training.py 启动训练，返回是否成功"""
    # 先停止当前训练
    if ACTIVE_PATH.exists():
        print("  正在停止当前训练...")
        subprocess.run([sys.executable, str(PAUSE_TRAINING)], cwd=str(ROOT), capture_output=True, timeout=30)
        pkill = subprocess.run(
            ["pkill", "-f", "uv run train Mjlab-Velocity-Flat-Unitree-G1"],
            capture_output=True, timeout=10
        )
        if pkill.returncode not in (0, 1):
            print(f"  pkill 返回值: {pkill.returncode}")
        time.sleep(3)

    result = subprocess.run(
        [sys.executable, str(START_TRAINING)],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=30
    )
    print(f"  start_mjlab_training 退出码: {result.returncode}")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  stderr: {line}")
    return result.returncode == 0


def is_training_alive() -> bool:
    """检查 active_training.json 中的进程是否还在运行"""
    active = load_json(ACTIVE_PATH)
    if not active or active.get("status") != "running":
        return False
    pid = active.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_state() -> dict | None:
    return load_json(STATE_PATH)


def ensure_state() -> dict:
    state = get_state()
    if state is None:
        state = {
            "version": 1,
            "status": "idle",           # idle | running | paused | completed | error
            "current_index": 0,
            "experiment_started_at": None,
            "experiment_duration_seconds": None,
            "completed_indices": [],
            "error": None,
            "last_updated": now_iso(),
        }
        dump_json(STATE_PATH, state)
    return state


def save_state(state: dict) -> None:
    state["last_updated"] = now_iso()
    dump_json(STATE_PATH, state)


def tick() -> str:
    """
    每次 cron 调用的主逻辑。
    返回人类可读的状态摘要。
    """
    state = ensure_state()
    lines = []

    if state["status"] == "completed":
        lines.append("🏁 所有实验已完成。")
        lines.append(f"  已完成: {state['completed_indices']}")
        return "\n".join(lines)

    if state["status"] == "error":
        lines.append(f"❌ 调度器处于错误状态: {state.get('error')}")
        lines.append(f"  手动修复后设置 status = idle 可恢复")
        return "\n".join(lines)

    current_idx = state["current_index"]
    if current_idx >= len(EXPERIMENTS):
        state["status"] = "completed"
        save_state(state)
        lines.append("🏁 所有实验已完成！")
        lines.append(f"  已完成实验: {state['completed_indices']}")
        return "\n".join(lines)

    exp = EXPERIMENTS[current_idx]

    if state["status"] == "idle":
        # 启动当前实验
        lines.append(f"🚦 启动第 {current_idx + 1} 个实验: {exp['name']}")
        lines.append(f"  num_envs={exp['num_envs']}, max_iterations={exp['max_iterations']}")
        lines.append(f"  预期时长: {exp['run_duration_seconds'] // 60} 分钟")

        update_go1_config(exp)
        update_feedback_run_id(exp["run_id"])

        ok = start_training()
        if not ok:
            state["status"] = "error"
            state["error"] = f"启动训练失败 (实验 {current_idx + 1})"
            save_state(state)
            lines.append(f"❌ 启动训练失败")
            return "\n".join(lines)

        state["status"] = "running"
        state["experiment_started_at"] = now_iso()
        state["experiment_started_ts"] = now_ts()
        state["experiment_duration_seconds"] = exp["run_duration_seconds"]
        save_state(state)
        lines.append(f"✅ 实验 {exp['name']} 已启动")

    elif state["status"] == "running":
        elapsed = now_ts() - state.get("experiment_started_ts", now_ts())
        duration = state["experiment_duration_seconds"] or exp["run_duration_seconds"]
        remaining = max(0, duration - elapsed)
        alive = is_training_alive()

        lines.append(f"⏳ 实验 {current_idx + 1}: {exp['name']}")
        lines.append(f"  已运行: {elapsed / 60:.1f} 分钟 / 目标 {duration / 60:.1f} 分钟")
        lines.append(f"  剩余: {remaining / 60:.1f} 分钟")
        lines.append(f"  训练进程: {'存活' if alive else '已停止'}")

        if elapsed >= duration:
            # 时间到，切换到下一个实验
            lines.append("⏰ 时间到！切换到下一组实验...")
            state["completed_indices"].append(current_idx)
            state["current_index"] = current_idx + 1
            state["status"] = "idle"
            state["experiment_started_at"] = None
            state["experiment_started_ts"] = None
            state["experiment_duration_seconds"] = None
            save_state(state)

            # 停止当前训练（如果还活着）
            if alive:
                subprocess.run([sys.executable, str(PAUSE_TRAINING)], cwd=str(ROOT), capture_output=True, timeout=30)
                subprocess.run(["pkill", "-f", "uv run train Mjlab-Velocity-Flat-Unitree-G1"], capture_output=True, timeout=10)
                time.sleep(2)

            lines.append(f"✅ 实验 {current_idx + 1} 完成")
            lines.append(f"  下一个实验: {current_idx + 2}/{len(EXPERIMENTS)}")

            # 如果这是最后一个实验，这里 status 已设为 idle，等下一次 tick 就会设为 completed
            if state["current_index"] >= len(EXPERIMENTS):
                state["status"] = "completed"
                save_state(state)
                lines.append("🏁 所有实验完成！")

    return "\n".join(lines)


def main() -> None:
    output = tick()
    print(output)


if __name__ == "__main__":
    main()
