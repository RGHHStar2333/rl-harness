import json
import os
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/tasks/mjlab/go1.yaml"

def now():
    return datetime.now().isoformat(timespec="seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))

    mj = cfg["mjlab"]
    agent = cfg.get("agent", {})
    reward_weights = cfg.get("reward_weights", {})

    run_id = mj["run_id"]
    project_dir = Path(mj["project_dir"])
    task = mj["task"]

    cmd = [
        "uv", "run", "train", task,
        "--env.scene.num-envs", str(mj["num_envs"]),
        "--agent.max-iterations", str(mj["max_iterations"]),
        "--agent.logger", "wandb",
        "--agent.wandb-project", str(mj["wandb_project"]),
        "--agent.run-name", str(mj["wandb_name"]),
    ]

    if "learning_rate" in agent:
        cmd += ["--agent.algorithm.learning-rate", str(agent["learning_rate"])]

    for name, value in reward_weights.items():
        cli_name = name.replace("_", "-")
        cmd += [f"--env.rewards.{cli_name}.weight", str(value)]

    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "training_process.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("🚀 启动 MJLab 训练")
    print("run_id:", run_id)
    print("project_dir:", project_dir)
    print("cmd:", " ".join(cmd))
    print("log:", log_path)

    if args.dry_run or args.print_command:
        print("dry_run: true")
        return

    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f"[{now()}] START\n")
    log_file.write("cmd: " + " ".join(cmd) + "\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(project_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    active = {
        "status": "running",
        "kind": "mjlab",
        "run_id": run_id,
        "pid": proc.pid,
        "project_dir": str(project_dir),
        "command": " ".join(cmd),
        "learning_rate": agent.get("learning_rate"),
        "reward_weights": reward_weights,
        "log": str(log_path),
        "started_at": now(),
    }

    (ROOT / "runs/active_training.json").write_text(
        json.dumps(active, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ MJLab 训练已后台启动")
    print("PID:", proc.pid)
    print("看日志：")
    print(f"tail -f {log_path}")

if __name__ == "__main__":
    main()
