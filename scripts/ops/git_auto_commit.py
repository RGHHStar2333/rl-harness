import argparse
import os
import subprocess
import time


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd, check=False):
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def git_config_get(key):
    result = run(["git", "config", "--get", key])
    return result.stdout.strip()


def git_config_set(key, value):
    run(["git", "config", key, value])


def ensure_git_identity():
    name = git_config_get("user.name")
    email = git_config_get("user.email")

    if not name:
        git_config_set("user.name", "RL Harness Agent")

    if not email:
        git_config_set("user.email", "rl-harness-agent@example.local")


def stage_allowed_files():
    # 只提交代码、配置和调整记录；不提交 checkpoints、wandb、大日志。
    run(["git", "add", "-A", "configs", "scripts"], check=False)

    force_paths = [
        "runs/adjustments.jsonl",
    ]

    for path in force_paths:
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            run(["git", "add", "-f", path], check=False)

    normal_paths = [
        "AGENTS.md",
        ".gitignore",
    ]

    for path in normal_paths:
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            run(["git", "add", path], check=False)


def staged_files():
    result = run(["git", "diff", "--cached", "--name-only"])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def has_staged_changes():
    result = run(["git", "diff", "--cached", "--quiet"])
    return result.returncode != 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="AUTO")
    parser.add_argument("--reason", default="automatic harness update")
    args = parser.parse_args()

    inside = run(["git", "rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0:
        print("⚠️ 当前目录不是 Git 仓库，跳过自动 commit。")
        return

    ensure_git_identity()
    stage_allowed_files()

    files = staged_files()

    if not has_staged_changes():
        print("✅ Git auto commit：没有需要提交的变更。")
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    message = f"auto: {args.level} harness update"

    body = [
        f"reason: {args.reason}",
        f"time: {timestamp}",
        "",
        "files:",
        *[f"- {f}" for f in files],
    ]

    commit_cmd = ["git", "commit", "-m", message]
    for line in body:
        commit_cmd.extend(["-m", line])

    result = run(commit_cmd)

    if result.returncode == 0:
        print("✅ Git auto commit 成功。")
        print(result.stdout.strip())
    else:
        print("⚠️ Git auto commit 失败，但不影响训练闭环继续运行。")
        print(result.stdout)
        print(result.stderr)


if __name__ == "__main__":
    main()
