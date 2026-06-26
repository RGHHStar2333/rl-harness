import argparse
import json
import os
import signal
import subprocess
import time


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_PATH = os.path.join(ROOT, "runs", "active_training.json")


def load_state():
    if not os.path.exists(STATE_PATH):
        return None

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def ps_line(pid):
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid,ppid,pgid,stat,cmd"],
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ps 查询失败: {e}"


def pause_training(reason, timeout, force):
    state = load_state()

    if not state:
        print("⚠️ 没有找到 runs/active_training.json。")
        print("说明：没有被 start_training.sh 管理的训练进程。")
        return 1

    pid = int(state.get("pid", 0))

    if pid <= 0:
        print("⚠️ active_training.json 里没有有效 PID。")
        return 1

    print("📌 当前训练进程：")
    print(ps_line(pid))

    if not pid_alive(pid):
        state["status"] = "not_running"
        state["stopped_at"] = time.time()
        state["stop_reason"] = "process already not running"
        save_state(state)
        print("⚠️ 进程已经不在运行。状态已更新。")
        return 0

    print("")
    print(f"🛑 正在暂停训练 PID={pid}")
    print(f"原因：{reason}")

    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            print(f"❌ 发送 SIGTERM 失败: {e}")
            return 1

    deadline = time.time() + timeout

    while time.time() < deadline:
        if not pid_alive(pid):
            break
        time.sleep(0.3)

    if pid_alive(pid):
        if not force:
            print("⚠️ SIGTERM 后进程仍在运行。")
            print("需要强制停止可运行：python3 scripts/training/pause_training.py --force")
            return 1

        print("⚠️ 进程未正常退出，执行 SIGKILL 强制停止。")
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception as e:
                print(f"❌ 发送 SIGKILL 失败: {e}")
                return 1

        time.sleep(1)

    state["status"] = "paused"
    state["stopped_at"] = time.time()
    state["stop_reason"] = reason
    state["stop_method"] = "SIGKILL" if force else "SIGTERM"
    save_state(state)

    print("✅ 训练进程已暂停。")
    print(f"状态文件：{STATE_PATH}")
    return 0


def show_status():
    state = load_state()

    if not state:
        print("暂无 active training 状态。")
        return

    pid = state.get("pid")
    print(json.dumps(state, ensure_ascii=False, indent=2))

    if pid:
        print("")
        print("进程状态：")
        print(ps_line(pid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="manual pause")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    raise SystemExit(pause_training(args.reason, args.timeout, args.force))


if __name__ == "__main__":
    main()
