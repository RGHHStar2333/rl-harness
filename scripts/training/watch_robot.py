import os
import re
import time
import yaml
import gymnasium as gym
from stable_baselines3 import PPO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_latest_checkpoint(checkpoint_dir):
    candidates = []

    for name in os.listdir(checkpoint_dir):
        m = re.match(r"model_step_(\d+)\.zip$", name)
        if m:
            step = int(m.group(1))
            candidates.append((step, os.path.join(checkpoint_dir, name)))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    final_model = os.path.join(checkpoint_dir, "final_model.zip")
    if os.path.exists(final_model):
        return final_model

    raise FileNotFoundError(f"没有找到 checkpoint: {checkpoint_dir}")

pipeline = load_yaml("configs/pipeline.yaml")
reward_cfg = load_yaml(pipeline["paths"]["reward_config"])

checkpoint_dir = os.path.join(ROOT, pipeline["paths"]["checkpoint_dir"])
checkpoint = find_latest_checkpoint(checkpoint_dir)

print("✅ 使用 checkpoint:")
print(checkpoint)

env = gym.make(
    pipeline["environment"]["env_id"],
    render_mode="human",
    **reward_cfg.get("env_kwargs", {})
)

model = PPO.load(
    checkpoint,
    env=env,
    device=pipeline["environment"]["device"]
)

obs, info = env.reset()

print("✅ MuJoCo 窗口已启动。按 Ctrl+C 退出。")

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        time.sleep(1 / 60)

        if terminated or truncated:
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\n已退出观看。")

finally:
    env.close()
