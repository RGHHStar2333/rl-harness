import argparse
import json
import os
import time

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class JsonlLoggingCallback(BaseCallback):
    def __init__(self, run_dir, checkpoint_dir, save_interval, log_every=1000):
        super().__init__()
        self.run_dir = run_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.log_every = log_every

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.log_path = os.path.join(self.run_dir, "train.jsonl")
        self.ckpt_index_path = os.path.join(self.checkpoint_dir, "index.jsonl")

    def _on_step(self):
        step = self.num_timesteps

        reward_mean = None
        if len(self.model.ep_info_buffer) > 0:
            reward_mean = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(
                self.model.ep_info_buffer
            )

        if step % self.log_every == 0:
            record = {
                "timestamp": time.time(),
                "train/step": step,
                "train/episode_reward_mean": reward_mean,
                "train/policy_loss": None,
                "system/device": str(self.model.device),
            }

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step > 0 and step % self.save_interval == 0:
            ckpt_path = os.path.join(self.checkpoint_dir, f"model_step_{step}.zip")
            self.model.save(ckpt_path)

            ckpt_record = {
                "timestamp": time.time(),
                "step": step,
                "path": ckpt_path,
                "train/episode_reward_mean": reward_mean,
            }

            with open(self.ckpt_index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ckpt_record, ensure_ascii=False) + "\n")

            print(f"✅ checkpoint saved: {ckpt_path}")

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    pipeline = load_yaml(args.config)
    hyper = load_yaml(pipeline["paths"]["hyper_config"])
    reward_cfg = load_yaml(pipeline["paths"]["reward_config"])

    run_dir = os.path.join(ROOT, pipeline["paths"]["run_dir"])
    checkpoint_dir = os.path.join(ROOT, pipeline["paths"]["checkpoint_dir"])

    env_id = pipeline["environment"]["env_id"]
    total_timesteps = pipeline["environment"]["total_timesteps"]
    device = pipeline["environment"]["device"]

    env_kwargs = reward_cfg.get("env_kwargs", {})
    env = gym.make(env_id, **env_kwargs)

    ppo_cfg = hyper["ppo"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        verbose=1,
        device=device,
    )

    callback = JsonlLoggingCallback(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        save_interval=pipeline["checkpoint"]["save_interval"],
        log_every=1000,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)

    env.close()

    print(f"✅ training finished: {final_path}")


if __name__ == "__main__":
    main()
