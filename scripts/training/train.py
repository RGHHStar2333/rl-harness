import argparse
import json
import os
import time
import yaml
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path):
    full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full_path, "r", encoding="utf-8") as f:
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
            reward_mean = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)

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

            print(f"✅ checkpoint saved: {ckpt_path}", flush=True)

        return True


def build_wandb_config(pipeline, hyper, reward_cfg, resume_checkpoint=None):
    return {
        "project_name": pipeline["project"]["name"],
        "task": pipeline["project"]["task"],
        "run_id": pipeline["project"]["run_id"],
        "env_id": pipeline["environment"]["env_id"],
        "algorithm": pipeline["environment"]["algorithm"],
        "total_timesteps": pipeline["environment"]["total_timesteps"],
        "device": pipeline["environment"]["device"],
        "resume_checkpoint": resume_checkpoint,
        "ppo": hyper.get("ppo", {}),
        "env_kwargs": reward_cfg.get("env_kwargs", {}),
        "reward_weights": reward_cfg.get("reward_weights", {}),
        "checkpoint": pipeline.get("checkpoint", {}),
        "feedback_flywheel": pipeline.get("feedback_flywheel", {}),
    }


def apply_current_learning_rate(model, new_lr):
    new_lr = float(new_lr)

    model.learning_rate = new_lr
    model.lr_schedule = lambda _: new_lr

    if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
        for group in model.policy.optimizer.param_groups:
            group["lr"] = new_lr

    print(f"✅ 当前配置 learning_rate 已应用到模型: {new_lr}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()

    pipeline = load_yaml(args.config)
    hyper = load_yaml(pipeline["paths"]["hyper_config"])
    reward_cfg = load_yaml(pipeline["paths"]["reward_config"])

    run_dir = os.path.join(ROOT, pipeline["paths"]["run_dir"])
    checkpoint_dir = os.path.join(ROOT, pipeline["paths"]["checkpoint_dir"])
    tensorboard_dir = os.path.join(run_dir, "tensorboard")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    env_id = pipeline["environment"]["env_id"]
    total_timesteps = int(pipeline["environment"]["total_timesteps"])
    device = pipeline["environment"]["device"]

    ppo_cfg = hyper["ppo"]
    current_lr = float(ppo_cfg["learning_rate"])

    tracking_cfg = pipeline.get("tracking", {})
    wandb_enabled = bool(tracking_cfg.get("wandb_enabled", False))

    wandb_run = None

    if wandb_enabled:
        if wandb is None or WandbCallback is None:
            raise RuntimeError("你开启了 wandb_enabled，但没有安装 wandb。请运行：pip install wandb")

        wandb_kwargs = {
            "project": tracking_cfg.get("wandb_project", "rl-harness"),
            "name": pipeline["project"]["run_id"],
            "config": build_wandb_config(pipeline, hyper, reward_cfg, args.resume_checkpoint),
            "sync_tensorboard": bool(tracking_cfg.get("sync_tensorboard", True)),
            "save_code": bool(tracking_cfg.get("save_code", True)),
        }

        entity = tracking_cfg.get("wandb_entity")
        if entity:
            wandb_kwargs["entity"] = entity

        wandb_run = wandb.init(**wandb_kwargs)
        print(f"✅ wandb 已启动: project={wandb_kwargs['project']}, run={pipeline['project']['run_id']}", flush=True)

    env_kwargs = reward_cfg.get("env_kwargs", {})
    env = gym.make(env_id, **env_kwargs)
    env = Monitor(env)

    if args.resume_checkpoint:
        ckpt = args.resume_checkpoint
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(ROOT, ckpt)

        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"找不到 resume checkpoint: {ckpt}")

        print(f"🔁 从 checkpoint 恢复训练: {ckpt}", flush=True)

        model = PPO.load(
            ckpt,
            env=env,
            device=device,
            tensorboard_log=tensorboard_dir,
        )

        apply_current_learning_rate(model, current_lr)

        reset_num_timesteps = False

    else:
        print("🚀 从头启动训练。", flush=True)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=current_lr,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            verbose=1,
            device=device,
            tensorboard_log=tensorboard_dir,
        )

        reset_num_timesteps = True

    callbacks = [
        JsonlLoggingCallback(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            save_interval=pipeline["checkpoint"]["save_interval"],
            log_every=1000,
        )
    ]

    if wandb_enabled:
        callbacks.append(
            WandbCallback(
                model_save_path=os.path.join(checkpoint_dir, "wandb_models"),
                model_save_freq=pipeline["checkpoint"]["save_interval"],
                gradient_save_freq=0,
                verbose=2,
            )
        )

    callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=pipeline["project"]["run_id"],
        reset_num_timesteps=reset_num_timesteps,
    )

    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)

    env.close()

    if wandb_run is not None:
        wandb_run.finish()

    print(f"✅ training finished: {final_path}", flush=True)


if __name__ == "__main__":
    main()
