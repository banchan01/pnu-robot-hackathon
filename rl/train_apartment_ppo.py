"""Train PPO policy from scratch on the Webots Apartment Environment.

- Vectorized environment replicating worlds/apartment.wbt (walls, furniture, moving pedestrian).
- Trains local obstacle avoidance, door navigation, and human-dodging policy.
- Saves trained model to rl/amr_apartment_ppo_model.zip and amr_apartment_policy.pt.
"""

import os
import sys
import time
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.apartment_env import ApartmentGymEnv


def make_env_fn():
    def _init():
        return ApartmentGymEnv()
    return _init


import argparse

def main():
    parser = argparse.ArgumentParser(description="Train AMR PPO in Apartment Environment")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps to train (default: 1,000,000)")
    parser.add_argument("--envs", type=int, default=6, help="Number of parallel environments (default: 6)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--subproc", action="store_true",
                        help="Use SubprocVecEnv (one process per env) instead of DummyVecEnv")
    parser.add_argument("--device", type=str, default="auto",
                        help="torch device: auto | cuda | cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO minibatch size (default: 128)")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout steps per env (default: 1024)")
    parser.add_argument("--tag", type=str, default="", help="Suffix for saved model filenames")
    args = parser.parse_args()

    print("=" * 70)
    print("  AMR PPO Training from Scratch: Webots Apartment Environment")
    print("  - 50 Apartment Walls & Furniture Obstacles")
    print("  - Dynamic Pedestrian Walking at 0.2 m/s")
    print("  - LDS-01 36-ray LiDAR & Waypoint Tracking")
    print("=" * 70)

    num_envs = args.envs
    vec_env_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec_env = make_vec_env(ApartmentGymEnv, n_envs=num_envs, vec_env_cls=vec_env_cls)

    eval_env = DummyVecEnv([make_env_fn()])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./rl/best_apartment_model",
        log_path="./rl/eval_logs",
        eval_freq=25000 // num_envs,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    total_timesteps = args.steps

    if args.device == "auto":
        training_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        training_device = args.device

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        device=training_device,
    )

    print(f"\n[DEVICE] Training on: {training_device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Starting training for {total_timesteps:,} steps across {num_envs} envs...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)
    elapsed = time.time() - t0

    print(f"\n[DONE] Training complete in {elapsed/60:.2f} minutes ({total_timesteps/elapsed:.1f} steps/s)!")

    # Save final model
    tag = f"_{args.tag}" if args.tag else ""
    final_zip_path = os.path.join(os.path.dirname(__file__), f"amr_apartment_ppo_model{tag}.zip")
    model.save(final_zip_path)
    print(f"Saved Stable-Baselines3 model to: {final_zip_path}")

    # Also update amr_ppo_model.zip so amr_controller picks it up immediately.
    # Skipped when --tag is given, so experimental runs do not clobber the deployed model.
    if not args.tag:
        default_zip_path = os.path.join(os.path.dirname(__file__), "amr_ppo_model.zip")
        model.save(default_zip_path)
        print(f"Updated default model: {default_zip_path}")

    # Export pure PyTorch policy
    policy_net = model.policy
    policy_pt_path = os.path.join(os.path.dirname(__file__), f"amr_apartment_policy{tag}.pt")
    torch.save(policy_net.state_dict(), policy_pt_path)
    print(f"Saved PyTorch weights to: {policy_pt_path}")


if __name__ == "__main__":
    main()
