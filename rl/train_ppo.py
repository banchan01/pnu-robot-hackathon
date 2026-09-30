"""PPO training script for AMR Navigation with Domain Randomization.

Trains a PPO policy on AMRFastEnv using Stable-Baselines3.
"""

import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.fast_env import AMRFastEnv


class TrainingProgressCallback(BaseCallback):
    """Prints training progress periodically."""

    def __init__(self, check_freq: int = 20480, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.n_calls / elapsed if elapsed > 0 else 0
            print(
                f"[PPO Train] Steps: {self.n_calls:,} | Elapsed: {elapsed:.1f}s | Speed: {fps:.0f} steps/s"
            )
        return True


def evaluate_policy(env: AMRFastEnv, model: PPO, num_episodes: int = 25) -> dict:
    """Evaluates the policy across diverse random obstacle environments."""
    successes = 0
    collisions = 0
    timeouts = 0
    total_rewards = []

    print(f"\nEvaluating policy over {num_episodes} random obstacle layouts...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        res = info.get("result", "unknown")
        if res == "success":
            successes += 1
        elif res == "collision":
            collisions += 1
        else:
            timeouts += 1

        total_rewards.append(ep_reward)

    success_rate = (successes / num_episodes) * 100.0
    collision_rate = (collisions / num_episodes) * 100.0
    timeout_rate = (timeouts / num_episodes) * 100.0
    avg_reward = float(np.mean(total_rewards))

    print("=" * 60)
    print(f"Evaluation Results ({num_episodes} Episodes):")
    print(f"  Success Rate   : {success_rate:.1f}% ({successes}/{num_episodes})")
    print(f"  Collision Rate : {collision_rate:.1f}% ({collisions}/{num_episodes})")
    print(f"  Timeout Rate   : {timeout_rate:.1f}% ({timeouts}/{num_episodes})")
    print(f"  Average Reward : {avg_reward:.2f}")
    print("=" * 60)

    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_reward": avg_reward,
    }


def main():
    total_timesteps = 1_000_000
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "amr_ppo_model.zip")
    pt_path = os.path.join(model_dir, "amr_policy.pt")

    print("=" * 60)
    print("Starting AMR PPO Reinforcement Learning Training (1 Million Steps)")
    print(f"Target Timesteps: {total_timesteps:,}")
    print(f"Domain: 6x6m Arena with 5+ Random Obstacles & Walls")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print("=" * 60)

    env = AMRFastEnv(num_random_obstacles=5)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    callback = TrainingProgressCallback(check_freq=51200)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - t0
    print(f"\n[DONE] Training finished in {training_time:.1f}s ({total_timesteps/training_time:.0f} steps/s)!")

    # Save SB3 model
    model.save(model_path)
    print(f"[SAVE] Stable-Baselines3 model saved to: {model_path}")

    # Export pure PyTorch policy
    torch.save(model.policy.state_dict(), pt_path)
    print(f"[SAVE] Standalone PyTorch policy saved to: {pt_path}")

    # Evaluate trained policy
    evaluate_policy(env, model, num_episodes=30)


if __name__ == "__main__":
    main()
