"""Evaluation script to benchmark the trained AMR PPO model across diverse obstacle densities."""

import os
import sys

from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.fast_env import AMRFastEnv
from rl.train_ppo import evaluate_policy


def main():
    model_path = os.path.join(os.path.dirname(__file__), "amr_ppo_model.zip")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_ppo.py first!")
        return

    print("=" * 60)
    print("Loading trained PPO Model...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    print("=" * 60)

    for num_obs in [3, 5, 7]:
        print(f"\n>>> Stress Testing with {num_obs} Random Obstacles:")
        env = AMRFastEnv(num_random_obstacles=num_obs)
        evaluate_policy(env, model, num_episodes=20)


if __name__ == "__main__":
    main()
