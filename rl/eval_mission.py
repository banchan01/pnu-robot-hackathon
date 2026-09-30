"""Evaluate a PPO model on the full fixed mission and optionally plot trajectories.

    python rl/eval_mission.py --model rl/amr_mission_ppo_model.zip --episodes 10 --plot
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.mission_env import RED_APPLES, START_POSE, make_eval_env  # noqa: E402

RL_DIR = os.path.dirname(os.path.abspath(__file__))


def run_episode(model: PPO, env, deterministic: bool = True):
    obs, _ = env.reset()
    traj = [(env.robot_x, env.robot_y)]
    ped = [(env.ped_x, env.ped_y)]
    info = {}
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        traj.append((env.robot_x, env.robot_y))
        ped.append((env.ped_x, env.ped_y))
        steps += 1
        if terminated or truncated:
            break
    return traj, ped, info, steps


def evaluate(model: PPO, episodes: int = 10, verbose: bool = True) -> Dict[str, float]:
    env = make_eval_env()
    succ = wall = pedc = 0
    legs_total = 0
    steps_total = 0
    trajs: List[Tuple[List, Dict]] = []
    for ep in range(episodes):
        traj, ped, info, steps = run_episode(model, env)
        succ += int(info.get("is_success", False))
        wall += int(info.get("wall_collision", False))
        pedc += int(info.get("ped_collision", False))
        legs_total += int(info.get("legs_done", 0))
        steps_total += steps
        trajs.append((traj, info))
        if verbose:
            outcome = "SUCCESS" if info.get("is_success") else ("WALL" if info.get("wall_collision") else ("PED" if info.get("ped_collision") else "TIMEOUT"))
            print(f"  ep {ep+1:2d}: {outcome:8s} legs_done={info.get('legs_done',0)} steps={steps} ({steps*0.064:.0f}s sim)")
    stats = {
        "success_rate": succ / episodes,
        "wall_rate": wall / episodes,
        "ped_rate": pedc / episodes,
        "avg_legs": legs_total / episodes,
        "avg_steps": steps_total / episodes,
    }
    stats["_trajs"] = trajs  # type: ignore[assignment]
    stats["_env"] = env  # type: ignore[assignment]
    return stats


def plot(stats: Dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:  # pragma: no cover
        print(f"[PLOT] matplotlib not available ({e}); skipping plot")
        return
    env = stats["_env"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(env.num_boxes):
        x0, y0 = env.box_mins[i]
        x1, y1 = env.box_maxs[i]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#444444"))
    for traj, info in stats["_trajs"]:
        xs, ys = zip(*traj)
        color = "#2a9d8f" if info.get("is_success") else "#e76f51"
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.8)
    ax.plot(START_POSE[0], START_POSE[1], "bs", markersize=9, label="start")
    for k, (ax_, ay_) in enumerate(RED_APPLES):
        ax.plot(ax_, ay_, "ro", markersize=9, label=f"red apple {k+1}")
    ax.set_xlim(env.grid_min_x, env.grid_max_x)
    ax.set_ylim(env.grid_min_y, env.grid_max_y)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"Mission eval: success {stats['success_rate']*100:.0f}%  (green=success, red=fail)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[PLOT] saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(RL_DIR, "amr_mission_ppo_model.zip"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model}")
    model = PPO.load(args.model, device="cpu")
    stats = evaluate(model, episodes=args.episodes, verbose=True)
    print(f"\nsuccess={stats['success_rate']*100:.0f}%  legs/ep={stats['avg_legs']:.2f}  "
          f"wall={stats['wall_rate']*100:.0f}%  ped={stats['ped_rate']*100:.0f}%  steps/ep={stats['avg_steps']:.0f}")
    if args.plot:
        plot(stats, os.path.join(RL_DIR, "mission_eval.png"))


if __name__ == "__main__":
    main()
