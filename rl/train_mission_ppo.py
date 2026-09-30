"""Time-budgeted PPO training for the fixed apartment mission (start -> apple A -> apple B -> start).

Usage (Windows, from repo root, inside the venv):
    python rl\\train_mission_ppo.py --minutes 13 --envs 14 --tag win1

- Stops on a wall-clock budget (default 13 min) and always saves, so it fits a 15 min slot.
- Warm-starts from rl/amr_apartment_ppo_model.zip (3M-step teammate model) when present.
- Observation/action interface is identical to amr_controller.predict_rl_speeds, so the
  result can replace rl/amr_ppo_model.zip directly (use --deploy).
"""

import argparse
import os
import shutil
import sys
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.mission_env import MissionEnv, make_eval_env  # noqa: E402

RL_DIR = os.path.dirname(os.path.abspath(__file__))


class TimeBudgetCallback(BaseCallback):
    """Stop training once the wall-clock budget is used up."""

    def __init__(self, minutes: float, verbose: int = 1):
        super().__init__(verbose)
        self.budget_s = minutes * 60.0
        self.t0 = None

    def _on_training_start(self) -> None:
        self.t0 = time.time()

    def _on_step(self) -> bool:
        elapsed = time.time() - self.t0
        if elapsed >= self.budget_s:
            if self.verbose:
                print(f"\n[TIME] Budget of {self.budget_s/60:.1f} min reached after {self.num_timesteps:,} steps. Stopping.")
            return False
        return True


class ProgressCallback(BaseCallback):
    """Print throughput and rolling success/collision stats every rollout."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.t0 = time.time()
        self.episodes = 0
        self.successes = 0
        self.wall = 0
        self.ped = 0
        self.legs = 0
        self.last_print = time.time()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes += 1
                self.successes += int(info.get("is_success", False))
                self.wall += int(info.get("wall_collision", False))
                self.ped += int(info.get("ped_collision", False))
                self.legs += int(info.get("legs_done", 0))
        return True

    def _on_rollout_end(self) -> None:
        if self.episodes == 0 or time.time() - self.last_print < 20.0:
            return
        self.last_print = time.time()
        elapsed = time.time() - self.t0
        sps = self.num_timesteps / max(1e-6, elapsed)
        print(
            f"[{elapsed/60:5.1f} min] steps={self.num_timesteps:>10,} ({sps:,.0f} sps) "
            f"episodes={self.episodes} success={100*self.successes/self.episodes:5.1f}% "
            f"wall={100*self.wall/self.episodes:5.1f}% ped={100*self.ped/self.episodes:5.1f}% "
            f"legs/ep={self.legs/self.episodes:.2f}"
        )
        self.episodes = self.successes = self.wall = self.ped = self.legs = 0


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the fixed apartment mission with a time budget")
    parser.add_argument("--minutes", type=float, default=13.0, help="Wall-clock training budget in minutes (default 13)")
    parser.add_argument("--max-steps", type=int, default=50_000_000, help="Upper bound on timesteps (budget usually hits first)")
    parser.add_argument("--envs", type=int, default=max(2, (os.cpu_count() or 4) - 2), help="Parallel envs (default: cores-2)")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--warm-start", type=str, default=os.path.join(RL_DIR, "amr_apartment_ppo_model.zip"),
                        help="Existing SB3 zip to continue from ('' to train from scratch)")
    parser.add_argument("--tag", type=str, default="mission", help="Suffix for saved files")
    parser.add_argument("--deploy", action="store_true",
                        help="Also overwrite rl/amr_ppo_model.zip (the file amr_controller loads)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Full-mission eval episodes after training")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "-"

    print("=" * 72)
    print("  Fixed Apartment Mission PPO: START -> RED APPLE A -> RED APPLE B -> START")
    print(f"  device={device} ({gpu_name})  envs={args.envs}  budget={args.minutes} min")
    print("=" * 72)

    vec_cls = DummyVecEnv if args.no_subproc else SubprocVecEnv
    vec_env = make_vec_env(MissionEnv, n_envs=args.envs, vec_env_cls=vec_cls,
                           env_kwargs=dict(random_leg_start=True, randomize_pedestrian=True))
    eval_env = DummyVecEnv([make_eval_env])

    save_dir = os.path.join(RL_DIR, f"best_{args.tag}_model")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(RL_DIR, "eval_logs", args.tag),
        eval_freq=max(1, 60_000 // args.envs),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )
    callbacks = [TimeBudgetCallback(args.minutes), ProgressCallback(), eval_cb]

    ppo_kwargs = dict(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=0,
        device=device,
    )

    if args.warm_start and os.path.exists(args.warm_start):
        print(f"[INIT] Warm start from {args.warm_start}")
        model = PPO.load(args.warm_start, env=vec_env, **ppo_kwargs)  # kwargs override saved hyper-params
    else:
        print("[INIT] Training from scratch (256x256 Tanh MLP)")
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh),
            **ppo_kwargs,
        )

    t0 = time.time()
    model.learn(total_timesteps=args.max_steps, callback=callbacks, progress_bar=False, reset_num_timesteps=True)
    elapsed = time.time() - t0
    print(f"\n[DONE] {model.num_timesteps:,} steps in {elapsed/60:.1f} min ({model.num_timesteps/elapsed:,.0f} steps/s)")

    final_zip = os.path.join(RL_DIR, f"amr_{args.tag}_ppo_model.zip")
    model.save(final_zip)
    torch.save(model.policy.state_dict(), os.path.join(RL_DIR, f"amr_{args.tag}_policy.pt"))
    print(f"[SAVE] {final_zip}")

    # Pick the better of final vs. best-on-eval, then run a full-mission evaluation
    best_zip = os.path.join(save_dir, "best_model.zip")
    from rl.eval_mission import evaluate  # noqa: E402

    candidates = [("final", final_zip)] + ([("best", best_zip)] if os.path.exists(best_zip) else [])
    results = []
    for name, path in candidates:
        m = PPO.load(path, device="cpu")
        stats = evaluate(m, episodes=args.eval_episodes, verbose=False)
        print(f"[EVAL:{name}] success={stats['success_rate']*100:.0f}% legs/ep={stats['avg_legs']:.2f} "
              f"wall={stats['wall_rate']*100:.0f}% ped={stats['ped_rate']*100:.0f}% steps/ep={stats['avg_steps']:.0f}")
        results.append((stats["success_rate"], stats["avg_legs"], name, path))
    results.sort(reverse=True)
    _, _, chosen_name, chosen_path = results[0]
    print(f"[PICK] {chosen_name} -> {chosen_path}")

    if args.deploy:
        deploy_zip = os.path.join(RL_DIR, "amr_ppo_model.zip")
        shutil.copyfile(chosen_path, deploy_zip)
        print(f"[DEPLOY] copied to {deploy_zip} (amr_controller loads this file)")
    else:
        print("[INFO] Not deployed. Re-run with --deploy, or copy the chosen zip to rl/amr_ppo_model.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
