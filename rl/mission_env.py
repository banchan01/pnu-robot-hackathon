"""Fixed-mission Apartment environment: start -> red apple 1 -> red apple 2 -> back to start.

Built on ApartmentGymEnv (same 50-box layout, LDS-01 36-ray LiDAR, moving pedestrian,
differential-drive kinematics) but the episode is the actual competition mission:

  leg 0: START (-0.30, -7.50)  -> RED APPLE A (-5.34, -10.54)
  leg 1: RED APPLE A           -> RED APPLE B (-12.02, -3.02)
  leg 2: RED APPLE B           -> START

Each leg is guided by A* waypoints exactly like the deployed amr_controller, and the
observation is the same 40-dim vector the controller builds, so a model trained here is a
drop-in replacement for rl/amr_ppo_model.zip.

Training trick: with `random_leg_start=True` an episode may begin at the start of leg 1 or
leg 2 (small pose jitter), so all three legs are learned in parallel instead of waiting
for the policy to solve leg 0 first. Evaluation uses the full mission from the start pad.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rl.apartment_env import (
    ApartmentGymEnv,
    DT,
    LIDAR_SECTORS,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    ROBOT_RADIUS,
)

START_POSE = (-0.30, -7.50, math.pi)
RED_APPLES: List[Tuple[float, float]] = [(-5.34, -10.54), (-12.02, -3.02)]

APPLE_REACH_DIST = 0.45     # controller counts a rescue when within 0.45 m of the apple
HOME_REACH_DIST = 0.30      # mission complete when back within 0.30 m of the start pad
WAYPOINT_REACH_DIST = 0.40  # same threshold as amr_controller

# Linear speed mapping for action[0] in [-1, 1]. The teammate controller used [0.10, 0.22]
# (never stops); with 0.0 the robot can stop and wait for the pedestrian in narrow corridors.
# The trainer writes the value actually used to rl/policy_meta.json for the controller.
MIN_LINEAR_SPEED = 0.0


class MissionEnv(ApartmentGymEnv):
    """Full search-and-rescue mission in the fixed apartment world."""

    def __init__(
        self,
        layout_path: Optional[str] = None,
        random_leg_start: bool = True,
        randomize_pedestrian: bool = True,
        max_steps: int = 4500,
        pose_jitter: float = 0.10,
        min_speed: float = MIN_LINEAR_SPEED,
    ):
        super().__init__(layout_path)
        self.min_speed = min_speed
        self.random_leg_start = random_leg_start
        self.randomize_pedestrian = randomize_pedestrian
        self.max_steps = max_steps
        self.pose_jitter = pose_jitter

        self.goals: List[Tuple[float, float]] = RED_APPLES + [START_POSE[:2]]
        self.leg = 0
        self.legs_done = 0
        self.start_x, self.start_y = START_POSE[0], START_POSE[1]

        # Leg start poses: start pad, then each apple (heading is randomized at reset)
        self.leg_start_pos = [START_POSE[:2]] + RED_APPLES

    # ------------------------------------------------------------------ helpers
    def _plan_leg(self):
        goal = self.goals[self.leg]
        # A* ends on the nearest free cell (apple A sits inside the inflated margin);
        # leg completion is judged on the true goal distance instead.
        self.waypoints = self._plan_a_star((self.robot_x, self.robot_y), goal)
        self.wp_idx = 1 if len(self.waypoints) > 1 else 0
        cur_wp = self.waypoints[self.wp_idx]
        self.prev_wp_dist = math.hypot(cur_wp[0] - self.robot_x, cur_wp[1] - self.robot_y)

    def _goal_reach_dist(self) -> float:
        return HOME_REACH_DIST if self.leg == len(self.goals) - 1 else APPLE_REACH_DIST

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Skip ApartmentGymEnv.reset (random start/goal); do gym bookkeeping ourselves.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.step_count = 0
        self.legs_done = 0

        if self.random_leg_start:
            # Bias towards the full mission but cover the later legs as well
            self.leg = random.choices([0, 1, 2], weights=[0.4, 0.3, 0.3])[0]
        else:
            self.leg = 0

        sx, sy = self.leg_start_pos[self.leg]
        if self.leg == 0:
            self.robot_x, self.robot_y, self.robot_theta = START_POSE
            if self.random_leg_start:
                self.robot_x += random.uniform(-0.05, 0.05)
                self.robot_y += random.uniform(-0.05, 0.05)
                self.robot_theta += random.uniform(-0.15, 0.15)
        else:
            # Apples can sit right next to furniture, so spawn on the nearest free grid
            # cell (where the previous leg actually ends) and reject colliding poses.
            gx, gy = self._find_nearest_free(*self._to_grid(sx, sy))
            fx = self.grid_min_x + (gx + 0.5) * self.grid_res
            fy = self.grid_min_y + (gy + 0.5) * self.grid_res
            for _ in range(20):
                self.robot_x = fx + random.uniform(-self.pose_jitter, self.pose_jitter)
                self.robot_y = fy + random.uniform(-self.pose_jitter, self.pose_jitter)
                if not self._check_collision()[0]:
                    break
            else:
                self.robot_x, self.robot_y = fx, fy
            self.robot_theta = random.uniform(-math.pi, math.pi)

        # Pedestrian: random phase along its loop (its timing vs. the robot is not reliable)
        if self.ped_segments:
            if self.randomize_pedestrian:
                self.ped_seg_idx = random.randint(0, len(self.ped_segments) - 1)
                self.ped_seg_progress = random.uniform(0.0, self.ped_segments[self.ped_seg_idx][2])
            else:
                self.ped_seg_idx = 0
                self.ped_seg_progress = 0.0
            self._step_pedestrian()

        self._plan_leg()
        lidar = self._cast_lidar()
        return self._get_obs(lidar), {"leg": self.leg}

    def step(self, action: np.ndarray):
        self.step_count += 1

        v = (float(action[0]) + 1.0) / 2.0 * (MAX_LINEAR_SPEED - self.min_speed) + self.min_speed
        w = float(action[1]) * MAX_ANGULAR_SPEED

        self.robot_theta += w * DT
        self.robot_theta = math.atan2(math.sin(self.robot_theta), math.cos(self.robot_theta))
        self.robot_x += v * math.cos(self.robot_theta) * DT
        self.robot_y += v * math.sin(self.robot_theta) * DT

        self._step_pedestrian()
        wall_coll, ped_coll = self._check_collision()
        lidar = self._cast_lidar()

        # Progress towards current waypoint
        cur_wp = self.waypoints[self.wp_idx]
        cur_wp_dist = math.hypot(cur_wp[0] - self.robot_x, cur_wp[1] - self.robot_y)
        progress = self.prev_wp_dist - cur_wp_dist
        self.prev_wp_dist = cur_wp_dist

        reward = progress * 4.0 - 0.02
        reward -= 0.03 * (w / MAX_ANGULAR_SPEED) ** 2

        # Keep clear of the pedestrian and of walls
        dist_ped = math.hypot(self.robot_x - self.ped_x, self.robot_y - self.ped_y)
        if dist_ped < 0.8:
            reward -= 0.3 * (0.8 - dist_ped)
        min_lidar = float(np.min(lidar))
        if min_lidar < 0.20:
            reward -= 0.1 * (0.20 - min_lidar) / 0.20

        # Intermediate waypoint reached
        last_wp = self.wp_idx >= len(self.waypoints) - 1
        if not last_wp and cur_wp_dist < WAYPOINT_REACH_DIST:
            self.wp_idx += 1
            cur_wp = self.waypoints[self.wp_idx]
            self.prev_wp_dist = math.hypot(cur_wp[0] - self.robot_x, cur_wp[1] - self.robot_y)
            reward += 5.0

        terminated = False
        mission_done = False
        leg_done = False

        if wall_coll:
            reward = -100.0
            terminated = True
        elif ped_coll:
            reward = -150.0
            terminated = True
        else:
            goal = self.goals[self.leg]
            goal_dist = math.hypot(goal[0] - self.robot_x, goal[1] - self.robot_y)
            if goal_dist < self._goal_reach_dist():
                leg_done = True
                self.legs_done += 1
                if self.leg == len(self.goals) - 1:
                    reward += 200.0
                    mission_done = True
                    terminated = True
                else:
                    reward += 100.0
                    self.leg += 1
                    self._plan_leg()

        truncated = (not terminated) and self.step_count >= self.max_steps

        obs = self._get_obs(lidar)
        info = {
            "is_success": mission_done,
            "leg": self.leg,
            "leg_done": leg_done,
            "legs_done": self.legs_done,
            "ped_collision": ped_coll,
            "wall_collision": wall_coll,
        }
        return obs, reward, terminated, truncated, info


def make_train_env(min_speed: float = MIN_LINEAR_SPEED) -> MissionEnv:
    return MissionEnv(random_leg_start=True, randomize_pedestrian=True, min_speed=min_speed)


def make_eval_env(min_speed: float = MIN_LINEAR_SPEED) -> MissionEnv:
    return MissionEnv(random_leg_start=False, randomize_pedestrian=True, min_speed=min_speed)
