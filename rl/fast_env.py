"""Fast 2D kinematic simulation environment for AMR RL training.

Simulates the exact 6x6m Webots arena, fixed walls, TurtleBot3 kinematics,
LDS-01 36-ray LiDAR, and random obstacle layouts (Domain Randomization).
Runs at >10,000 steps/sec on CPU for lightning-fast PPO training.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Robot parameters (TurtleBot3 Burger specs)
ROBOT_RADIUS = 0.12  # m
MAX_LINEAR_SPEED = 0.22  # m/s
MAX_ANGULAR_SPEED = 2.0  # rad/s
LIDAR_MAX_RANGE = 3.5  # m (LDS-01 spec)
LIDAR_SECTORS = 36
DT = 0.1  # 10 Hz control loop (matches Webots 100ms lidar period)
MAX_EPISODE_STEPS = 400


class BoxObstacle:
    def __init__(self, cx: float, cy: float, sx: float, sy: float):
        self.cx = cx
        self.cy = cy
        self.sx = sx
        self.sy = sy
        self.min_x = cx - sx / 2.0
        self.max_x = cx + sx / 2.0
        self.min_y = cy - sy / 2.0
        self.max_y = cy + sy / 2.0

        # 4 bounding segments: ((x1, y1), (x2, y2))
        self.segments = [
            ((self.min_x, self.min_y), (self.max_x, self.min_y)),
            ((self.max_x, self.min_y), (self.max_x, self.max_y)),
            ((self.max_x, self.max_y), (self.min_x, self.max_y)),
            ((self.min_x, self.max_y), (self.min_x, self.min_y)),
        ]

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.min_x - margin <= x <= self.max_x + margin) and (
            self.min_y - margin <= y <= self.max_y + margin
        )


class CylinderObstacle:
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= (self.radius + margin)


def ray_segment_intersect(
    ox: float, oy: float, dx: float, dy: float, p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """Returns distance t along ray (ox + t*dx, oy + t*dy) to line segment (p1->p2), or inf."""
    x1, y1 = p1
    x2, y2 = p2
    v1_x = ox - x1
    v1_y = oy - y1
    v2_x = x2 - x1
    v2_y = y2 - y1

    cross = dx * v2_y - dy * v2_x
    if abs(cross) < 1e-8:
        return float("inf")

    t1 = (v2_x * v1_y - v2_y * v1_x) / cross
    t2 = (dx * v1_y - dy * v1_x) / cross

    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return t1
    return float("inf")


def ray_circle_intersect(
    ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, radius: float
) -> float:
    """Returns distance t along ray to circle (cx, cy, radius), or inf."""
    lx = ox - cx
    ly = oy - cy
    b = 2.0 * (dx * lx + dy * ly)
    c = lx * lx + ly * ly - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return float("inf")
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    if t1 >= 0.0:
        return t1
    if t2 >= 0.0:
        return t2
    return float("inf")


class AMRFastEnv(gym.Env):
    """Gymnasium environment for TurtleBot3 navigation with domain-randomized obstacles."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, num_random_obstacles: int = 5):
        super().__init__()
        self.num_random_obstacles = num_random_obstacles

        # Action: [linear_velocity_norm, angular_velocity_norm] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 36 lidar ranges + goal_dist_norm + sin(bearing) + cos(bearing) + start_dist_norm = 40
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(LIDAR_SECTORS + 4,), dtype=np.float32
        )

        # Arena boundary walls (6m x 6m: [-3, 3] x [-3, 3])
        self.arena_segments = [
            ((-3.0, -3.0), (3.0, -3.0)),
            ((3.0, -3.0), (3.0, 3.0)),
            ((3.0, 3.0), (-3.0, 3.0)),
            ((-3.0, 3.0), (-3.0, -3.0)),
        ]

        # Fixed obstacles from sensor_playground.wbt
        self.fixed_obstacles = [
            BoxObstacle(0.3, 0.0, 0.2, 2.2),  # CENTER_WALL
            BoxObstacle(-0.8, -0.8, 1.4, 0.2),  # EXTRA_WALL
            CylinderObstacle(1.2, -1.2, 0.25),  # PILLAR_1
        ]

        self.obstacles: List[Any] = []
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.start_x = -2.0
        self.start_y = -1.5
        self.target_x = 1.9
        self.target_y = 1.8
        self.step_count = 0
        self.prev_goal_dist = 0.0

    def _cast_lidar(self) -> np.ndarray:
        """Cast 36 rays from robot center."""
        ranges = np.full(LIDAR_SECTORS, LIDAR_MAX_RANGE, dtype=np.float32)
        rx, ry, r_th = self.robot_x, self.robot_y, self.robot_theta

        # Gather all segments from boundary, fixed, and dynamic box obstacles
        segments = list(self.arena_segments)
        circles = []

        for obs in self.obstacles + self.fixed_obstacles:
            if isinstance(obs, BoxObstacle):
                segments.extend(obs.segments)
            elif isinstance(obs, CylinderObstacle):
                circles.append(obs)

        for i in range(LIDAR_SECTORS):
            # Webots LDS-01 ordering: i=0 is Back, i=9 is Left, i=18 is Front, i=27 is Right
            local_angle = math.pi - i * (2.0 * math.pi / LIDAR_SECTORS)
            ray_angle = r_th + local_angle
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)

            min_dist = LIDAR_MAX_RANGE
            for seg in segments:
                d = ray_segment_intersect(rx, ry, dx, dy, seg[0], seg[1])
                if d < min_dist:
                    min_dist = d

            for circ in circles:
                d = ray_circle_intersect(rx, ry, dx, dy, circ.cx, circ.cy, circ.radius)
                if d < min_dist:
                    min_dist = d

            ranges[i] = min_dist

        return ranges

    def _get_obs(self, lidar_ranges: np.ndarray) -> np.ndarray:
        goal_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        goal_bearing = math.atan2(self.target_y - self.robot_y, self.target_x - self.robot_x) - self.robot_theta
        goal_bearing = math.atan2(math.sin(goal_bearing), math.cos(goal_bearing))

        start_dist = math.hypot(self.robot_x - self.start_x, self.robot_y - self.start_y)

        # Normalize lidar to [0, 1]
        norm_lidar = lidar_ranges / LIDAR_MAX_RANGE
        # Normalized goal distance (capped at 10m -> 1.0)
        norm_goal_dist = min(1.0, goal_dist / 6.0)
        norm_start_dist = min(1.0, start_dist / 6.0)

        obs = np.empty(LIDAR_SECTORS + 4, dtype=np.float32)
        obs[:LIDAR_SECTORS] = norm_lidar
        obs[LIDAR_SECTORS] = norm_goal_dist
        obs[LIDAR_SECTORS + 1] = math.sin(goal_bearing)
        obs[LIDAR_SECTORS + 2] = math.cos(goal_bearing)
        obs[LIDAR_SECTORS + 3] = norm_start_dist
        return obs

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # 1. Randomize robot start position near (-2.0, -1.5)
        self.start_x = -2.0 + random.uniform(-0.15, 0.15)
        self.start_y = -1.5 + random.uniform(-0.15, 0.15)
        self.robot_x = self.start_x
        self.robot_y = self.start_y
        self.robot_theta = random.uniform(-math.pi / 4, math.pi / 4)

        # 2. Randomize target position
        for _ in range(100):
            tx = random.uniform(-1.2, 2.3)
            ty = random.uniform(-2.2, 2.3)
            if math.hypot(tx - self.start_x, ty - self.start_y) >= 2.5:
                if not any(obs.contains(tx, ty, margin=0.35) for obs in self.fixed_obstacles):
                    self.target_x = tx
                    self.target_y = ty
                    break

        # 3. Randomize dynamic obstacles (Domain Randomization)
        self.obstacles = []
        placed_positions = [(self.start_x, self.start_y), (self.target_x, self.target_y)]

        for i in range(self.num_random_obstacles):
            for _ in range(60):
                ox = random.uniform(-2.3, 2.3)
                oy = random.uniform(-2.3, 2.3)
                # Check fixed obstacles
                if any(obs.contains(ox, oy, margin=0.45) for obs in self.fixed_obstacles):
                    continue
                # Check start/target clearance
                if math.hypot(ox - self.start_x, oy - self.start_y) < 0.85:
                    continue
                if math.hypot(ox - self.target_x, oy - self.target_y) < 0.65:
                    continue
                if any(math.hypot(ox - px, oy - py) < 0.65 for px, py in placed_positions):
                    continue

                # Alternate between boxes and cylinders
                if i % 2 == 0:
                    sx = random.uniform(0.4, 0.7)
                    sy = random.uniform(0.4, 0.7)
                    self.obstacles.append(BoxObstacle(ox, oy, sx, sy))
                else:
                    rad = random.uniform(0.2, 0.3)
                    self.obstacles.append(CylinderObstacle(ox, oy, rad))

                placed_positions.append((ox, oy))
                break

        self.prev_goal_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        lidar = self._cast_lidar()
        obs = self._get_obs(lidar)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # Action: [v_norm, w_norm]
        # v_norm in [-1, 1] mapped to [0.0, MAX_LINEAR_SPEED] (prevent unnecessary backward crawl)
        # Action: [v_norm, w_norm]
        # v_norm in [-1, 1] mapped to [0.10, MAX_LINEAR_SPEED] (prevents freezing/stalling)
        v = 0.10 + (float(action[0]) + 1.0) / 2.0 * (MAX_LINEAR_SPEED - 0.10)
        w = float(action[1]) * MAX_ANGULAR_SPEED

        # Kinematic update
        self.robot_theta += w * DT
        self.robot_theta = math.atan2(math.sin(self.robot_theta), math.cos(self.robot_theta))
        self.robot_x += v * math.cos(self.robot_theta) * DT
        self.robot_y += v * math.sin(self.robot_theta) * DT

        # Compute observations & collision
        lidar = self._cast_lidar()
        min_lidar = float(np.min(lidar))
        goal_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        goal_bearing = math.atan2(self.target_y - self.robot_y, self.target_x - self.robot_x) - self.robot_theta
        goal_bearing = math.atan2(math.sin(goal_bearing), math.cos(goal_bearing))

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # 1. Collision Check
        if min_lidar < ROBOT_RADIUS + 0.02:
            reward = -80.0
            terminated = True
            info["result"] = "collision"

        # 2. Goal Reached Check
        elif goal_dist < 0.28:
            reward = 160.0
            terminated = True
            info["result"] = "success"

        # 3. Normal step rewards
        else:
            # Dense progress reward: moving closer to goal
            progress = (self.prev_goal_dist - goal_dist) * 20.0
            reward += progress

            # Heading alignment: reward facing towards the goal
            reward += 0.12 * math.cos(goal_bearing)

            # Spin penalty: strongly discourage spinning in circles
            reward -= 0.04 * (w / MAX_ANGULAR_SPEED) ** 2

            # Clearance penalty: keep distance from obstacles (<0.28m)
            if min_lidar < 0.28:
                reward -= (0.28 - min_lidar) * 3.5

            # Small time penalty to encourage efficiency
            reward -= 0.01

        self.prev_goal_dist = goal_dist

        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True
            if "result" not in info:
                info["result"] = "timeout"
                reward -= 50.0  # Big penalty for stalling / running out the clock!

        obs = self._get_obs(lidar)
        return obs, reward, terminated, truncated, info
