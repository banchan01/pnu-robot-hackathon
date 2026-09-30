"""Apartment Environment for Reinforcement Learning (PPO) Navigation.

Faithfully replicates Webots worlds/apartment.wbt:
- 50 static boxes (all 27 walls + furniture bounding boxes)
- LDS-01 36-ray 360-degree LiDAR raycasting
- Moving Pedestrian walking along the exact 25-waypoint trajectory at 0.2 m/s
- Differential drive kinematics matching TurtleBot3 Burger
- A* waypoint subgoals guiding the robot through doors and corridors
"""

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

LIDAR_SECTORS = 36
LIDAR_MAX_RANGE = 3.5
MAX_LINEAR_SPEED = 0.22   # m/s (TurtleBot3 Burger max speed)
MAX_ANGULAR_SPEED = 2.0   # rad/s
DT = 0.064                # simulation step in seconds
ROBOT_RADIUS = 0.105      # m (TurtleBot3 Burger collision radius)
PEDESTRIAN_RADIUS = 0.25  # m (human collision radius)


class ApartmentGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layout_path: Optional[str] = None):
        super().__init__()

        if layout_path is None:
            layout_path = os.path.join(os.path.dirname(__file__), "apartment_layout.json")

        with open(layout_path, "r") as f:
            layout = json.load(f)

        # 1. Parse static boxes (walls + furniture)
        boxes_data = layout["boxes"]
        self.num_boxes = len(boxes_data)
        self.box_mins = np.empty((self.num_boxes, 2), dtype=np.float32)
        self.box_maxs = np.empty((self.num_boxes, 2), dtype=np.float32)
        for i, b in enumerate(boxes_data):
            self.box_mins[i, 0] = b["min_x"]
            self.box_maxs[i, 0] = b["max_x"]
            self.box_mins[i, 1] = b["min_y"]
            self.box_maxs[i, 1] = b["max_y"]

        # 2. Parse pedestrian trajectory
        self.ped_waypoints = [tuple(p) for p in layout.get("pedestrian_trajectory", [])]
        self.ped_speed = layout.get("pedestrian_speed", 0.2)
        self.ped_x = -5.24
        self.ped_y = -3.83
        self.ped_seg_idx = 0
        self.ped_seg_progress = 0.0

        # Precompute total trajectory segments
        self.ped_segments = []
        for i in range(len(self.ped_waypoints)):
            p1 = self.ped_waypoints[i]
            p2 = self.ped_waypoints[(i + 1) % len(self.ped_waypoints)]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self.ped_segments.append((p1, p2, max(0.01, dist)))

        # 3. Action and Observation spaces
        # Action: [v_norm, w_norm] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 36 lidar ranges + subgoal_dist + sin(bearing) + cos(bearing) + start_dist = 40
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(LIDAR_SECTORS + 4,), dtype=np.float32
        )

        # 4. A* Grid Planner on the apartment
        self.grid_res = 0.10
        self.grid_min_x, self.grid_max_x = -13.0, 1.0
        self.grid_min_y, self.grid_max_y = -14.0, 1.0
        self.nx = int((self.grid_max_x - self.grid_min_x) / self.grid_res)
        self.ny = int((self.grid_max_y - self.grid_min_y) / self.grid_res)
        self._build_grid(margin=0.14)

        # Candidate goals / rooms for domain randomization
        self.apple_positions = [tuple(a["pos"][:2]) for a in layout.get("apples", [])]
        self.room_keypoints = [
            (-0.3, -7.5),   # Entrance
            (-1.5, -7.5),   # Hallway
            (-5.0, -7.5),   # Central corridor
            (-5.0, -4.0),   # North corridor
            (-2.5, -2.0),   # North room
            (-8.0, -5.0),   # Living room
            (-11.5, -3.0),  # Kitchen / West room
            (-5.0, -9.5),   # South corridor
            (-5.5, -11.0),  # South bedroom
            (-8.0, -11.0),  # South-west room
        ]

        # Robot state
        self.robot_x = -0.3
        self.robot_y = -7.5
        self.robot_theta = math.pi
        self.start_x = -0.3
        self.start_y = -7.5
        self.waypoints: List[Tuple[float, float]] = []
        self.wp_idx = 0
        self.step_count = 0
        self.max_steps = 450
        self.prev_wp_dist = 0.0

    def _build_grid(self, margin=0.14):
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)
        for i in range(self.num_boxes):
            bx0 = self.box_mins[i, 0] - margin
            bx1 = self.box_maxs[i, 0] + margin
            by0 = self.box_mins[i, 1] - margin
            by1 = self.box_maxs[i, 1] + margin

            gx0 = max(0, int((bx0 - self.grid_min_x) / self.grid_res))
            gx1 = min(self.nx - 1, int((bx1 - self.grid_min_x) / self.grid_res))
            gy0 = max(0, int((by0 - self.grid_min_y) / self.grid_res))
            gy1 = min(self.ny - 1, int((by1 - self.grid_min_y) / self.grid_res))
            self.grid[gx0 : gx1 + 1, gy0 : gy1 + 1] = True

    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.grid_min_x) / self.grid_res)
        gy = int((y - self.grid_min_y) / self.grid_res)
        return max(0, min(self.nx - 1, gx)), max(0, min(self.ny - 1, gy))

    def _find_nearest_free(self, gx: int, gy: int, max_r: int = 5) -> Tuple[int, int]:
        if not self.grid[gx, gy]:
            return gx, gy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx_c, ny_c = gx + dx, gy + dy
                    if 0 <= nx_c < self.nx and 0 <= ny_c < self.ny and not self.grid[nx_c, ny_c]:
                        return nx_c, ny_c
        return gx, gy

    def _plan_a_star(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        import heapq

        s_gx, s_gy = self._find_nearest_free(*self._to_grid(*start_pos))
        g_gx, g_gy = self._find_nearest_free(*self._to_grid(*goal_pos))

        open_set = []
        heapq.heappush(open_set, (0, (s_gx, s_gy)))
        came_from = {}
        g_score = {(s_gx, s_gy): 0.0}

        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == (g_gx, g_gy):
                # Reconstruct path
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path = path[::-1]

                # Convert to world coordinates and subsample waypoints (every 4-5 cells = ~0.5m)
                wps = []
                step = 4
                for idx in range(0, len(path), step):
                    c = path[idx]
                    wx = self.grid_min_x + (c[0] + 0.5) * self.grid_res
                    wy = self.grid_min_y + (c[1] + 0.5) * self.grid_res
                    wps.append((wx, wy))
                if path and (len(path) - 1) % step != 0:
                    c = path[-1]
                    wps.append((self.grid_min_x + (c[0] + 0.5) * self.grid_res, self.grid_min_y + (c[1] + 0.5) * self.grid_res))
                return wps

            cur_g = g_score[cur]
            for d, c in zip(dirs, costs):
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if 0 <= nxt[0] < self.nx and 0 <= nxt[1] < self.ny and not self.grid[nxt[0], nxt[1]]:
                    tentative = cur_g + c
                    if nxt not in g_score or tentative < g_score[nxt]:
                        g_score[nxt] = tentative
                        h = math.hypot(nxt[0] - g_gx, nxt[1] - g_gy)
                        heapq.heappush(open_set, (tentative + h, nxt))
                        came_from[nxt] = cur

        return [goal_pos]

    def _step_pedestrian(self):
        """Advance the pedestrian along the trajectory."""
        if not self.ped_segments:
            return
        p1, p2, seg_len = self.ped_segments[self.ped_seg_idx]
        advance = self.ped_speed * DT
        self.ped_seg_progress += advance
        if self.ped_seg_progress >= seg_len:
            self.ped_seg_progress -= seg_len
            self.ped_seg_idx = (self.ped_seg_idx + 1) % len(self.ped_segments)
            p1, p2, seg_len = self.ped_segments[self.ped_seg_idx]

        ratio = min(1.0, max(0.0, self.ped_seg_progress / seg_len))
        self.ped_x = p1[0] + ratio * (p2[0] - p1[0])
        self.ped_y = p1[1] + ratio * (p2[1] - p1[1])

    def _cast_lidar(self) -> np.ndarray:
        """Vectorized 36-ray 360-degree LiDAR raycast matching Webots LDS-01."""
        # Ray angles: index 0 is Back (theta + pi), index 9 is Left (+pi/2),
        # index 18 is Front (theta), index 27 is Right (-pi/2)
        angles = np.empty(LIDAR_SECTORS, dtype=np.float32)
        for i in range(LIDAR_SECTORS):
            angles[i] = self.robot_theta + (math.pi - i * (2.0 * math.pi / LIDAR_SECTORS))

        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (36, 2)
        inv_d = 1.0 / np.where(np.abs(ray_dirs) < 1e-6, 1e-6, ray_dirs)  # (36, 2)

        ro = np.array([self.robot_x, self.robot_y], dtype=np.float32)  # (2,)

        # Slab intersection against all 50 boxes
        t1 = (self.box_mins[None, :, :] - ro[None, None, :]) * inv_d[:, None, :]  # (36, 50, 2)
        t2 = (self.box_maxs[None, :, :] - ro[None, None, :]) * inv_d[:, None, :]

        tmin = np.maximum(np.minimum(t1[:, :, 0], t2[:, :, 0]), np.minimum(t1[:, :, 1], t2[:, :, 1]))
        tmax = np.minimum(np.maximum(t1[:, :, 0], t2[:, :, 0]), np.maximum(t1[:, :, 1], t2[:, :, 1]))

        hit = (tmax >= tmin) & (tmax > 0.0)
        dists = np.where(hit & (tmin > 0.0), tmin, np.where(hit & (tmax > 0.0), 0.0, LIDAR_MAX_RANGE))
        lidar = np.min(dists, axis=1)  # (36,)

        # Raycast against moving pedestrian cylinder
        ped_vec = np.array([self.ped_x - self.robot_x, self.ped_y - self.robot_y], dtype=np.float32)
        # Dot product with ray directions
        proj = np.sum(ray_dirs * ped_vec[None, :], axis=1)  # (36,)
        d2 = np.sum(ped_vec**2) - proj**2
        r2 = PEDESTRIAN_RADIUS**2
        ped_hit = (proj > 0.0) & (d2 < r2) & (d2 >= 0.0)
        ped_dists = np.where(ped_hit, proj - np.sqrt(np.maximum(0.0, r2 - d2)), LIDAR_MAX_RANGE)

        lidar = np.minimum(lidar, ped_dists)
        return np.clip(lidar, 0.0, LIDAR_MAX_RANGE)

    def _check_collision(self) -> Tuple[bool, bool]:
        """Check if robot collided with static boxes or moving pedestrian."""
        rx, ry = self.robot_x, self.robot_y
        r = ROBOT_RADIUS

        # 1. Collision with static boxes
        box_coll = np.any(
            (rx + r > self.box_mins[:, 0])
            & (rx - r < self.box_maxs[:, 0])
            & (ry + r > self.box_mins[:, 1])
            & (ry - r < self.box_maxs[:, 1])
        )

        # 2. Collision with moving pedestrian
        dist_ped = math.hypot(rx - self.ped_x, ry - self.ped_y)
        ped_coll = dist_ped < (ROBOT_RADIUS + PEDESTRIAN_RADIUS)

        return bool(box_coll), bool(ped_coll)

    def _get_obs(self, lidar: np.ndarray) -> np.ndarray:
        if self.wp_idx < len(self.waypoints):
            target = self.waypoints[self.wp_idx]
        else:
            target = (self.robot_x, self.robot_y)

        dx = target[0] - self.robot_x
        dy = target[1] - self.robot_y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - self.robot_theta
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))

        start_dist = math.hypot(self.robot_x - self.start_x, self.robot_y - self.start_y)

        obs = np.empty(LIDAR_SECTORS + 4, dtype=np.float32)
        obs[:LIDAR_SECTORS] = lidar / LIDAR_MAX_RANGE
        obs[LIDAR_SECTORS] = min(1.0, dist / 6.0)
        obs[LIDAR_SECTORS + 1] = math.sin(bearing)
        obs[LIDAR_SECTORS + 2] = math.cos(bearing)
        obs[LIDAR_SECTORS + 3] = min(1.0, start_dist / 6.0)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Pick random start and goal from rooms/apples
        candidates = self.room_keypoints + self.apple_positions
        s_pos = random.choice(candidates)
        # Add slight jitter
        self.start_x = s_pos[0] + random.uniform(-0.15, 0.15)
        self.start_y = s_pos[1] + random.uniform(-0.15, 0.15)
        self.robot_x = self.start_x
        self.robot_y = self.start_y
        self.robot_theta = random.uniform(-math.pi, math.pi)

        # Pick different goal at least 2.5m away
        g_pos = s_pos
        for _ in range(50):
            cand = random.choice(candidates)
            if math.hypot(cand[0] - self.start_x, cand[1] - self.start_y) >= 2.5:
                g_pos = cand
                break

        # Randomize pedestrian position along trajectory
        if self.ped_segments:
            self.ped_seg_idx = random.randint(0, len(self.ped_segments) - 1)
            self.ped_seg_progress = random.uniform(0.0, self.ped_segments[self.ped_seg_idx][2])
            self._step_pedestrian()

        # Plan A* waypoints
        self.waypoints = self._plan_a_star((self.robot_x, self.robot_y), g_pos)
        self.wp_idx = 1 if len(self.waypoints) > 1 else 0

        cur_wp = self.waypoints[self.wp_idx]
        self.prev_wp_dist = math.hypot(cur_wp[0] - self.robot_x, cur_wp[1] - self.robot_y)

        lidar = self._cast_lidar()
        return self._get_obs(lidar), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # Action: [v_norm, w_norm]
        # v in [0.05, MAX_LINEAR_SPEED], w in [-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED]
        v = (float(action[0]) + 1.0) / 2.0 * (MAX_LINEAR_SPEED - 0.05) + 0.05
        w = float(action[1]) * MAX_ANGULAR_SPEED

        # Advance robot kinematics
        self.robot_theta += w * DT
        self.robot_theta = math.atan2(math.sin(self.robot_theta), math.cos(self.robot_theta))
        self.robot_x += v * math.cos(self.robot_theta) * DT
        self.robot_y += v * math.sin(self.robot_theta) * DT

        # Advance moving pedestrian
        self._step_pedestrian()

        # Check collisions
        wall_coll, ped_coll = self._check_collision()

        lidar = self._cast_lidar()

        # Check waypoint progress
        cur_wp = self.waypoints[self.wp_idx]
        cur_wp_dist = math.hypot(cur_wp[0] - self.robot_x, cur_wp[1] - self.robot_y)
        progress = self.prev_wp_dist - cur_wp_dist
        self.prev_wp_dist = cur_wp_dist

        reward = progress * 4.0 - 0.02
        # Angular spin penalty
        reward -= 0.03 * (w / MAX_ANGULAR_SPEED) ** 2

        # Pedestrian proximity penalty (keep distance)
        dist_ped = math.hypot(self.robot_x - self.ped_x, self.robot_y - self.ped_y)
        if dist_ped < 0.6:
            reward -= 0.2 * (0.6 - dist_ped)

        # Advance waypoint if reached
        if cur_wp_dist < 0.35:
            if self.wp_idx < len(self.waypoints) - 1:
                self.wp_idx += 1
                cur_wp = self.waypoints[self.wp_idx]
                self.prev_wp_dist = math.hypot(cur_wp[0] - self.robot_x, cur_wp[1] - self.robot_y)
                reward += 15.0

        terminated = False
        truncated = self.step_count >= self.max_steps

        if wall_coll:
            reward = -100.0
            terminated = True
        elif ped_coll:
            reward = -150.0
            terminated = True
        elif self.wp_idx >= len(self.waypoints) - 1 and cur_wp_dist < 0.30:
            reward += 100.0
            terminated = True

        obs = self._get_obs(lidar)
        info = {
            "is_success": (not wall_coll and not ped_coll and self.wp_idx >= len(self.waypoints) - 1 and cur_wp_dist < 0.30),
            "ped_collision": ped_coll,
            "wall_collision": wall_coll,
        }
        return obs, reward, terminated, truncated, info
