"""Real-time 2D Occupancy Grid SLAM / Mapper for AMR in Webots.

- Uses wheel encoder odometry + 360-degree LDS-01 LiDAR.
- Continuously maps free space and static/dynamic obstacles on a 5cm grid.
- Records robot trajectory and rescue locations.
- Exports real-time and final occupancy grid map images.
"""

import math
import os
from typing import List, Tuple

import cv2
import numpy as np


class OccupancyGridMapper:
    def __init__(
        self,
        min_x: float = -13.5,
        max_x: float = 1.5,
        min_y: float = -14.5,
        max_y: float = 1.5,
        resolution: float = 0.05,  # 5cm grid resolution
    ):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.res = resolution

        self.nx = int(math.ceil((max_x - min_x) / resolution))
        self.ny = int(math.ceil((max_y - min_y) / resolution))

        # Log-odds occupancy grid: 0 = unknown, <0 = free, >0 = occupied
        # Initialized to 0.0 (unknown)
        self.log_odds = np.zeros((self.nx, self.ny), dtype=np.float32)

        self.trajectory: List[Tuple[float, float]] = []
        self.rescue_points: List[Tuple[float, float]] = []

        # Log-odds update constants
        self.l_occ = 0.85
        self.l_free = -0.40
        self.l_max = 5.0
        self.l_min = -5.0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.min_x) / self.res)
        gy = int((y - self.min_y) / self.res)
        return max(0, min(self.nx - 1, gx)), max(0, min(self.ny - 1, gy))

    def update(self, rx: float, ry: float, r_theta: float, lidar_ranges: list):
        """Update occupancy grid with current pose and LiDAR scan."""
        self.trajectory.append((rx, ry))
        gx0, gy0 = self.world_to_grid(rx, ry)

        count = len(lidar_ranges)
        if count == 0:
            return

        # Sample every 2nd or 4th ray for speed and consistency
        step = 1 if count <= 72 else 2
        for i in range(0, count, step):
            r = lidar_ranges[i]
            if math.isnan(r) or math.isinf(r) or r <= 0.12:
                continue

            # LDS-01 ray angle convention:
            # index 0 is Back, count//4 is Left, count//2 is Front, 3*count//4 is Right
            ray_angle = r_theta + (math.pi - i * (2.0 * math.pi / count))

            is_hit = r < 3.45
            valid_r = min(3.45, r)

            hit_x = rx + valid_r * math.cos(ray_angle)
            hit_y = ry + valid_r * math.sin(ray_angle)
            gx1, gy1 = self.world_to_grid(hit_x, hit_y)

            # Bresenham line raytracing
            self._raytrace(gx0, gy0, gx1, gy1, is_hit)

    def _raytrace(self, x0: int, y0: int, x1: int, y1: int, is_hit: bool):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cur_x, cur_y = x0, y0
        while True:
            if cur_x == x1 and cur_y == y1:
                if is_hit:
                    self.log_odds[cur_x, cur_y] = min(
                        self.l_max, self.log_odds[cur_x, cur_y] + self.l_occ
                    )
                break

            # Free space along ray
            self.log_odds[cur_x, cur_y] = max(
                self.l_min, self.log_odds[cur_x, cur_y] + self.l_free
            )

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cur_x += sx
            if e2 < dx:
                err += dx
                cur_y += sy

    def mark_rescue(self, x: float, y: float):
        self.rescue_points.append((x, y))

    def export_map_image(self, save_path: str = "slam_map.png") -> np.ndarray:
        """Render the occupancy grid map into an OpenCV BGR image (Unknown=Gray, Free=White, Occupied=Black)."""
        try:
            # Convert log odds to probability in [0, 1]
            prob = 1.0 / (1.0 + np.exp(-self.log_odds))
            # Image in grayscale: 0=black (wall), 255=white (free), 127=unknown
            img = np.full((self.nx, self.ny), 127, dtype=np.uint8)
            img[prob < 0.35] = 255
            img[prob > 0.65] = 0

            # Rotate/flip so X is horizontal (right) and Y is vertical (up)
            img = np.flipud(img.T)
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Draw trajectory in bright blue
            if len(self.trajectory) > 1:
                pts = []
                for tx, ty in self.trajectory[::5]:
                    gx, gy = self.world_to_grid(tx, ty)
                    pts.append([gx, self.ny - 1 - gy])
                cv2.polylines(
                    bgr, [np.array(pts, dtype=np.int32)], isClosed=False, color=(255, 120, 0), thickness=2, lineType=cv2.LINE_AA
                )

            # Draw rescue points as red circles with yellow border
            for rx, ry in self.rescue_points:
                gx, gy = self.world_to_grid(rx, ry)
                px, py = gx, self.ny - 1 - gy
                cv2.circle(bgr, (px, py), 7, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(bgr, (px, py), 9, (0, 240, 255), 2, lineType=cv2.LINE_AA)

            # Draw title text banner
            cv2.putText(
                bgr, f"PNU AMR SLAM Map | Rescued: {len(self.rescue_points)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, lineType=cv2.LINE_AA
            )

            cv2.imwrite(save_path, bgr)
            return bgr
        except Exception as e:
            print(f"[SLAM] Map export warning: {e}")
            return None
