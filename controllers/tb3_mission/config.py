"""상수와 미션 설정 로더.

로봇 물리 파라미터는 TurtleBot3 Burger PROTO에서, LiDAR 사양은 RobotisLds01 PROTO에서,
카메라 사양은 apartment.wbt의 extensionSlot 정의에서 가져왔다.
조정 가능한 값은 전부 mission.json에 두고 코드 상수는 손대지 않는다.
"""
import json
import math
import os

import numpy as np

CTRL_DIR = os.path.dirname(os.path.abspath(__file__))
MISSION_FILE = os.path.join(CTRL_DIR, "mission.json")
OUTPUT_DIR = os.environ.get("TB3_OUTPUT_DIR") or os.path.join(CTRL_DIR, "output")   # 테스트 간 충돌 방지용 override
# 명령 파일: 기본은 컨트롤러 폴더의 commands.txt (웹 콘솔이 쓰는 위치).
# 격리된 헤드리스 테스트(TB3_OUTPUT_DIR 지정)에서는 그 폴더 안의 commands.txt를 읽어 서로 간섭하지 않게 한다.
COMMAND_FILE = os.environ.get("TB3_COMMAND_FILE") or (
    os.path.join(OUTPUT_DIR, "commands.txt") if os.environ.get("TB3_OUTPUT_DIR") else os.path.join(CTRL_DIR, "commands.txt"))

# --- 로봇 (TurtleBot3 Burger) ---
WHEEL_RADIUS = 0.033          # m
WHEEL_SEPARATION = 0.160      # m
ROBOT_RADIUS = 0.105          # m
WHEEL_MAX_SPEED = 6.67        # rad/s (선속도 최대 0.22 m/s)

# --- LiDAR (LDS-01) ---
LIDAR_N = 360
LIDAR_MIN = 0.12
LIDAR_MAX = 3.5
LIDAR_OFFSET_X = -0.03        # 로봇 중심 기준 LiDAR 위치 (extensionSlot Pose)
# 예제 기준 인덱스 규약: 0=후방, 90=좌측, 180=전방, 270=우측
# 로봇 프레임 각도(반시계 양수) = pi - i * (2pi / 360)
LIDAR_ANGLES = math.pi - np.arange(LIDAR_N) * (2.0 * math.pi / LIDAR_N)

# --- 카메라 ---
CAM_W = 640
CAM_H = 480
CAM_FOV = 1.0472                                   # rad (60도)
CAM_FX = (CAM_W / 2.0) / math.tan(CAM_FOV / 2.0)   # 약 554 px
CAM_OFFSET_X = -0.03 + 0.05                        # 로봇 중심 기준 카메라 전방 거리
CAM_HEIGHT = 0.153 - 0.08                          # 바닥 기준 카메라 높이 (약 0.073 m)

# --- 목표 물체 ---
APPLE_RADIUS = 0.05

_DEFAULTS = {
    "target_colors": ["red"],
    "target_count": 2,
    "forbidden_colors": ["green", "purple", "orange"],
    "time_limit_s": 600.0,
    "return_at_ratio": 0.7,

    "v_max": 0.18,
    "v_min": 0.08,
    "w_max": 1.5,
    "w_rotate": 0.8,
    "lookahead_m": 0.35,
    "goal_tol_m": 0.15,
    "heading_rotate_deg": 60.0,

    "stop_dist_m": 0.25,
    "slow_dist_m": 0.40,
    "side_dist_m": 0.16,
    "wait_before_replan_s": 5.0,
    "stuck_window_s": 2.5,
    "stuck_move_m": 0.05,

    "map_origin": [-3.0, -9.0],
    "map_size": [18.0, 18.0],
    "map_res": 0.05,
    "l_occ": 1.2,
    "l_free": 0.25,
    "l_clamp": 3.5,
    "occ_threshold": 0.5,
    "max_free_range": 2.5,
    "yaw_rate_skip": 1.0,
    "inflate_cells": 4,
    "soft_cells": 8,
    "min_frontier_cells": 6,
    "frontier_max_cost": 150,
    "frontier_min_dist_m": 0.4,
    "blacklist_radius_m": 0.5,
    "blacklist_ttl_s": 60.0,

    "approach_standoff_m": 0.35,
    "approach_stop_m": 0.25,
    "approach_v": 0.07,
    "visual_timeout_s": 12.0,
    "goal_fail_blacklist": 3,
    "protect_radius_m": 0.12,
    "rescue_pause_s": 2.0,
    "rescued_ignore_m": 0.6,

    "hsv": {
        "red":    [[0, 150, 70], [8, 255, 255], [172, 150, 70], [180, 255, 255]],
        "green":  [[35, 80, 40], [85, 255, 255]],
        "purple": [[125, 80, 40], [160, 255, 255]],
        "orange": [[9, 120, 80], [25, 255, 255]]
    },
    "det_min_area": 50,
    "det_min_circularity": 0.55,
    "det_floor_row": 228,
    "det_confirm_frames": 5,
    "det_window": 7,
    "det_cluster_m": 0.35,
    "det_max_range": 4.0,
    "det_ground_tol_px": 30.0,
    "save_every_s": 0.0,
    "camera_period_mult": 2,

    "compass_sign": -1.0,
    "scan_match_every_s": 0.0,
    "scan_match_search_m": 0.06,
    "scan_match_step_m": 0.01,
    "scan_match_min_gain": 0.004,
    "scan_match_apply": 0.7,
    "show_window": True,
    "draw_every": 8,
    "debug_ground_truth": False,
    "log_every_s": 5.0
}


def load_mission():
    cfg = dict(_DEFAULTS)
    if os.path.exists(MISSION_FILE):
        with open(MISSION_FILE, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg.update(user)
    if os.environ.get("TB3_HEADLESS") == "1":
        cfg["show_window"] = False
    if os.environ.get("TB3_DEBUG_GT") == "1":
        cfg["debug_ground_truth"] = True
    if os.environ.get("TB3_TIME_LIMIT"):
        cfg["time_limit_s"] = float(os.environ["TB3_TIME_LIMIT"])
    if os.environ.get("TB3_SAVE_EVERY"):
        cfg["save_every_s"] = float(os.environ["TB3_SAVE_EVERY"])
    return cfg


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
