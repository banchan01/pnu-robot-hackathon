"""지역 제어와 안전 계층 (담당 C).

- Look-ahead(Pure Pursuit) 경로 추종: 노트북의 곡률 공식 k = 2*y / (x^2 + y^2), w = v*k
- 바퀴 속도 변환: v_r = v + wL/2, v_l = v - wL/2, 바퀴 각속도 = v / R
- 안전 계층: 전방 구간 최소 거리로 정지/감속, 측면 근접 시 반대쪽으로 보정
- 끼임 감지: 명령이 있는데 일정 시간 위치 변화가 없으면 True
"""
import math

import numpy as np

from config import (WHEEL_RADIUS, WHEEL_SEPARATION, WHEEL_MAX_SPEED,
                    CAM_FX, APPLE_RADIUS, CAM_W, LIDAR_MIN, wrap_angle)

# LiDAR 인덱스 구간 (180=전방, 90=좌, 270=우)
FRONT_IDX = np.arange(150, 211)          # 전방 ±30도
LEFT_IDX = np.arange(100, 150)           # 좌전방 30~80도
RIGHT_IDX = np.arange(211, 261)          # 우전방 -80~-30도


def _sector_min(ranges, idx):
    r = np.asarray(ranges, dtype=np.float32)[idx]
    r = r[np.isfinite(r) & (r >= LIDAR_MIN)]
    return float(r.min()) if r.size else float("inf")


class ScanMotionMonitor:
    """연속 LiDAR 스캔의 변화량으로 로봇이 실제로 움직이는지 판정한다.
    바퀴가 헛돌면 엔코더는 움직였다고 하지만 스캔은 그대로다."""

    def __init__(self, window_s=0.4, static_thresh_m=0.008):
        self.window_s = window_s
        self.thresh = static_thresh_m
        self._hist = []                  # (t, ranges)
        self.metric = 0.0
        self.static = False
        self.static_since = None

    def update(self, t, ranges):
        r = np.asarray(ranges, dtype=np.float32)
        self._hist.append((t, r))
        while len(self._hist) > 2 and t - self._hist[0][0] > self.window_s:
            self._hist.pop(0)
        t0, r0 = self._hist[0]
        if t - t0 < self.window_s * 0.8 or r0.shape != r.shape:
            self.metric = 0.0
            self.static = False
            self.static_since = None
            return
        ok = np.isfinite(r) & np.isfinite(r0) & (r >= LIDAR_MIN) & (r0 >= LIDAR_MIN)
        if ok.sum() < 30:
            self.static = False
            self.static_since = None
            return
        self.metric = float(np.median(np.abs(r[ok] - r0[ok])))
        now_static = self.metric < self.thresh
        if now_static:
            if self.static_since is None:
                self.static_since = t
        else:
            self.static_since = None
        self.static = now_static

    def static_for(self, t):
        return 0.0 if self.static_since is None else t - self.static_since


class Motion:
    def __init__(self, left_motor, right_motor, cfg):
        self.lm = left_motor
        self.rm = right_motor
        self.lm.setPosition(float("inf"))
        self.rm.setPosition(float("inf"))
        self.lm.setVelocity(0.0)
        self.rm.setVelocity(0.0)

        self.v_max = float(cfg["v_max"])
        self.v_min = float(cfg["v_min"])
        self.w_max = float(cfg["w_max"])
        self.w_rotate = float(cfg["w_rotate"])
        self.lookahead = float(cfg["lookahead_m"])
        self.goal_tol = float(cfg["goal_tol_m"])
        self.heading_rotate = math.radians(float(cfg["heading_rotate_deg"]))
        self.stop_dist = float(cfg["stop_dist_m"])
        self.slow_dist = float(cfg["slow_dist_m"])
        self.side_dist = float(cfg["side_dist_m"])
        self.stuck_window = float(cfg["stuck_window_s"])
        self.stuck_move = float(cfg["stuck_move_m"])
        self.approach_v = float(cfg["approach_v"])
        self.approach_stop_px = CAM_FX * (2 * APPLE_RADIUS) / float(cfg["approach_stop_m"])

        self.speed_scale = 1.0
        self.side_bias = 0.0
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.wanted_v = 0.0
        self.blocked_forward = False
        self.left_min = float("inf")
        self.right_min = float("inf")
        self._path_id = None
        self._idx = 0
        self._rot_ticks = 0
        self._hist = []          # (t, x, y)
        self.front_min = float("inf")

    # ---------- 저수준 ----------
    def set_cmd(self, v, w):
        v = max(-self.v_max, min(self.v_max, v))
        self.wanted_v = v
        # 안전 배율은 전진에만 적용한다. 후진과 제자리 회전은 항상 허용.
        if v > 0.0:
            v *= self.speed_scale
        self.blocked_forward = (self.wanted_v > 0.02 and self.speed_scale == 0.0)
        w = max(-self.w_max, min(self.w_max, w))
        vl = (v - w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        vr = (v + w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        m = max(abs(vl), abs(vr))
        if m > WHEEL_MAX_SPEED:
            vl *= WHEEL_MAX_SPEED / m
            vr *= WHEEL_MAX_SPEED / m
        self.lm.setVelocity(vl)
        self.rm.setVelocity(vr)
        self.cmd_v, self.cmd_w = v, w

    def stop(self):
        self.lm.setVelocity(0.0)
        self.rm.setVelocity(0.0)
        self.cmd_v = self.cmd_w = 0.0
        self.wanted_v = 0.0
        self.blocked_forward = False

    def freer_side(self):
        """회피 회전 방향: 좌측이 더 비어 있으면 +1, 아니면 -1."""
        return 1.0 if self.left_min >= self.right_min else -1.0

    # ---------- 안전 ----------
    def apply_safety(self, ranges):
        """'clear' | 'slow' | 'blocked' 를 돌려주고 속도 배율과 측면 보정을 설정한다."""
        f = _sector_min(ranges, FRONT_IDX)
        l = _sector_min(ranges, LEFT_IDX)
        r = _sector_min(ranges, RIGHT_IDX)
        self.front_min = f
        self.left_min = l
        self.right_min = r
        self.side_bias = 0.0
        if l < self.side_dist:
            self.side_bias -= 0.4      # 좌측이 가까우면 우회전 보정
        if r < self.side_dist:
            self.side_bias += 0.4
        if f < self.stop_dist:
            self.speed_scale = 0.0
            return "blocked"
        # 좁은 통로(수납장 문짝 등 얇은 장애물이 빔 사이로 빠지는 곳)에서는 전방 ±80도 안에
        # 0.35 m 이내 장애물이 있어도 감속한다
        if f < self.slow_dist or min(l, r) < 0.35:
            self.speed_scale = 0.5
            return "slow"
        self.speed_scale = 1.0
        return "clear"

    def moving_cmd(self):
        return abs(self.cmd_v) > 0.02 or abs(self.cmd_w) > 0.2

    def is_stuck(self, t, static_for):
        """움직이라는 명령이 있는데 LiDAR 스캔이 stuck_window 이상 정지해 있으면 끼임."""
        if not self.moving_cmd():
            self._stuck_t0 = None
            return False
        if static_for <= 0.0:
            return False
        return static_for >= self.stuck_window

    def clear_stuck(self):
        self._hist = []

    # ---------- 경로 추종 ----------
    def reset_path(self):
        self._path_id = None
        self._idx = 0

    def follow(self, pose, path):
        """경로를 따라가고 종점 도달 시 True."""
        if not path:
            self.stop()
            return True
        if self._path_id is not id(path):
            self._path_id = id(path)
            self._idx = 0
        x, y, yaw = pose
        n = len(path)

        # 가장 가까운 waypoint (현재 인덱스 이후 창 안에서)
        best, best_d = self._idx, float("inf")
        for k in range(self._idx, min(n, self._idx + 25)):
            d = math.hypot(path[k][0] - x, path[k][1] - y)
            if d < best_d:
                best, best_d = k, d
        self._idx = best

        gx, gy = path[-1]
        if math.hypot(gx - x, gy - y) < self.goal_tol:
            self.stop()
            return True

        # Look-ahead 지점: 현재 인덱스 이후에서 로봇과의 직선 거리가 처음으로 lookahead 이상이 되는 waypoint
        # (경로가 접혀 되돌아오는 구간에서 로봇 뒤쪽 점을 고르는 진동을 막는다)
        la = path[-1]
        for k in range(best, n):
            if math.hypot(path[k][0] - x, path[k][1] - y) >= self.lookahead:
                la = path[k]
                break

        dx = la[0] - x
        dy = la[1] - y
        # 로봇 프레임으로 변환
        rx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        ry = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        err = math.atan2(ry, rx)

        if abs(err) > self.heading_rotate:
            self.set_cmd(0.0, math.copysign(self.w_rotate, err))
            self._rot_ticks += 1
            if self._rot_ticks >= 60 and self._rot_ticks % 75 == 0:      # 약 4초 이상 회전이 이어지면 진단 출력
                print(f"[follow-diag] rot_ticks={self._rot_ticks} pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) "
                      f"best={best}/{n} la=({la[0]:.2f},{la[1]:.2f}) d_la={math.hypot(dx, dy):.2f} err={math.degrees(err):.0f}deg "
                      f"path0=({path[0][0]:.2f},{path[0][1]:.2f}) pathN=({path[-1][0]:.2f},{path[-1][1]:.2f})", flush=True)
            return False
        self._rot_ticks = 0

        d2 = rx * rx + ry * ry
        k = 2.0 * ry / d2 if d2 > 1e-6 else 0.0
        v = self.v_max * (1.0 - 0.7 * abs(err) / self.heading_rotate)
        v = max(self.v_min, v)
        w = v * k + self.side_bias
        self.set_cmd(v, w)
        return False

    def rotate_to(self, pose, target_yaw, tol_deg=3.0):
        err = wrap_angle(target_yaw - pose[2])
        if abs(err) < math.radians(tol_deg):
            self.stop()
            return True
        w = max(-self.w_rotate, min(self.w_rotate, 2.0 * err))
        if abs(w) < 0.25:
            w = math.copysign(0.25, w)
        self.set_cmd(0.0, w)
        return False

    def back_off(self, v=-0.08):
        self.set_cmd(v, 0.0)

    # ---------- 시각 서보 ----------
    def approach_visual(self, det):
        """사과가 화면 중앙에 오도록 회전하며 전진.
        성공: 정지 거리(픽셀 지름)에 도달했거나, 안전 계층에 막혔지만 사과가 0.6 m 이내에 정면으로 보일 때."""
        cx = det["pixel"][0]
        diam = 2.0 * det["radius_px"]
        err_px = (CAM_W / 2.0) - cx
        centered = abs(err_px) < 60
        if diam >= self.approach_stop_px and centered:
            self.stop()
            return True
        if self.speed_scale == 0.0 and det["dist"] < 0.6 and centered:
            self.stop()
            return True
        w = max(-0.6, min(0.6, 0.004 * err_px))
        v = self.approach_v if abs(err_px) < 40 else 0.0
        self.set_cmd(v, w)
        return False
