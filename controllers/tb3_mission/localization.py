"""위치 추정 (담당 A).

- 방향(heading_source 설정):
    "gyro"    : IMU 자이로 z축 각속도를 적분 (기본값. 팀 센서 정책상 나침반은 사용하지 않음)
    "encoder" : 좌우 바퀴 이동 거리 차로 회전각 계산 (자이로도 없을 때)
    "compass" : 나침반 상대 각도 (비교·디버그용)
- 위치: 좌우 휠 엔코더 회전각 증분으로 이동 거리를 구하고 현재 방향으로 분해하여 적분한다.
- 좌표계: 시작 pose가 원점 (0, 0, 0)인 odom 프레임.
"""
import math

import numpy as np

from config import WHEEL_RADIUS, WHEEL_SEPARATION, wrap_angle


class Localizer:
    def __init__(self, left_sensor, right_sensor, compass, cfg, gyro=None):
        self.left_sensor = left_sensor
        self.right_sensor = right_sensor
        self.compass = compass
        self.gyro = gyro
        self.heading_source = str(cfg.get("heading_source", "gyro"))
        if self.heading_source == "gyro" and gyro is None:
            self.heading_source = "encoder"
        if self.heading_source == "compass" and compass is None:
            self.heading_source = "encoder"
        self.compass_sign = float(cfg.get("compass_sign", -1.0))

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.travel = 0.0            # 누적 이동 거리 (보고용)
        self.frozen_dist = 0.0       # 헛돔으로 버린 엔코더 거리 (보고용)
        self.correction_dist = 0.0   # 스캔 매칭으로 보정한 누적 거리 (보고용)
        self.correction_yaw = 0.0    # 스캔 매칭으로 보정한 누적 각도 (보고용)

        self._phi_l = None
        self._phi_r = None
        self._yaw_c0 = None
        self._yaw_prev = 0.0
        self._t_prev = None
        self.ready = False

    def _compass_yaw(self):
        if self.compass is None or self.heading_source != "compass":
            return None
        c = self.compass.getValues()
        if c is None or any(map(math.isnan, c[:2])):
            return None
        return math.atan2(c[1], c[0])

    def reset_origin(self, t):
        """센서가 유효해진 뒤 한 번 호출한다."""
        yc = self._compass_yaw()
        pl = self.left_sensor.getValue()
        pr = self.right_sensor.getValue()
        if math.isnan(pl) or math.isnan(pr):
            return False
        if self.heading_source == "compass" and yc is None:
            return False
        if self.heading_source == "gyro":
            g = self.gyro.getValues()
            if g is None or any(map(math.isnan, g)):
                return False
        self._yaw_c0 = yc if yc is not None else 0.0
        self._phi_l = pl
        self._phi_r = pr
        self.x = self.y = self.yaw = 0.0
        self._yaw_prev = 0.0
        self._t_prev = t
        self.ready = True
        return True

    def update(self, t, freeze=False):
        """freeze=True면 (바퀴 헛돔 판정) 엔코더 증분을 버리고 위치를 유지한다. 방향은 나침반이라 항상 갱신."""
        if not self.ready:
            return (self.x, self.y, self.yaw)

        pl = self.left_sensor.getValue()
        pr = self.right_sensor.getValue()
        if math.isnan(pl) or math.isnan(pr):
            return (self.x, self.y, self.yaw)

        d_l = (pl - self._phi_l) * WHEEL_RADIUS
        d_r = (pr - self._phi_r) * WHEEL_RADIUS
        self._phi_l, self._phi_r = pl, pr
        if freeze:
            self.frozen_dist += abs(0.5 * (d_l + d_r))
            d_l = d_r = 0.0
        ds = 0.5 * (d_l + d_r)

        dt = (t - self._t_prev) if (self._t_prev is not None and t > self._t_prev) else 0.0
        if self.heading_source == "gyro":
            gz = self.gyro.getValues()[2]
            if math.isnan(gz):
                gz = 0.0
            yaw_new = wrap_angle(self.yaw + gz * dt)
        elif self.heading_source == "compass":
            yc = self._compass_yaw()
            yaw_new = (wrap_angle(self.compass_sign * wrap_angle(yc - self._yaw_c0)) if yc is not None
                       else wrap_angle(self.yaw + (d_r - d_l) / WHEEL_SEPARATION))
        else:
            # 엔코더 차동: 헛돌 때는 회전도 믿을 수 없으므로 freeze 시 유지
            yaw_new = wrap_angle(self.yaw + ((d_r - d_l) / WHEEL_SEPARATION if not freeze else 0.0))

        dtheta = wrap_angle(yaw_new - self.yaw)
        yaw_mid = self.yaw + 0.5 * dtheta
        self.x += ds * math.cos(yaw_mid)
        self.y += ds * math.sin(yaw_mid)
        self.yaw = yaw_new
        self.travel += abs(ds)

        if self._t_prev is not None and t > self._t_prev:
            self.yaw_rate = dtheta / (t - self._t_prev)
        self._t_prev = t
        return (self.x, self.y, self.yaw)

    def pose(self):
        return (self.x, self.y, self.yaw)

    def correct(self, dx, dy, dyaw=0.0):
        """스캔 매칭 보정량을 반영한다."""
        self.x += dx
        self.y += dy
        self.yaw = wrap_angle(self.yaw + dyaw)
        self.correction_dist += math.hypot(dx, dy)
        self.correction_yaw += abs(dyaw)
