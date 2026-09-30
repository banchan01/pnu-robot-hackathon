"""위치 추정 (담당 A).

- 방향: 나침반. Webots 나침반은 노이즈 설정이 없으면 정확한 북쪽 벡터를 주므로
  시작 시점 대비 상대 각도를 그대로 방향으로 쓴다. 자이로 적분보다 drift가 없다.
- 위치: 좌우 휠 엔코더 회전각 증분으로 이동 거리를 구하고 현재 방향으로 분해하여 적분한다.
- 좌표계: 시작 pose가 원점 (0, 0, 0)인 odom 프레임.
"""
import math

import numpy as np

from config import WHEEL_RADIUS, WHEEL_SEPARATION, wrap_angle


class Localizer:
    def __init__(self, left_sensor, right_sensor, compass, cfg):
        self.left_sensor = left_sensor
        self.right_sensor = right_sensor
        self.compass = compass
        self.compass_sign = float(cfg.get("compass_sign", -1.0))

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.travel = 0.0            # 누적 이동 거리 (보고용)
        self.frozen_dist = 0.0       # 헛돔으로 버린 엔코더 거리 (보고용)
        self.correction_dist = 0.0   # 스캔 매칭으로 보정한 누적 거리 (보고용)

        self._phi_l = None
        self._phi_r = None
        self._yaw_c0 = None
        self._yaw_prev = 0.0
        self._t_prev = None
        self.ready = False

    def _compass_yaw(self):
        c = self.compass.getValues()
        if c is None or any(map(math.isnan, c[:2])):
            return None
        return math.atan2(c[1], c[0])

    def reset_origin(self, t):
        """센서가 유효해진 뒤 한 번 호출한다."""
        yc = self._compass_yaw()
        pl = self.left_sensor.getValue()
        pr = self.right_sensor.getValue()
        if yc is None or math.isnan(pl) or math.isnan(pr):
            return False
        self._yaw_c0 = yc
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

        yc = self._compass_yaw()
        if yc is not None:
            yaw_new = wrap_angle(self.compass_sign * wrap_angle(yc - self._yaw_c0))
        else:
            yaw_new = wrap_angle(self.yaw + (d_r - d_l) / WHEEL_SEPARATION)

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

    def correct(self, dx, dy):
        """스캔 매칭 보정량을 반영한다."""
        self.x += dx
        self.y += dy
        self.correction_dist += math.hypot(dx, dy)
