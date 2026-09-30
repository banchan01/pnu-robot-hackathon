"""색상 기반 사과 탐지 (담당 D).

파이프라인: BGRA → BGR → 가우시안 블러 → HSV → 색상 범위 이진 마스크 → Opening/Closing
→ 외곽 컨투어 → 면적/원형도/종횡비/바닥 위치 필터 → 최소 외접원으로 중심과 반지름
→ 핀홀 모델로 거리와 방위 → 로봇 pose로 지도 좌표 변환.

확정 로직: 최근 N프레임 중 M프레임 이상에서 같은 위치(반경 내)로 목표 색이 검출되면 확정.
금지 색은 검출하되 절대 확정하지 않고 기록만 한다.
"""
import math

import cv2
import numpy as np

import os

from config import CAM_W, CAM_H, CAM_FX, CAM_OFFSET_X, CAM_HEIGHT, APPLE_RADIUS, OUTPUT_DIR

HORIZON_ROW = CAM_H / 2.0


class AppleDetector:
    def __init__(self, camera, cfg):
        self.camera = camera
        self.hsv_ranges = cfg["hsv"]
        self.target_colors = list(cfg["target_colors"])
        self.forbidden_colors = list(cfg["forbidden_colors"])
        self.min_area = float(cfg["det_min_area"])
        self.min_circ = float(cfg["det_min_circularity"])
        self.floor_row = int(cfg["det_floor_row"])
        self.confirm_frames = int(cfg["det_confirm_frames"])
        self.window = int(cfg["det_window"])
        self.cluster_m = float(cfg["det_cluster_m"])
        self.max_range = float(cfg["det_max_range"])
        self.rescued_ignore = float(cfg["rescued_ignore_m"])
        self.ground_tol_px = float(cfg.get("det_ground_tol_px", 30.0))
        self.suppressed = []         # 오탐/도달 불가 목표 억제 [(x, y, expire_t)]
        self.rejected_ground = 0     # 바닥 기하 검사로 기각된 수 (보고용)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.history = []            # 최근 프레임별 목표 색 검출 리스트
        self.last_frame = None       # 디버그 표시용 BGR
        self.last_detections = []
        self.last_target = None      # 가장 최근 프레임의 목표 색 검출 (미확정 포함)
        self.forbidden_seen = []     # 금지 색 검출 기록 [(t, color, x, y)]
        self.rescued = []            # 이미 구출한 좌표 [(x, y)]

    def set_targets(self, colors):
        self.target_colors = list(colors)
        self.history = []

    def _mask(self, hsv, color):
        rng = self.hsv_ranges.get(color)
        if not rng:
            return None
        mask = None
        for i in range(0, len(rng), 2):
            lo = np.array(rng[i], dtype=np.uint8)
            hi = np.array(rng[i + 1], dtype=np.uint8)
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask

    def _near_rescued(self, x, y):
        for (rx, ry) in self.rescued:
            if math.hypot(x - rx, y - ry) < self.rescued_ignore:
                return True
        return False

    def _suppressed(self, x, y, t):
        self.suppressed = [s for s in self.suppressed if s[2] > t]
        for (sx, sy, _) in self.suppressed:
            if math.hypot(x - sx, y - sy) < self.rescued_ignore:
                return True
        return False

    def suppress(self, x, y, t, ttl_s=45.0):
        self.suppressed.append((float(x), float(y), t + ttl_s))
        self.history = []
        self.last_target = None

    def save_snapshot(self, name):
        img = self.overlay()
        if img is None:
            return None
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"detect_{name}.png")
        cv2.imwrite(path, img)
        return path

    def detect(self, pose, t):
        img = self.camera.getImage()
        if img is None:
            return []
        frame = np.frombuffer(img, np.uint8).reshape((CAM_H, CAM_W, 4))
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        blur = cv2.GaussianBlur(bgr, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        x, y, yaw = pose
        dets = []
        for color in set(self.target_colors) | set(self.forbidden_colors):
            mask = self._mask(hsv, color)
            if mask is None:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                per = cv2.arcLength(cnt, True)
                if per <= 0:
                    continue
                circ = 4.0 * math.pi * area / (per * per)
                if circ < self.min_circ:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / float(bh) if bh > 0 else 0
                # 사과는 거의 원형 (소화기 등 수직으로 긴 물체 배제)
                if aspect < (0.70 if color == "red" else 0.60) or aspect > 1.45:
                    continue

                # 소화기 배제 1: 사과는 바닥 위 독립된 구체이므로 위쪽에 붉은색 물체가 이어지지 않음.
                # 소화기는 상단 실린더와 라벨로 인해 컨투어 위쪽 수직 영역에 붉은 픽셀이 다량 존재함.
                top_y = max(0, by - 6)
                if color == "red" and top_y > 0:
                    x1 = max(0, bx - 15)
                    x2 = min(CAM_W, bx + bw + 15)
                    above_crop = mask[0:top_y, x1:x2]
                    if cv2.countNonZero(above_crop) > 35:
                        continue

                (cx, cy), r_px = cv2.minEnclosingCircle(cnt)
                if cy < self.floor_row:           # 탁자 위나 벽에 걸린 물체 제외
                    continue
                if r_px < 3:
                    continue
                dist = CAM_FX * (2.0 * APPLE_RADIUS) / (2.0 * r_px)
                if dist > self.max_range or dist < 0.12:
                    continue
                # 바닥 기하 일관성: 바닥 위 사과의 중심은 수평선 아래 f*(h_cam - r)/d 위치에 있어야 한다
                expect_cy = HORIZON_ROW + CAM_FX * (CAM_HEIGHT - APPLE_RADIUS) / dist
                if abs(cy - expect_cy) > self.ground_tol_px:
                    self.rejected_ground += 1
                    continue
                bearing = -math.atan((cx - CAM_W / 2.0) / CAM_FX)
                lx = CAM_OFFSET_X + dist * math.cos(bearing)
                ly = dist * math.sin(bearing)
                wx = x + math.cos(yaw) * lx - math.sin(yaw) * ly
                wy = y + math.sin(yaw) * lx + math.cos(yaw) * ly

                # 소화기 배제 2: 시작점 복도 벽면에 거치된 소화기 위치 (odom ~0.14, 0.66) 주변 배제
                if color == "red" and math.hypot(wx - 0.14, wy - 0.66) < 0.9:
                    continue
                dets.append({
                    "color": color,
                    "world": (wx, wy),
                    "pixel": (float(cx), float(cy)),
                    "radius_px": float(r_px),
                    "dist": float(dist),
                    "area": float(area),
                    "confirmed": False,
                    "t": t,
                })

        self.last_frame = bgr
        self.last_detections = dets
        targets = [d for d in dets if d["color"] in self.target_colors
                   and not self._near_rescued(*d["world"])
                   and not self._suppressed(d["world"][0], d["world"][1], t)]
        targets.sort(key=lambda d: -d["area"])
        self.last_target = targets[0] if targets else None
        for d in dets:
            if d["color"] in self.forbidden_colors:
                self.forbidden_seen.append((t, d["color"], d["world"][0], d["world"][1]))
        self.history.append(targets)
        if len(self.history) > self.window:
            self.history.pop(0)
        return dets

    def confirm(self):
        """최근 window 프레임에서 confirm_frames 이상 같은 위치에 목표가 있으면 확정 dict."""
        if len(self.history) < self.confirm_frames:
            return None
        frames = [f for f in self.history if f]
        if len(frames) < self.confirm_frames:
            return None
        # 마지막 프레임의 가장 큰 검출을 기준으로 클러스터링
        ref = frames[-1][0]
        cluster = []
        for f in frames:
            for d in f:
                if math.hypot(d["world"][0] - ref["world"][0], d["world"][1] - ref["world"][1]) < self.cluster_m:
                    cluster.append(d)
                    break
        if len(cluster) < self.confirm_frames:
            return None
        wx = float(np.mean([d["world"][0] for d in cluster]))
        wy = float(np.mean([d["world"][1] for d in cluster]))
        out = dict(ref)
        out["world"] = (wx, wy)
        out["confirmed"] = True
        out["n"] = len(cluster)
        return out

    def mark_rescued(self, x, y):
        self.rescued.append((float(x), float(y)))
        self.history = []
        self.last_target = None

    def overlay(self):
        """디버그 창용 카메라 프레임 (검출 표시 포함)."""
        if self.last_frame is None:
            return None
        img = self.last_frame.copy()
        cv2.line(img, (0, self.floor_row), (CAM_W, self.floor_row), (80, 80, 80), 1)
        for d in self.last_detections:
            c = (0, 0, 255) if d["color"] in self.target_colors else (255, 0, 255)
            cx, cy = int(d["pixel"][0]), int(d["pixel"][1])
            cv2.circle(img, (cx, cy), int(d["radius_px"]), c, 2)
            cv2.putText(img, f'{d["color"]} {d["dist"]:.2f}m', (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        return img
