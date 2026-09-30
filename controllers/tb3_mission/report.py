"""디버그 화면과 결과 보고 (담당 A).

- draw(): 지도, 로봇, 경로, frontier, 검출, 구출 위치를 한 화면에 그린다 (OpenCV 창).
- save(): 지도 PNG, 궤적 CSV, 요약 JSON을 output/ 폴더에 저장한다.
"""
import json
import math
import os

import cv2
import numpy as np

from config import OUTPUT_DIR
from mapping import LETHAL, INSCRIBED, UNKNOWN


# 디버그 전용: apartment.wbt의 사과 실제 위치를 시작 pose 기준 odom 좌표로 바꾼 값
# (world → odom: x = -(wx + 0.3), y = -(wy + 7.5), 시작 방향 180도)
DEBUG_APPLES_ODOM = [
    (11.72, -4.48, "red"), (5.04, 3.04, "red"),
    (8.56, 0.17, "green"), (8.07, -2.52, "green"),
    (2.54, -6.16, "purple"), (7.66, 4.18, "purple"),
    (7.98, -2.69, "orange"),
]
_MARK = {"red": (0, 0, 255), "green": (0, 160, 0), "purple": (200, 0, 200), "orange": (0, 140, 255)}


class Reporter:
    def __init__(self, grid, cfg, show_window):
        self.grid = grid
        self.cfg = cfg
        self.show = show_window
        self.scale = 2
        self.traj = []              # (t, x, y, yaw)
        self.events = []            # (t, msg)
        self.gt_errors = []         # (t, pos_err, yaw_err)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if self.show:
            try:
                cv2.namedWindow("tb3_mission map", cv2.WINDOW_NORMAL)
                cv2.namedWindow("tb3_mission camera", cv2.WINDOW_NORMAL)
            except cv2.error:
                self.show = False

    def log(self, t, msg):
        # 같은 메시지가 1초 안에 반복되면 한 번만 남긴다 (매 tick 실패 로그 폭주 방지)
        last = getattr(self, "_last_msg", None)
        if last is not None and last[1] == msg and t - last[0] < 1.0:
            return
        self._last_msg = (t, msg)
        line = f"[{t:7.1f}s] {msg}"
        print(line, flush=True)
        self.events.append((t, msg))

    def record_pose(self, t, pose):
        if not self.traj or t - self.traj[-1][0] >= 0.5:
            self.traj.append((t, pose[0], pose[1], pose[2]))

    def _to_px(self, x, y):
        r, c = self.grid.world_to_grid(x, y)
        return int(c * self.scale), int((self.grid.rows - 1 - r) * self.scale)

    def render(self, costmap, bb, detector=None):
        cm = costmap
        img = np.full(cm.shape, 200, dtype=np.uint8)
        img[cm == 0] = 255
        soft = (cm > 0) & (cm < INSCRIBED)
        img[soft] = (255 - (cm[soft].astype(np.int32) * 0.5)).astype(np.uint8)
        img[cm == INSCRIBED] = 90
        img[cm == LETHAL] = 0
        img[cm == UNKNOWN] = 150
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        seen = (cm == 0) & self.grid.viewed
        img[seen] = (235, 245, 225)          # 카메라가 훑은 자유 공간은 연한 초록빛
        img = cv2.flip(img, 0)
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        for (x, y, *_ ) in bb.get("frontiers", []) or []:
            cv2.circle(img, self._to_px(x, y), 4, (0, 180, 0), -1)
        path = bb.get("path")
        if path:
            pts = [self._to_px(x, y) for (x, y) in path]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, a, b, (255, 120, 0), 2)
        goal = bb.get("goal")
        if goal:
            cv2.drawMarker(img, self._to_px(*goal), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        if len(self.traj) > 1:
            pts = [self._to_px(p[1], p[2]) for p in self.traj]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, a, b, (180, 180, 60), 1)
        cv2.drawMarker(img, self._to_px(0, 0), (200, 120, 0), cv2.MARKER_SQUARE, 14, 2)
        if detector is not None:
            for d in detector.last_detections:
                col = (0, 0, 255) if d["color"] in detector.target_colors else (200, 0, 200)
                cv2.circle(img, self._to_px(*d["world"]), 5, col, 1)
        det = bb.get("detection")
        if det and det.get("confirmed"):
            cv2.circle(img, self._to_px(*det["world"]), 8, (0, 0, 255), 2)
        for r in bb.get("rescued", []):
            cv2.circle(img, self._to_px(r["x"], r["y"]), 7, (0, 0, 255), -1)
        pose = bb.get("pose")
        if pose:
            p = self._to_px(pose[0], pose[1])
            cv2.circle(img, p, 6, (0, 140, 255), -1)
            q = (int(p[0] + 14 * math.cos(pose[2])), int(p[1] - 14 * math.sin(pose[2])))
            cv2.line(img, p, q, (0, 140, 255), 2)
        gt = bb.get("gt_pose")
        if gt:
            cv2.circle(img, self._to_px(gt[0], gt[1]), 6, (255, 0, 0), 1)
            for (ax, ay, col) in DEBUG_APPLES_ODOM:
                cv2.drawMarker(img, self._to_px(ax, ay), _MARK[col], cv2.MARKER_SQUARE, 10, 1)

        lines = [
            f"t={bb.get('t', 0):.0f}s  {bb.get('state_label', '')}  {bb.get('wait_label', '') if bb.get('safety_level') == 'blocked' else ''}",
            f"targets_left={bb.get('targets_left')}  rescued={len(bb.get('rescued', []))}  "
            f"explored={self.grid.explored_ratio() * 100:.1f}% viewed={self.grid.viewed_ratio() * 100:.0f}%",
            f"cmd={bb.get('command_state')}  safety={bb.get('safety_level')}  front={bb.get('front_min', 0):.2f}m",
        ]
        if gt and pose:
            e = math.hypot(gt[0] - pose[0], gt[1] - pose[1])
            lines.append(f"GT err pos={e:.3f}m yaw={math.degrees(abs((gt[2]-pose[2]+math.pi)%(2*math.pi)-math.pi)):.1f}deg")
        y0 = 22
        for ln in lines:
            cv2.putText(img, ln, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(img, ln, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y0 += 22
        return img

    def draw(self, costmap, bb, detector=None):
        if not self.show:
            return
        try:
            img = self.render(costmap, bb, detector)
            cv2.imshow("tb3_mission map", img)
            if detector is not None:
                cam = detector.overlay()
                if cam is not None:
                    cv2.imshow("tb3_mission camera", cam)
            cv2.waitKey(1)
        except cv2.error:
            self.show = False

    def save(self, costmap, bb, detector, stats, tag="final"):
        img = self.render(costmap, bb, detector)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"map_{tag}.png"), img)
        np.save(os.path.join(OUTPUT_DIR, f"logodds_{tag}.npy"), self.grid.logodds)
        with open(os.path.join(OUTPUT_DIR, f"trajectory_{tag}.csv"), "w") as f:
            f.write("t,x,y,yaw\n")
            for (t, x, y, yaw) in self.traj:
                f.write(f"{t:.2f},{x:.3f},{y:.3f},{yaw:.3f}\n")
        path_len = 0.0
        for a, b in zip(self.traj[:-1], self.traj[1:]):
            path_len += math.hypot(b[1] - a[1], b[2] - a[2])
        summary = {
            "elapsed_s": bb.get("t"),
            "rescued": bb.get("rescued"),
            "targets_left": bb.get("targets_left"),
            "explored_ratio": self.grid.explored_ratio(),
            "path_length_m": path_len,
            "final_pose": bb.get("pose"),
            "return_error_m": math.hypot(bb.get("pose")[0], bb.get("pose")[1]) if bb.get("pose") else None,
            "stats": stats,
            "forbidden_seen": len(detector.forbidden_seen) if detector else 0,
            "events": [{"t": t, "msg": m} for (t, m) in self.events],
        }
        if self.gt_errors:
            summary["gt_pos_err_max_m"] = max(e[1] for e in self.gt_errors)
            summary["gt_pos_err_final_m"] = self.gt_errors[-1][1]
        with open(os.path.join(OUTPUT_DIR, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        return summary
