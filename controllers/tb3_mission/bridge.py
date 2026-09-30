"""웹 지휘 콘솔 연결 계층 (담당 E).

컨트롤러 내부 상태를 주기적으로 파일로 내보낸다. 웹 서버(web/server.py)는 이 파일들만 읽고,
명령은 commands.txt 로만 보내므로 컨트롤러 본체와 웹은 파일 두 개로 느슨하게 결합된다.

내보내는 파일 (output/ 폴더):
- status.json   : 시각, 상태 라벨, 자세, 목표, 구출 기록, 안전 계층, Behavior Tree 상태, 최근 이벤트
- live_map.png  : Reporter.render() 가 그린 지도 화면
- live_cam.jpg  : 검출 결과가 표시된 카메라 프레임

파일은 임시 파일에 쓴 뒤 os.replace 로 교체하므로 웹 서버가 절반만 쓰인 파일을 읽는 일이 없다.
"""
import json
import math
import os

import cv2

from config import OUTPUT_DIR, CTRL_DIR
from bt_core import describe

# 웹 서버가 현재 실행 중인 컨트롤러의 출력 폴더를 찾도록 고정 위치에 포인터를 남긴다.
# (OUTPUT_DIR 은 TB3_OUTPUT_DIR 환경 변수로 바뀔 수 있다.)
LIVE_POINTER = os.path.join(CTRL_DIR, "live_output_dir.txt")


def _f(v, nd=3):
    """JSON 직렬화용: numpy 스칼라와 NaN/inf 를 안전한 float 로."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_image(path, img, ext, params=None):
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


class StatusBridge:
    def __init__(self, app, status_every=None, image_every=None, history_every_s=5.0):
        """주기는 tick 단위. basicTimeStep 64 ms 기준 카메라 2 tick(약 8 fps), 지도 8 tick(약 2 fps), 상태 4 tick."""
        self.app = app
        cfg = app.cfg
        self.status_every = int(status_every or cfg.get("web_status_every", 4))
        self.cam_every = int(cfg.get("web_cam_every", 2))
        self.image_every = int(image_every or cfg.get("web_map_every", 8))
        self.jpeg_quality = int(cfg.get("web_jpeg_quality", 75))
        self.history_every_s = float(history_every_s)
        self.history = []            # [(t, explored_ratio, rescued_count, travel_m)]
        self.objects = []            # 탐지 객체 등록부 [{id, color, x, y, n, first_t, last_t, status}]
        self._next_obj_id = 1
        self.cluster_m = 0.4
        self._last_hist_t = -1e9
        self._tick = 0
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            with open(LIVE_POINTER, "w", encoding="utf-8") as f:
                f.write(os.path.abspath(OUTPUT_DIR) + "\n")
        except OSError:
            pass
        self.status_path = os.path.join(OUTPUT_DIR, "status.json")
        self.map_path = os.path.join(OUTPUT_DIR, "live_map.png")
        self.cam_path = os.path.join(OUTPUT_DIR, "live_cam.jpg")

    def _register(self, color, x, y, t, status):
        """같은 색이 반경 내에 있으면 위치를 평균으로 갱신하고, 없으면 새 객체로 등록한다."""
        for o in self.objects:
            if o["color"] == color and math.hypot(o["x"] - x, o["y"] - y) < self.cluster_m:
                n = o["n"]
                o["x"] = (o["x"] * n + x) / (n + 1)
                o["y"] = (o["y"] * n + y) / (n + 1)
                o["n"] = n + 1
                o["last_t"] = t
                rank = {"seen": 0, "forbidden": 1, "confirmed": 2, "rescued": 3}
                if rank.get(status, 0) > rank.get(o["status"], 0):
                    o["status"] = status
                return o
        o = {"id": self._next_obj_id, "color": color, "x": x, "y": y, "n": 1,
             "first_t": t, "last_t": t, "status": status}
        self._next_obj_id += 1
        self.objects.append(o)
        return o

    def _update_objects(self, t):
        det = self.app.detector
        for d in det.last_detections:
            st = "forbidden" if d["color"] in det.forbidden_colors else "seen"
            self._register(d["color"], float(d["world"][0]), float(d["world"][1]), t, st)
        c = self.app.bb.get("detection")
        if c and c.get("confirmed"):
            self._register(c.get("color", "red"), float(c["world"][0]), float(c["world"][1]), t, "confirmed")
        for r in self.app.bb.get("rescued", []):
            self._register("red" if not det.target_colors else det.target_colors[0],
                           float(r["x"]), float(r["y"]), float(r.get("t", t)), "rescued")

    def _collect(self):
        app = self.app
        bb = app.bb
        cfg = app.cfg
        t = float(bb.get("t", 0.0))
        pose = bb.get("pose") or (0.0, 0.0, 0.0)
        gt = bb.get("gt_pose")
        det = bb.get("detection")
        goal = bb.get("goal")
        explored = app.grid.explored_ratio()

        if t - self._last_hist_t >= self.history_every_s:
            self._last_hist_t = t
            self.history.append((round(t, 1), round(explored, 4),
                                 len(bb.get("rescued", [])), round(app.localizer.travel, 2)))
            if len(self.history) > 600:
                self.history.pop(0)

        self._update_objects(t)
        # 잡음성 단발 검출은 숨기고, 여러 번 관측되었거나 확정·구조된 객체만 내보낸다
        min_n = int(cfg.get("web_object_min_obs", 6))
        objects = [{"id": o["id"], "color": o["color"], "x": _f(o["x"]), "y": _f(o["y"]), "n": o["n"],
                    "first_t": _f(o["first_t"], 1), "last_t": _f(o["last_t"], 1), "status": o["status"]}
                   for o in self.objects
                   if o["status"] in ("confirmed", "rescued") or o["n"] >= min_n]
        objects.sort(key=lambda o: ({"rescued": 0, "confirmed": 1}.get(o["status"], 2), -o["n"]))
        objects = objects[:40]
        g = app.grid
        map_meta = {"x0": _f(g.x0), "y0": _f(g.y0), "res": _f(g.res, 4), "rows": g.rows, "cols": g.cols,
                    "scale": app.reporter.scale}
        manual = None
        if bb.get("command_state") == "manual":
            mv, mw = app.commands.manual_cmd()
            manual = {"v": _f(mv), "w": _f(mw)}

        events = [{"t": _f(et, 1), "msg": m} for (et, m) in app.reporter.events[-80:]]
        forbidden = [{"t": _f(ft, 1), "color": c, "x": _f(x), "y": _f(y)}
                     for (ft, c, x, y) in app.detector.forbidden_seen[-10:]]
        detections = [{"color": d["color"], "x": _f(d["world"][0]), "y": _f(d["world"][1]),
                       "dist": _f(d["dist"], 2)} for d in app.detector.last_detections[:8]]

        status = {
            "wall_time": __import__("time").time(),
            "t": _f(t, 1),
            "time_limit_s": _f(cfg.get("time_limit_s"), 0),
            "return_at_ratio": _f(cfg.get("return_at_ratio"), 2),
            "state_label": bb.get("state_label", ""),
            "mode": bb.get("mode"),
            "command_state": bb.get("command_state"),
            "done": bool(bb.get("done", False)),
            "pose": {"x": _f(pose[0]), "y": _f(pose[1]), "yaw_deg": _f(math.degrees(pose[2]), 1)},
            "gt_error_m": _f(math.hypot(gt[0] - pose[0], gt[1] - pose[1])) if gt else None,
            "travel_m": _f(app.localizer.travel, 2),
            "explored_ratio": _f(explored, 4),
            "target_colors": list(app.detector.target_colors),
            "forbidden_colors": list(app.detector.forbidden_colors),
            "targets_left": bb.get("targets_left"),
            "rescued": [{"x": _f(r["x"]), "y": _f(r["y"]), "t": _f(r["t"], 1)} for r in bb.get("rescued", [])],
            "goal": {"x": _f(goal[0]), "y": _f(goal[1]), "kind": bb.get("goal_kind")} if goal else None,
            "detection": ({"color": det.get("color"), "x": _f(det["world"][0]), "y": _f(det["world"][1]),
                           "dist": _f(det.get("dist"), 2), "frames": det.get("n")} if det else None),
            "detections": detections,
            "forbidden_seen": forbidden,
            "forbidden_seen_count": len(app.detector.forbidden_seen),
            "safety": {"level": bb.get("safety_level"), "front_min_m": _f(app.motion.front_min, 2),
                       "cmd_v": _f(app.motion.cmd_v, 3), "cmd_w": _f(app.motion.cmd_w, 3)},
            "stats": {k: (_f(v, 3) if isinstance(v, float) else v) for k, v in app.stats.items()},
            "frontier_count": len(bb.get("frontiers") or []),
            "objects": objects,
            "map_meta": map_meta,
            "manual": manual,
            "bt": describe(app.tree.root),
            "events": events,
            "history": self.history,
        }
        return status

    def update(self):
        self._tick += 1
        if self._tick % self.status_every == 0:
            try:
                _atomic_write_text(self.status_path, json.dumps(self._collect(), ensure_ascii=False))
            except Exception as e:      # 웹 연결 계층의 오류가 미션을 멈추면 안 된다
                print(f"[bridge] status 저장 실패: {e}", flush=True)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
        if self._tick % self.cam_every == 0:
            try:
                cam = self.app.detector.overlay()
                if cam is not None:
                    _atomic_write_image(self.cam_path, cam, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            except Exception as e:
                print(f"[bridge] 카메라 저장 실패: {e}", flush=True)
        if self._tick % self.image_every == 0:
            try:
                img = self.app.reporter.render(self.app.costmap, self.app.bb, self.app.detector)
                _atomic_write_image(self.map_path, img, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 1])
            except Exception as e:
                print(f"[bridge] 지도 저장 실패: {e}", flush=True)
