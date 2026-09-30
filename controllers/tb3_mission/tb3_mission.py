"""TurtleBot3 Burger Search & Rescue 컨트롤러 진입점.

매 timestep:
  센서 읽기 → 위치 추정 → Bayesian 지도 갱신 → 사과 탐지 → 명령 수신
  → Blackboard 갱신 → Behavior Tree tick → 화면 갱신
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import Robot, Keyboard  # noqa: E402

from config import load_mission, wrap_angle  # noqa: E402
from localization import Localizer  # noqa: E402
from mapping import OccupancyGrid  # noqa: E402
from motion import Motion, ScanMotionMonitor  # noqa: E402
from perception import AppleDetector  # noqa: E402
from bt_core import Blackboard, describe  # noqa: E402
from bt_mission import build_tree  # noqa: E402
from commands import CommandSource  # noqa: E402
from report import Reporter  # noqa: E402
from bridge import StatusBridge  # noqa: E402


class App:
    def __init__(self):
        self.cfg = load_mission()
        cfg = self.cfg

        self.gt_node = None
        if cfg.get("debug_ground_truth"):
            try:
                from controller import Supervisor
                self.robot = Supervisor()
                self.gt_node = self.robot.getSelf()
            except Exception:
                self.robot = Robot()
        else:
            self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        self.dt = self.ts / 1000.0

        r = self.robot
        self.left_motor = r.getDevice("left wheel motor")
        self.right_motor = r.getDevice("right wheel motor")
        self.left_enc = self.left_motor.getPositionSensor()
        self.right_enc = self.right_motor.getPositionSensor()
        if self.left_enc is not None:
            self.left_enc.enable(self.ts)
        if self.right_enc is not None:
            self.right_enc.enable(self.ts)
        self.compass = r.getDevice("compass")
        if self.compass is not None:
            self.compass.enable(self.ts)
        self.lidar = r.getDevice("LDS-01")
        if self.lidar is not None:
            self.lidar.enable(self.ts)
        self.camera = r.getDevice("camera")
        if self.camera is not None:
            self.camera.enable(self.ts * int(self.cfg.get("camera_period_mult", 2)))
        self.keyboard = Keyboard()
        if self.keyboard is not None:
            self.keyboard.enable(self.ts)

        self.grid = OccupancyGrid(cfg)
        self.reporter = Reporter(self.grid, cfg, cfg.get("show_window", True))
        self.log = lambda msg: self.reporter.log(self.robot.getTime(), msg)
        self.localizer = Localizer(self.left_enc, self.right_enc, self.compass, cfg)
        self.motion = Motion(self.left_motor, self.right_motor, cfg)
        self.scan_monitor = ScanMotionMonitor()
        self.detector = AppleDetector(self.camera, cfg)
        self.commands = CommandSource(self.keyboard, self.log)

        self.bb = Blackboard()
        self.bb.set("targets_left", int(cfg["target_count"]))
        self.bb.set("rescued", [])
        self.bb.set("command_state", "run")
        self.bb.set("replan", False)
        self.bb.set("frontier_exhausted", False)
        self.bb.set("sweep_exhausted", False)
        self.bb.set("done", False)
        self.blacklist = []                 # (x, y, radius, expire_t)
        self.stats = {"replans": 0, "stuck_events": 0, "block_events": 0, "min_front_m": 99.0, "scan_corrections": 0}
        self.costmap = self.grid.costmap(force=True)
        self.tree, self.nodes = build_tree(self)
        self.bridge = StatusBridge(self)      # 웹 지휘 콘솔용 상태 내보내기
        self._last_log_t = 0.0
        self._last_confirm_t = -1e9
        self._last_match_t = 0.0
        self._last_save_t = 0.0
        self._tick = 0
        self._gt_yaw0 = None

    # ---------- 블랙리스트 ----------
    def blacklist_goal(self, goal):
        if not goal:
            return
        self.blacklist.append((goal[0], goal[1], float(self.cfg["blacklist_radius_m"]),
                               self.robot.getTime() + float(self.cfg["blacklist_ttl_s"])))

    def clear_blacklist(self):
        self.blacklist = []

    # ---------- ground truth (디버그 전용) ----------
    def _gt_pose(self):
        if self.gt_node is None:
            return None
        p = self.gt_node.getPosition()
        o = self.gt_node.getOrientation()
        yaw = math.atan2(o[3], o[0])
        if self._gt_yaw0 is None:
            self._gt_yaw0 = (p[0], p[1], yaw)
        x0, y0, yaw0 = self._gt_yaw0
        dx, dy = p[0] - x0, p[1] - y0
        # 시작 pose 기준 odom 프레임으로 변환
        rx = math.cos(-yaw0) * dx - math.sin(-yaw0) * dy
        ry = math.sin(-yaw0) * dx + math.cos(-yaw0) * dy
        return (rx, ry, wrap_angle(yaw - yaw0))

    def finish_report(self):
        s = self.reporter.save(self.costmap, self.bb, self.detector, self.stats, tag="final")
        self.log(f"결과 저장: output/ (탐색률 {s['explored_ratio']*100:.1f}%, 이동 {s['path_length_m']:.1f} m)")

    # ---------- 메인 루프 ----------
    def run(self):
        cfg = self.cfg
        r = self.robot
        self.log(f"tb3_mission 시작. timestep={self.ts} ms, 목표={cfg['target_colors']} x{cfg['target_count']}, "
                 f"제한시간={cfg['time_limit_s']}s")
        for line in describe(self.tree.root):
            print("  " + line)

        # 센서 워밍업
        warm = 0
        while r.step(self.ts) != -1:
            warm += 1
            if warm >= 3 and self.localizer.reset_origin(r.getTime()):
                break
        self.log("원점 설정 완료. 탐색 시작.")

        draw_every = int(cfg["draw_every"])
        while r.step(self.ts) != -1:
            t = r.getTime()
            self._tick += 1

            ranges = self.lidar.getRangeImage()
            self.scan_monitor.update(t, ranges)
            # 명령은 있는데 스캔이 정지해 있으면 바퀴가 헛도는 것 → 엔코더 적분 정지
            freeze = self.scan_monitor.static and self.motion.moving_cmd() and self.scan_monitor.static_for(t) > 0.3
            pose = self.localizer.update(t, freeze=freeze)
            # Scan-to-Map Matching: 지도에 이 스캔을 반영하기 전에 위치를 보정한다
            if t - self._last_match_t >= float(cfg["scan_match_every_s"]) and abs(self.localizer.yaw_rate) < 0.6:
                self._last_match_t = t
                m = self.grid.match_scan(pose, ranges, float(cfg["scan_match_search_m"]), float(cfg["scan_match_step_m"]))
                if m is not None:
                    dx, dy, s0, s1 = m
                    if (dx != 0.0 or dy != 0.0) and s1 - s0 >= float(cfg["scan_match_min_gain"]):
                        a = float(cfg.get("scan_match_apply", 0.7))
                        self.localizer.correct(a * dx, a * dy)
                        pose = self.localizer.pose()
                        self.stats["scan_corrections"] += 1
            self.grid.update(pose, ranges, self.localizer.yaw_rate)
            if self._tick % 2 == 0 or self.grid._costmap_dirty:
                self.costmap = self.grid.costmap()

            if self._tick % int(self.cfg.get("camera_period_mult", 2)) == 0:
                self.grid.mark_viewed(pose, self.costmap)
                self.detector.detect(pose, t)
                det = self.detector.confirm()
                if det is not None:
                    self.bb.set("detection", det)
                    self._last_confirm_t = t
                elif self.bb.get("detection") is not None:
                    # 접근 중이 아니면 오래된 확정 검출은 만료시킨다
                    if self.bb.get("mode") != "approach" and t - self._last_confirm_t > 2.0:
                        self.bb.set("detection", None)

            self.commands.poll()
            self.bb.set("command_state", self.commands.state)
            if self.commands.pending_target:
                color, count = self.commands.pending_target
                self.commands.pending_target = None
                self.detector.set_targets([color])
                if count:
                    self.bb.set("targets_left", count)
                self.bb.set("detection", None)
            if self.commands.want_status:
                self.commands.want_status = False
                self.log(f"상태: {self.bb.get('state_label')} pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}deg) "
                         f"남은 목표={self.bb.get('targets_left')} 탐색률={self.grid.explored_ratio()*100:.1f}%")
                for line in describe(self.tree.root):
                    print("  " + line)

            level = self.motion.apply_safety(ranges)
            self.stats["min_front_m"] = min(self.stats["min_front_m"], self.motion.front_min)

            self.bb.set("t", t)
            self.bb.set("pose", pose)
            self.bb.set("ranges", ranges)
            self.bb.set("safety_level", level)
            self.bb.set("front_min", self.motion.front_min)
            self.bb.set("scan_static_s", self.scan_monitor.static_for(t))
            gt = self._gt_pose()
            if gt is not None:
                self.bb.set("gt_pose", gt)
                self.reporter.gt_errors.append((t, math.hypot(gt[0] - pose[0], gt[1] - pose[1]),
                                                abs(wrap_angle(gt[2] - pose[2]))))

            self.tree.tick()
            self.bridge.update()

            self.reporter.record_pose(t, pose)
            if self._tick % draw_every == 0:
                self.reporter.draw(self.costmap, self.bb, self.detector)
            if t - self._last_log_t >= float(cfg["log_every_s"]):
                self._last_log_t = t
                extra = ""
                if gt is not None:
                    extra = (f" | GT오차 {math.hypot(gt[0]-pose[0], gt[1]-pose[1]):.3f} m / {math.degrees(abs(wrap_angle(gt[2]-pose[2]))):.1f} deg"
                             f" GT=({gt[0]:.2f},{gt[1]:.2f}) cmd_v={self.motion.cmd_v:.2f} cmd_w={self.motion.cmd_w:+.2f} "
                             f"scale={self.motion.speed_scale:.1f} static={self.scan_monitor.static_for(t):.1f}s"
                             f" run={[l.strip() for l in describe(self.tree.root) if '[R]' in l and 'Action' in l]}")
                self.log(f"{self.bb.get('state_label')} | pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}deg) "
                         f"탐색률={self.grid.explored_ratio()*100:.1f}% 시야={self.grid.viewed_ratio()*100:.0f}% 남은목표={self.bb.get('targets_left')} "
                         f"헛돔={self.localizer.frozen_dist:.2f}m 보정={self.localizer.correction_dist:.2f}m{extra}")

            save_every = float(cfg.get("save_every_s", 0.0))
            if save_every > 0 and t - self._last_save_t >= save_every:
                self._last_save_t = t
                self.reporter.save(self.costmap, self.bb, self.detector, self.stats, tag=f"t{int(t):04d}")
            if self.bb.get("done") and not self.bb.has("done_t"):
                self.bb.set("done_t", t)


if __name__ == "__main__":
    App().run()
