"""미션용 Behavior Tree 잎 노드와 트리 조립 (담당 D).

트리 구조:
Root: ReactiveSequence
├─ SafetyGate: ReactiveSequence
│   ├─ ManualOverride         웹 수동 조종 모드면 (v, w)를 직접 적용 (RUNNING)
│   ├─ NotExternallyStopped   외부 정지 명령이면 정지 유지 (RUNNING)
│   ├─ Recovery(NotStuck, Escape)  끼임이면 후진·회전 후 재계획
│   └─ FrontClear             전방 막힘이면 정지 대기, 오래 막히면 재계획 요청
└─ Mission: ReactiveSelector
    ├─ ForcedReturn: Sequence [조건, ReturnHome...]
    ├─ Approach:     Sequence [조건, ProtectTarget, PlanToStandoff, FollowPath, VisualApproach, MarkRescued]
    ├─ Explore:      Sequence [조건, SelectFrontier, PlanToGoal(unknown 허용), FollowPath]
    ├─ ReturnHome:   Sequence [조건, PlanHome, FollowPath, AlignToStartYaw, Finish]
    └─ Idle
"""
import math

from bt_core import (Status, Sequence, ReactiveSequence, ReactiveSelector,
                     Recovery, Condition, Action, BehaviorTree)
from planner import plan_world, path_blocked
from mapping import UNKNOWN
from config import wrap_angle


class MissionNodes:
    """잎 노드들이 공유하는 컨텍스트. app은 tb3_mission.App."""

    def __init__(self, app):
        self.app = app
        self.bb = app.bb
        self.cfg = app.cfg
        self._wait_start = None
        self._escape_phase = 0
        self._escape_t0 = 0.0
        self._pause_t0 = None
        self._no_frontier = 0
        self._blacklist_cleared = False
        self._follow_ticks = 0
        self._visual_lost_t0 = None
        self._announce_fail = 0
        self._force_escape = False
        self._escape_dir = 1.0
        self._goal_fail = {}
        self._home_fail = 0
        self._visual_t0 = None
        self._scan_t0 = None
        self._scan_prev_yaw = None
        self._scan_accum_yaw = 0.0
        self._scan_saw_target_t0 = None
        self._path_target_t0 = None
        self._log = app.log

    # ---------- 편의 ----------
    def now(self):
        return self.bb.get("t", 0.0)

    def pose(self):
        return self.bb.get("pose")

    def label(self, s):
        self.bb.set("state_label", s)

    # ---------- Safety ----------
    def manual_override(self):
        """웹 수동 조종. 안전 계층(전방 정지·감속)은 set_cmd 안에서 그대로 적용된다."""
        if self.bb.get("command_state") != "manual":
            return Status.SUCCESS
        v, w = self.app.commands.manual_cmd()
        self.app.motion.clear_stuck()
        self.app.motion.set_cmd(v, w)
        self.bb.set("mode", None)
        self.label(f"MANUAL v={v:+.2f} w={w:+.2f}")
        return Status.RUNNING

    def not_externally_stopped(self):
        if self.bb.get("command_state") == "stopped":
            self.app.motion.stop()
            self.label("STOPPED (외부 명령)")
            return Status.RUNNING
        return Status.SUCCESS

    def not_stuck(self):
        if self.bb.get("done"):
            return Status.SUCCESS
        if self._force_escape:
            self._force_escape = False
            self._log("전방 막힘 지속 → 회피 기동")
            self.app.motion.clear_stuck()
            return Status.FAILURE
        if self.app.motion.is_stuck(self.now(), self.bb.get("scan_static_s", 0.0), self.pose()):
            self._log(f"끼임/헛돔 감지 → 복구 행동(후진·회피), 전방 0.24 m 가상 장애물 기록")
            self.app.motion.clear_stuck()
            self.app.stats["stuck_events"] += 1
            px, py, yaw = self.pose()
            # LiDAR에 안 보이는 낮은 장애물(침대 프레임 등)로 판단하고 지도에 남긴다
            self.app.grid.add_virtual_obstacle(px + 0.24 * math.cos(yaw), py + 0.24 * math.sin(yaw), 0.06)
            return Status.FAILURE
        return Status.SUCCESS

    def escape(self):
        m = self.app.motion
        t = self.now()
        if self._escape_phase == 0:
            self._escape_t0 = t
            self._escape_phase = 1
            self._escape_dir = m.freer_side()
            self.label("RECOVERY 후진")
        if self._escape_phase == 1:
            # 후방에 16cm 이상 여유가 있을 때만 후진 (-0.08 m/s)
            b = getattr(m, "back_min", float("inf"))
            if b > 0.16:
                m.back_off(-0.08)
            else:
                m.stop()
            # 1.2초 후진했거나 후방이 막히면 회전 탈출 단계로 전환
            if t - self._escape_t0 > 1.2 or b <= 0.16:
                self._escape_phase = 2
                self._escape_t0 = t
                self._escape_dir = m.freer_side()
                self.label("RECOVERY 회전")
            return Status.RUNNING
        if self._escape_phase == 2:
            m.set_cmd(0.0, 1.0 * self._escape_dir)
            dt = t - self._escape_t0
            # 최소 0.5초(약 30도) 이상 회전하고 전방 시야가 열렸거나(0.40m 이상), 최대 2.2초(약 125도) 회전하면 탈출 완료
            if (dt >= 0.5 and m.front_min > 0.40) or dt > 2.2:
                m.stop()
                self._escape_phase = 0
                self.bb.set("replan", True)
                self._blacklist_current_goal()
                return Status.SUCCESS
            return Status.RUNNING
        return Status.SUCCESS

    def escape_reset(self):
        self._escape_phase = 0

    def front_clear(self):
        """전방 막힘은 motion.set_cmd가 전진 속도를 0으로 만들어 처리한다.
        여기서는 '전진하려는데 막힌' 시간을 재서, 오래 지속되면 회피 기동을 요청한다.
        제자리 회전과 후진은 막힘 중에도 허용되므로 Mission tick을 막지 않는다."""
        if self.bb.get("done"):
            return Status.SUCCESS
        m = self.app.motion
        t = self.now()
        if not m.blocked_forward or self.bb.get("mode") == "visual":
            self._wait_start = None
            return Status.SUCCESS
        if self._wait_start is None:
            self._wait_start = t
            self.app.stats["block_events"] += 1
        waited = t - self._wait_start
        self.bb.set("wait_label", f"WAIT 전방 {m.front_min:.2f} m ({waited:.0f}s)")
        if waited > float(self.cfg["wait_before_replan_s"]):
            self._wait_start = None
            self._force_escape = True
            self._blacklist_current_goal()
        return Status.SUCCESS

    def _blacklist_current_goal(self):
        """같은 frontier 목표를 향해 끼임·막힘이 반복될 때만 블랙리스트에 넣는다.
        (경로 중간의 낮은 장애물 때문에 목적지까지 버리는 것을 막기 위함)"""
        g = self.bb.get("goal")
        if not g or self.bb.get("goal_kind") != "frontier":
            return
        key = (round(g[0], 1), round(g[1], 1))
        self._goal_fail[key] = self._goal_fail.get(key, 0) + 1
        if self._goal_fail[key] >= int(self.cfg.get("goal_fail_blacklist", 3)):
            self._log(f"목표 {key} 반복 실패 → 블랙리스트")
            self.app.blacklist_goal(g)
            self._goal_fail[key] = 0

    # ---------- 조건 ----------
    def cond_forced_return(self):
        if self.bb.get("done"):
            return False
        if self.bb.get("command_state") == "return":
            return True
        limit = float(self.cfg["time_limit_s"]) * float(self.cfg["return_at_ratio"])
        return self.now() > limit

    def cond_targets_left(self):
        return self.bb.get("targets_left", 0) > 0

    def cond_target_confirmed(self):
        if not self.cond_targets_left():
            return False
        det = self.bb.get("detection")
        return bool(det and det.get("confirmed"))

    def cond_return_home(self):
        if self.bb.get("done"):
            return True
        if not self.cond_targets_left():
            return True
        return bool(self.bb.get("frontier_exhausted"))

    # ---------- Approach ----------
    def protect_target(self):
        det = self.bb.get("detection")
        x, y = det["world"]
        self.bb.set("approach_target", (x, y))
        self.bb.set("mode", "approach")
        last = self.bb.get("last_announced")
        if last is None or math.hypot(x - last[0], y - last[1]) > 0.3:
            self.bb.set("last_announced", (x, y))
            self._announce_fail = 0
            self._log(f"목표 확정 ({det['color']}) 위치=({x:.2f}, {y:.2f}) 거리={det['dist']:.2f} m "
                      f"(연속 {det.get('n', 0)}프레임)")
            snap = self.app.detector.save_snapshot(f"{self.now():.0f}s")
            if snap:
                self._log(f"검출 스냅샷 저장: {snap}")
        return Status.SUCCESS

    def plan_to_standoff(self):
        px, py, _ = self.pose()
        tx, ty = self.bb.get("approach_target")
        d = math.hypot(tx - px, ty - py)
        so = float(self.cfg["approach_standoff_m"])
        if d < 1e-3:
            return Status.FAILURE
        base = math.atan2(py - ty, px - tx)      # 목표에서 로봇을 향하는 방향
        self.label("APPROACH 경로 계획")
        # 로봇 쪽 standoff부터 시작해 목표 둘레의 대안 지점을 차례로 시도한다
        for radius in (so, so * 1.4):
            for k in (0, 1, -1, 2, -2, 3, -3, 4):
                a = base + k * (math.pi / 4.0)
                gx = tx + radius * math.cos(a)
                gy = ty + radius * math.sin(a)
                self.bb.set("goal", (gx, gy))
                self.bb.set("goal_kind", "standoff")
                if self._plan(allow_unknown=True, quiet=True) == Status.SUCCESS:
                    self.app.grid.add_virtual_obstacle(tx, ty, float(self.cfg["protect_radius_m"]))
                    return Status.SUCCESS
        self._announce_fail += 1
        if self._announce_fail >= 3:
            self._log(f"목표 ({tx:.2f}, {ty:.2f})까지 경로 없음 → 45초 동안 억제하고 탐색 계속")
            self.app.detector.suppress(tx, ty, self.now())
            self.bb.set("detection", None)
            self.bb.set("last_announced", None)
            self._announce_fail = 0
        return Status.FAILURE

    def visual_approach(self):
        self.bb.set("mode", "visual")
        self.label("APPROACH 시각 접근")
        det = self.app.detector.last_target
        t = self.now()
        if self._visual_t0 is None:
            self._visual_t0 = t
        if t - self._visual_t0 > float(self.cfg.get("visual_timeout_s", 12.0)):
            tx, ty = self.bb.get("approach_target")
            px, py, _ = self.pose()
            d = math.hypot(tx - px, ty - py)
            self._log(f"시각 접근 제한 시간 초과 (목표까지 {d:.2f} m) → 구출 처리")
            self._visual_t0 = None
            self.bb.set("mode", None)
            return Status.SUCCESS
        if det is None:
            if self._visual_lost_t0 is None:
                self._visual_lost_t0 = t
            # 놓치면 목표 방향으로 회전하며 찾는다
            tx, ty = self.bb.get("approach_target")
            px, py, yaw = self.pose()
            target_yaw = math.atan2(ty - py, tx - px)
            self.app.motion.rotate_to(self.pose(), target_yaw, tol_deg=4.0)
            if t - self._visual_lost_t0 > 6.0:
                self._log("시각 접근 중 목표 상실 → standoff 위치에서 구출 처리")
                self._visual_lost_t0 = None
                self.bb.set("mode", None)
                return Status.SUCCESS
            return Status.RUNNING
        self._visual_lost_t0 = None
        if self.app.motion.approach_visual(det):
            self._log(f"사과 앞 도착: 추정 거리 {det['dist']:.2f} m, 전방 LiDAR {self.app.motion.front_min:.2f} m")
            self._visual_t0 = None
            self.bb.set("mode", None)
            return Status.SUCCESS
        return Status.RUNNING

    def visual_reset(self):
        self._visual_lost_t0 = None
        self._visual_t0 = None
        self.bb.set("mode", None)

    def mark_rescued(self):
        t = self.now()
        if self._pause_t0 is None:
            self._pause_t0 = t
            tx, ty = self.bb.get("approach_target")
            det = self.app.detector.last_target
            if det is not None:
                tx, ty = det["world"]
            self.app.detector.mark_rescued(tx, ty)
            left = self.bb.get("targets_left") - 1
            self.bb.set("targets_left", left)
            self.bb.get("rescued").append({"x": tx, "y": ty, "t": t})
            self.app.motion.stop()
            self._log(f"구출 완료 #{len(self.bb.get('rescued'))} 위치=({tx:.2f}, {ty:.2f}) 남은 목표={left}")
            self.bb.set("detection", None)
        self.label("RESCUE 기록 중")
        if t - self._pause_t0 >= float(self.cfg["rescue_pause_s"]):
            self._pause_t0 = None
            self.bb.set("frontier_exhausted", False)
            self.bb.set("mode", None)
            self.bb.set("last_announced", None)
            self.app.clear_blacklist()
            return Status.SUCCESS
        return Status.RUNNING

    # ---------- Explore ----------
    def select_frontier(self):
        cm = self.app.costmap
        fr = self.app.grid.frontiers(cm, self.pose(), self.app.blacklist, self.now(),
                                     rescued=self.bb.get("rescued", []))
        if not fr:
            self._no_frontier += 1
            if self._no_frontier == 2 and not self._blacklist_cleared:
                self.app.clear_blacklist()
                self._blacklist_cleared = True
                self._log("frontier 없음 → 블랙리스트 초기화 후 재시도")
            if self._no_frontier >= 4:
                if not self.bb.get("frontier_exhausted"):
                    self._log("frontier 소진 → 탐색 종료")
                self.bb.set("frontier_exhausted", True)
            self.app.motion.stop()
            return Status.FAILURE
        self._no_frontier = 0
        x, y, size, d = fr[0]
        self.bb.set("goal", (x, y))
        self.bb.set("goal_kind", "frontier")
        self.bb.set("frontiers", fr[:12])
        self.label(f"🍎 사과 탐색 이동 ({x:.1f}, {y:.1f})")
        return Status.SUCCESS

    def plan_explore(self):
        s = self._plan(allow_unknown=True)
        if s == Status.FAILURE:
            self.app.blacklist_goal(self.bb.get("goal"))
        return s

    # ---------- Return ----------
    def plan_home(self):
        self.bb.set("goal", (0.0, 0.0))
        self.bb.set("goal_kind", "home")
        self.label("RETURN 경로 계획")
        s = self._plan(allow_unknown=False, quiet=True)
        if s == Status.FAILURE:
            s = self._plan(allow_unknown=True, quiet=True)
        if s == Status.SUCCESS:
            self._home_fail = 0
            return s
        self._home_fail += 1
        px, py, _ = self.pose()
        d = math.hypot(px, py)
        if self._home_fail == 1 or self._home_fail % 30 == 0:
            self._log(f"복귀 경로 계획 실패 {self._home_fail}회 (원점까지 {d:.2f} m)")
        if d < 0.5:
            self._log("원점 0.5 m 이내이므로 복귀 완료로 처리")
            self.bb.set("path", [(px, py), (0.0, 0.0)])
            self._home_fail = 0
            return Status.SUCCESS
        if self._home_fail >= 3:
            # 가상 장애물이 로봇을 가두었을 수 있으므로 제거하고 회피 기동 후 재시도
            if self.app.grid.virtual_obstacles:
                self._log("가상 장애물을 모두 제거하고 복귀 재시도")
                self.app.grid.virtual_obstacles = []
                self.app.grid._costmap_dirty = True
                self.app.costmap = self.app.grid.costmap(force=True)
            self._force_escape = True
            self._home_fail = 0
        return Status.FAILURE

    def align_home(self):
        self.label("RETURN 방향 정렬")
        if self.app.motion.rotate_to(self.pose(), 0.0):
            return Status.SUCCESS
        return Status.RUNNING

    def finish(self):
        if not self.bb.get("done"):
            self.bb.set("done", True)
            self.app.motion.stop()
            px, py, yaw = self.pose()
            self._log(f"미션 종료: 복귀 오차 {math.hypot(px, py):.3f} m, "
                      f"방향 오차 {math.degrees(abs(yaw)):.1f} deg, 구출 {len(self.bb.get('rescued'))}개")
            self.app.finish_report()
        self.label("DONE")
        self.app.motion.stop()
        return Status.RUNNING

    def idle(self):
        self.app.motion.stop()
        self.label("IDLE")
        return Status.RUNNING

    # ---------- 공통 ----------
    def _plan(self, allow_unknown, quiet=False):
        px, py, _ = self.pose()
        goal = self.bb.get("goal")
        path = plan_world(self.app.grid, self.app.costmap, (px, py), goal, allow_unknown=allow_unknown)
        if not path:
            if not quiet:
                self._log(f"경로 계획 실패 → 목표 {goal[0]:.2f}, {goal[1]:.2f}")
            self.bb.set("path", None)
            return Status.FAILURE
        self.bb.set("path", path)
        self.bb.set("replan", False)
        self.app.motion.reset_path()
        self._follow_ticks = 0
        return Status.SUCCESS

    def follow_path(self):
        path = self.bb.get("path")
        if not path:
            return Status.FAILURE
        if self.bb.get("replan"):
            self.bb.set("replan", False)
            self._log("재계획 플래그 → 경로 추종 중단")
            return Status.FAILURE
        self._follow_ticks += 1
        kind = self.bb.get("goal_kind")
        if self._follow_ticks % 8 == 0:
            if path_blocked(path, self.app.grid, self.app.costmap, self.app.motion._idx, self.pose()):
                self._log("경로가 새 장애물에 막힘 → 재계획")
                self.app.stats["replans"] += 1
                return Status.FAILURE
            if kind == "frontier" and self._follow_ticks % 24 == 0 and self._goal_observed():
                self.app.motion.stop()
                return Status.SUCCESS
        # 주행 중 빨간 사과가 시야에 들어온 경우: 속도를 줄여 5프레임 확정을 돕는다
        if kind == "frontier" and self.app.detector.last_target is not None:
            t = self.now()
            if self._path_target_t0 is None:
                self._path_target_t0 = t
            if t - self._path_target_t0 < 1.0:
                self.label(f"🍎 사과 포착! 확정 대기 중... ({self.app.detector.last_target['dist']:.2f}m)")
                self.app.motion.set_cmd(0.04, 0.0)
                return Status.RUNNING
        else:
            self._path_target_t0 = None

        if kind == "frontier":
            self.label(self.bb.get("state_label", "🍎 사과 탐색 주행"))
        elif kind == "home":
            self.label("RETURN 복귀 주행")
        done = self.app.motion.follow(self.pose(), path)
        return Status.SUCCESS if done else Status.RUNNING

    def scan_for_apple(self):
        """방/frontier에 도착했을 때 제자리 360도 회전하며 빨간 사과를 능동 탐색한다.
        카메라 FOV가 60도이므로 전진 주행만으로는 놓치기 쉬운 주변 방의 사과를 찾는다."""
        # 1. 이미 목표가 확정되었거나 남은 목표가 없으면 성공 종료
        if self.bb.get("detection") and self.bb.get("detection").get("confirmed"):
            self.scan_reset()
            return Status.SUCCESS
        if not self.cond_targets_left():
            self.scan_reset()
            return Status.SUCCESS

        t = self.now()
        px, py, yaw = self.pose()

        # 2. 회전 중 카메라에 빨간 사과가 감지되면 즉시 정지하고 5프레임 확정을 대기
        target = self.app.detector.last_target
        if target is not None:
            self.app.motion.stop()
            if self._scan_saw_target_t0 is None:
                self._scan_saw_target_t0 = t
                self._log(f"🍎 사과 발견! ({target['dist']:.2f}m) 확정 대기 중...")
            self.label(f"🍎 사과 포착! 확정 대기 ({target['dist']:.2f}m)")
            if self.bb.get("detection") and self.bb.get("detection").get("confirmed"):
                self.scan_reset()
                return Status.SUCCESS
            if t - self._scan_saw_target_t0 > 1.5:
                self._scan_saw_target_t0 = None
            else:
                return Status.RUNNING

        self._scan_saw_target_t0 = None

        if self._scan_t0 is None:
            self._scan_t0 = t
            self._scan_prev_yaw = yaw
            self._scan_accum_yaw = 0.0

        # 회전각 누적
        dyaw = wrap_angle(yaw - self._scan_prev_yaw)
        self._scan_accum_yaw += abs(dyaw)
        self._scan_prev_yaw = yaw

        deg = int(math.degrees(self._scan_accum_yaw))
        # 360도(2*pi) 회전 완료 또는 12초 초과 시 탐색 완료
        if self._scan_accum_yaw >= 2.0 * math.pi or (t - self._scan_t0) > 12.0:
            self._log(f"🍎 360° 사과 탐색 회전 완료 ({deg}°) → 다음 구역으로 이동")
            self.scan_reset()
            return Status.SUCCESS

        # 초당 약 0.55 rad (약 31.5도/s)로 부드러운 제자리 회전
        self.app.motion.set_cmd(0.0, 0.55)
        self.label(f"🍎 사과 탐색 회전 중 ({deg}°/360°)")
        return Status.RUNNING

    def scan_reset(self):
        self._scan_t0 = None
        self._scan_prev_yaw = None
        self._scan_accum_yaw = 0.0
        self._scan_saw_target_t0 = None
        self.app.motion.stop()

    def _goal_observed(self):
        """frontier 목표 주변에 미관측 셀이 더 이상 없으면 True."""
        gx, gy = self.bb.get("goal")
        r, c = self.app.grid.world_to_grid(gx, gy)
        cm = self.app.costmap
        g = self.app.grid
        r0, r1 = max(0, r - 3), min(g.rows, r + 4)
        c0, c1 = max(0, c - 3), min(g.cols, c + 4)
        return not (cm[r0:r1, c0:c1] == UNKNOWN).any()


def build_tree(app):
    n = MissionNodes(app)

    safety = ReactiveSequence("SafetyGate", [
        Action("ManualOverride", n.manual_override),
        Action("NotExternallyStopped", n.not_externally_stopped),
        Recovery("StuckRecovery",
                 Action("NotStuck", n.not_stuck),
                 Action("Escape", n.escape, n.escape_reset),
                 max_retries=5),
        Action("FrontClear", n.front_clear),
    ])

    return_seq = [
        Action("PlanHome", n.plan_home),
        Action("FollowPath", n.follow_path),
        Action("AlignToStartYaw", n.align_home),
        Action("Finish", n.finish),
    ]

    forced_return = Sequence("ForcedReturn", [Condition("ForcedReturn?", n.cond_forced_return)] + return_seq)

    approach = Sequence("Approach", [
        Condition("TargetConfirmed?", n.cond_target_confirmed),
        Action("ProtectTarget", n.protect_target),
        Action("PlanToStandoff", n.plan_to_standoff),
        Action("FollowPath", n.follow_path),
        Action("VisualApproach", n.visual_approach, n.visual_reset),
        Action("MarkRescued", n.mark_rescued),
    ])

    explore = Sequence("Explore", [
        Condition("TargetsLeft?", n.cond_targets_left),
        Action("SelectFrontier", n.select_frontier),
        Action("PlanToGoal", n.plan_explore),
        Action("FollowPath", n.follow_path),
        Action("ScanForApple", n.scan_for_apple, n.scan_reset),
    ])

    return_home = Sequence("ReturnHome", [Condition("ReturnHome?", n.cond_return_home)] + [
        Action("PlanHome", n.plan_home),
        Action("FollowPath", n.follow_path),
        Action("AlignToStartYaw", n.align_home),
        Action("Finish", n.finish),
    ])

    mission = ReactiveSelector("Mission", [forced_return, approach, explore, return_home,
                                           Action("Idle", n.idle)])
    root = ReactiveSequence("Root", [safety, mission])
    return BehaviorTree(root), n
