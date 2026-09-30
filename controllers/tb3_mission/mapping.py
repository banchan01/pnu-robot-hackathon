"""Bayesian Occupancy Grid, Costmap, Frontier 추출 (담당 A).

지도 표현:
- logodds[row, col]  : 점유 확률의 log-odds. 0이면 사전 확률 0.5.
- observed[row, col] : 한 번이라도 관측된 셀.
- 갱신은 Bayesian 필터의 log-odds 형태  L_t = L_{t-1} + l(z_t)  를 따른다.
  점유 관측이면 l_occ를 더하고 자유 관측이면 l_free를 뺀다.

Costmap 값 규약 (강의 슬라이드):
- 0       자유 공간
- 1~252   팽창 구역 (장애물에 가까울수록 큼)
- 253     내접 충돌 구역
- 254     점유
- 255     미관측
"""
import math
import time

import cv2
import numpy as np

from config import LIDAR_ANGLES, LIDAR_MIN, LIDAR_MAX, LIDAR_OFFSET_X, CAM_FOV, CAM_OFFSET_X

LETHAL = 254
INSCRIBED = 253
UNKNOWN = 255


class OccupancyGrid:
    def __init__(self, cfg):
        self.res = float(cfg["map_res"])
        self.x0, self.y0 = [float(v) for v in cfg["map_origin"]]
        w_m, h_m = [float(v) for v in cfg["map_size"]]
        self.cols = int(round(w_m / self.res))
        self.rows = int(round(h_m / self.res))
        self.logodds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.observed = np.zeros((self.rows, self.cols), dtype=bool)
        self.viewed = np.zeros((self.rows, self.cols), dtype=bool)      # 카메라 시야가 한 번이라도 훑은 셀

        self.l_occ = float(cfg["l_occ"])
        self.l_free = float(cfg["l_free"])
        self.l_clamp = float(cfg["l_clamp"])
        self.occ_threshold = float(cfg["occ_threshold"])
        self.max_free_range = float(cfg["max_free_range"])
        self.yaw_rate_skip = float(cfg["yaw_rate_skip"])
        self.inflate_cells = int(cfg["inflate_cells"])
        self.soft_cells = int(cfg["soft_cells"])
        self.min_frontier_cells = int(cfg["min_frontier_cells"])
        self.frontier_min_dist = float(cfg["frontier_min_dist_m"])
        self.frontier_max_cost = int(cfg.get("frontier_max_cost", 100))

        self.virtual_obstacles = []      # [(x, y, r)] 사과 보호용, costmap 생성 시 매번 반영
        self._step = self.res * 0.5
        self._n_samples = int(math.ceil(LIDAR_MAX / self._step)) + 1
        self._t_axis = np.arange(self._n_samples, dtype=np.float32) * self._step
        self._costmap_cache = None
        self._costmap_dirty = True
        self._solid_dist_cache = None
        self._solid_dirty = True

    # ---------- 좌표 변환 ----------
    def world_to_grid(self, x, y):
        c = int((x - self.x0) / self.res)
        r = int((y - self.y0) / self.res)
        return r, c

    def grid_to_world(self, r, c):
        return (self.x0 + (c + 0.5) * self.res, self.y0 + (r + 0.5) * self.res)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    # ---------- Bayesian 갱신 ----------
    def update(self, pose, ranges, yaw_rate):
        if abs(yaw_rate) > self.yaw_rate_skip:
            return
        x, y, yaw = pose
        r = np.asarray(ranges, dtype=np.float32)
        if r.shape[0] != LIDAR_ANGLES.shape[0]:
            return

        sx = x + LIDAR_OFFSET_X * math.cos(yaw)
        sy = y + LIDAR_OFFSET_X * math.sin(yaw)
        ang = LIDAR_ANGLES + yaw
        finite = np.isfinite(r)
        valid = finite & (r >= LIDAR_MIN)
        hit = valid & (r < LIDAR_MAX - 0.05)

        # 자유 구간 길이: 반사가 있으면 반사 지점 직전까지, 없으면 max_free_range까지
        free_len = np.where(hit, r - self.res, self.max_free_range)
        free_len = np.where(valid | ~finite, free_len, 0.0)   # 최소 거리 미만은 무시
        free_len = np.clip(free_len, 0.0, LIDAR_MAX)

        cos_a = np.cos(ang)[:, None]
        sin_a = np.sin(ang)[:, None]
        t = self._t_axis[None, :]
        mask = t < free_len[:, None]
        px = sx + t * cos_a
        py = sy + t * sin_a
        cols = ((px - self.x0) / self.res).astype(np.int32)
        rows = ((py - self.y0) / self.res).astype(np.int32)
        inb = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        sel = mask & inb
        free_mask = np.zeros((self.rows, self.cols), dtype=bool)
        free_mask[rows[sel], cols[sel]] = True

        hx = sx + r[hit] * np.cos(ang[hit])
        hy = sy + r[hit] * np.sin(ang[hit])
        hc = ((hx - self.x0) / self.res).astype(np.int32)
        hr = ((hy - self.y0) / self.res).astype(np.int32)
        hin = (hr >= 0) & (hr < self.rows) & (hc >= 0) & (hc < self.cols)
        occ_mask = np.zeros((self.rows, self.cols), dtype=bool)
        occ_mask[hr[hin], hc[hin]] = True

        free_mask &= ~occ_mask
        self.logodds[free_mask] -= self.l_free
        self.logodds[occ_mask] += self.l_occ
        np.clip(self.logodds, -self.l_clamp, self.l_clamp, out=self.logodds)
        self.observed |= free_mask | occ_mask
        self._costmap_dirty = True
        self._solid_dirty = True

    def occupancy_prob(self):
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def is_occupied(self):
        return self.observed & (self.logodds > self.occ_threshold)

    # ---------- Costmap ----------
    def add_virtual_obstacle(self, x, y, radius):
        self.virtual_obstacles.append((float(x), float(y), float(radius)))
        self._costmap_dirty = True

    def costmap(self, force=False):
        if not force and not self._costmap_dirty and self._costmap_cache is not None:
            return self._costmap_cache
        occ = self.is_occupied().astype(np.uint8)
        for (vx, vy, vr) in self.virtual_obstacles:
            rr, cc = self.world_to_grid(vx, vy)
            cv2.circle(occ, (cc, rr), max(1, int(vr / self.res)), 1, -1)

        free_src = (1 - occ).astype(np.uint8)
        dist = cv2.distanceTransform(free_src, cv2.DIST_L2, 3)   # 장애물까지 거리(셀)
        cm = np.zeros((self.rows, self.cols), dtype=np.uint8)

        soft = self.soft_cells
        infl = self.inflate_cells
        in_soft = (dist > infl) & (dist <= soft)
        if soft > infl:
            grad = 1.0 - (dist - infl) / float(soft - infl)
            cm[in_soft] = np.clip((grad[in_soft] * 200.0).astype(np.uint8), 1, 252)
        cm[(dist <= infl) & (dist > 0)] = INSCRIBED
        cm[occ.astype(bool)] = LETHAL
        unknown = ~self.observed
        cm[unknown & (dist > infl)] = UNKNOWN
        self._costmap_cache = cm
        self._costmap_dirty = False
        return cm

    # ---------- Frontier ----------
    def wavefront_distance(self, costmap, start_rc, max_iter=900):
        """로봇 위치에서 통행 가능 셀을 따라 퍼지는 BFS 거리(셀 단위). 도달 불가면 -1."""
        traversable = (costmap < INSCRIBED)
        r0, c0 = start_rc
        # 시작 셀이 팽창 구역 안이면 근처 2셀을 통행 가능으로 허용
        r_lo, r_hi = max(0, r0 - 2), min(self.rows, r0 + 3)
        c_lo, c_hi = max(0, c0 - 2), min(self.cols, c0 + 3)
        traversable[r_lo:r_hi, c_lo:c_hi] |= (costmap[r_lo:r_hi, c_lo:c_hi] != LETHAL)

        dist = np.full((self.rows, self.cols), -1, dtype=np.int32)
        if not self.in_bounds(r0, c0):
            return dist
        front = np.zeros((self.rows, self.cols), dtype=np.uint8)
        front[r0, c0] = 1
        visited = front.astype(bool)
        dist[r0, c0] = 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        trav_u8 = traversable.astype(np.uint8)
        for i in range(1, max_iter):
            nxt = cv2.dilate(front, kernel) & trav_u8
            nxt_b = nxt.astype(bool) & ~visited
            if not nxt_b.any():
                break
            dist[nxt_b] = i
            visited |= nxt_b
            front = nxt_b.astype(np.uint8)
        return dist

    def frontiers(self, costmap, pose, blacklist=(), now=0.0):
        """점수순으로 정렬된 frontier 목표 후보 [(x, y, size, dist_m)]."""
        # 후보 셀은 장애물에서 충분히 떨어진 자유 셀만: 원거리 벽의 빔 간격 틈이 frontier로 잡히는 것을 막는다
        free = (costmap < self.frontier_max_cost)
        unknown = (costmap == UNKNOWN)
        kernel = np.ones((3, 3), dtype=np.uint8)
        near_unknown = cv2.dilate(unknown.astype(np.uint8), kernel).astype(bool)
        frontier = (free & near_unknown).astype(np.uint8)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier, connectivity=8)
        if n <= 1:
            return []
        r0, c0 = self.world_to_grid(pose[0], pose[1])
        dist = self.wavefront_distance(costmap, (r0, c0))

        out = []
        for k in range(1, n):
            size = int(stats[k, cv2.CC_STAT_AREA])
            if size < self.min_frontier_cells:
                continue
            ys, xs = np.where(labels == k)
            cy, cx = centroids[k][1], centroids[k][0]
            # 중심에 가장 가까운 실제 frontier 셀을 목표로 삼는다
            j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            rr, cc = int(ys[j]), int(xs[j])
            d_cells = dist[rr, cc]
            if d_cells < 0:
                # 대표 셀이 도달 불가면 컴포넌트 안에서 도달 가능한 셀을 찾는다
                reach = dist[ys, xs]
                ok = np.where(reach >= 0)[0]
                if ok.size == 0:
                    continue
                j = int(ok[np.argmin(reach[ok])])
                rr, cc = int(ys[j]), int(xs[j])
                d_cells = dist[rr, cc]
            d_m = d_cells * self.res
            if d_m < self.frontier_min_dist:
                continue
            wx, wy = self.grid_to_world(rr, cc)
            skip = False
            for (bx, by, br, bexp) in blacklist:
                if bexp > now and (wx - bx) ** 2 + (wy - by) ** 2 < br * br:
                    skip = True
                    break
            if skip:
                continue
            score = size / (d_m + 1.0)
            out.append((wx, wy, size, d_m, score))
        out.sort(key=lambda e: -e[4])
        return [(e[0], e[1], e[2], e[3]) for e in out]

    def snap_to_traversable(self, costmap, r, c, radius=8):
        """목표 셀이 통행 불가면 반경 내 가장 가까운 통행 가능 셀로 옮긴다."""
        if self.in_bounds(r, c) and costmap[r, c] < INSCRIBED:
            return r, c
        best = None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if self.in_bounds(rr, cc) and costmap[rr, cc] < INSCRIBED:
                    d = dr * dr + dc * dc
                    if best is None or d < best[0]:
                        best = (d, rr, cc)
        if best is None:
            return None
        return best[1], best[2]

    def explored_ratio(self):
        return float(self.observed.mean())

    # ---------- 카메라 시야 지도 (Sweep 단계용) ----------
    def mark_viewed(self, pose, costmap, max_range=2.0):
        """카메라 시야 원뿔(±FOV/2, max_range) 안에서 장애물에 가려지지 않은 셀을 viewed로 표시한다."""
        x, y, yaw = pose
        cx = x + CAM_OFFSET_X * math.cos(yaw)
        cy = y + CAM_OFFSET_X * math.sin(yaw)
        angs = yaw + np.linspace(-CAM_FOV / 2.0, CAM_FOV / 2.0, 61)
        ts = np.arange(0.1, max_range, self.res)
        px = cx + ts[None, :] * np.cos(angs)[:, None]
        py = cy + ts[None, :] * np.sin(angs)[:, None]
        cols = np.clip(((px - self.x0) / self.res).astype(np.int32), 0, self.cols - 1)
        rows = np.clip(((py - self.y0) / self.res).astype(np.int32), 0, self.rows - 1)
        blocked = (costmap[rows, cols] == LETHAL)
        before_hit = np.cumsum(blocked, axis=1) == 0        # 첫 장애물 이전 구간만
        self.viewed[rows[before_hit], cols[before_hit]] = True

    def viewed_ratio(self):
        free = self.observed & ~(self.logodds > self.occ_threshold)
        n = free.sum()
        return float((free & self.viewed).sum() / n) if n else 0.0

    def sweep_targets(self, costmap, pose, blacklist=(), now=0.0, min_cells=12):
        """카메라가 아직 보지 못한 비점유 관측 셀 덩어리 [(x, y, size, dist_m)] (가까운 순).
        벽 옆 틈(팽창 구역)도 후보에 포함한다. 목표는 계획 단계에서 근처 통행 가능 셀로 스냅된다."""
        cand = (costmap != LETHAL) & (costmap != UNKNOWN) & self.observed & ~self.viewed
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        if n <= 1:
            return []
        r0, c0 = self.world_to_grid(pose[0], pose[1])
        dist = self.wavefront_distance(costmap, (r0, c0))
        # 통행 불가 셀(팽창 구역)에도 근처 통행 가능 셀의 거리를 부여 (9x9 최대 필터)
        dist = cv2.dilate(dist.astype(np.float32), np.ones((9, 9), np.uint8)).astype(np.int32)
        out = []
        for k in range(1, n):
            size = int(stats[k, cv2.CC_STAT_AREA])
            if size < min_cells:
                continue
            ys, xs = np.where(labels == k)
            reach = dist[ys, xs]
            ok = np.where(reach >= 0)[0]
            if ok.size == 0:
                continue
            # 덩어리 안에서 로봇과 가장 가까운 도달 가능 셀을 대표점으로 (탐욕적 커버리지)
            j = int(ok[np.argmin(reach[ok])])
            rr, cc = int(ys[j]), int(xs[j])
            d_m = dist[rr, cc] * self.res
            if d_m < 0.3:
                # 바로 옆이면 제자리 회전만으로 충분: 대표점을 조금 안쪽 셀로
                far = ok[np.argsort(reach[ok])]
                j = int(far[min(len(far) - 1, 12)])
                rr, cc = int(ys[j]), int(xs[j])
                d_m = dist[rr, cc] * self.res
            wx, wy = self.grid_to_world(rr, cc)
            if any(bexp > now and (wx - bx) ** 2 + (wy - by) ** 2 < br * br for (bx, by, br, bexp) in blacklist):
                continue
            out.append((wx, wy, size, d_m))
        out.sort(key=lambda e: e[3])
        return out

    # ---------- Scan-to-Map Matching ----------
    def _solid_dist(self):
        """확신 있는 점유 셀까지의 거리 변환(셀 단위). 지도가 바뀔 때만 다시 계산한다."""
        if self._solid_dist_cache is not None and not self._solid_dirty:
            return self._solid_dist_cache
        solid = (self.observed & (self.logodds > 1.5)).astype(np.uint8)
        if solid.sum() < 50:
            return None
        self._solid_dist_cache = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 3)
        self._solid_dirty = False
        return self._solid_dist_cache

    def match_scan(self, pose, ranges, search_m=0.06, step_m=0.01, min_hits=40, sigma_cells=1.5,
                   yaw_search_deg=1.0, yaw_step_deg=0.5):
        """상관 스캔 매칭 (x, y, yaw), 완전 벡터화.
        확신 있는 점유 셀까지의 거리 변환 위에서 스캔 반사점이 가장 잘 얹히는 (dx, dy, dyaw)를 찾는다.
        자이로 적분 방향의 누적 오차도 여기서 보정한다.
        반환: (dx, dy, dyaw, score_before, score_after) 또는 None."""
        x, y, yaw = pose
        r = np.asarray(ranges, dtype=np.float32)
        hit = np.isfinite(r) & (r >= 0.3) & (r < LIDAR_MAX - 0.05)
        if hit.sum() < min_hits:
            return None
        dist = self._solid_dist()
        if dist is None:
            return None
        r = r[hit][::2]
        a0 = LIDAR_ANGLES[hit][::2]
        n = int(round(search_m / step_m))
        offs = (np.arange(-n, n + 1) * step_m).astype(np.float32)
        k = offs.size
        ny = int(round(yaw_search_deg / yaw_step_deg))
        yoffs = np.radians(np.arange(-ny, ny + 1) * yaw_step_deg)

        best = None
        base = None
        for j, dy_yaw in enumerate(yoffs):
            yw = yaw + dy_yaw
            sx = x + LIDAR_OFFSET_X * math.cos(yw)
            sy = y + LIDAR_OFFSET_X * math.sin(yw)
            ex = sx + r * np.cos(a0 + yw)
            ey = sy + r * np.sin(a0 + yw)
            cols = np.clip(((ex[None, :] + offs[:, None] - self.x0) / self.res).astype(np.int32), 0, self.cols - 1)
            rows = np.clip(((ey[None, :] + offs[:, None] - self.y0) / self.res).astype(np.int32), 0, self.rows - 1)
            R = np.broadcast_to(rows[:, None, :], (k, k, rows.shape[1]))
            C = np.broadcast_to(cols[None, :, :], (k, k, cols.shape[1]))
            d = dist[R, C]
            score = np.exp(-(d * d) / (2.0 * sigma_cells * sigma_cells)).mean(axis=2)      # (k_y, k_x)
            if j == ny:
                base = float(score[n, n])
            iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
            sc = float(score[iy, ix])
            if best is None or sc > best[0]:
                best = (sc, float(offs[ix]), float(offs[iy]), float(dy_yaw))
        if best is None or base is None:
            return None
        return (best[1], best[2], best[3], base, best[0])