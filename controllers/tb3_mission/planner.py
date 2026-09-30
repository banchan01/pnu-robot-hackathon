"""전역 경로 계획 (담당 B).

A* (8방향, Costmap 비용 반영) → 시선 검사로 경로 단축 → 0.2 m 간격 waypoint.
노트북의 A* 구현을 격자 배열과 Costmap 비용에 맞게 확장한 것이다.
"""
import heapq
import math

import numpy as np

from mapping import LETHAL, INSCRIBED, UNKNOWN

_NEIGHBORS = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
              (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
              (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]

UNKNOWN_STEP_COST = 3.0


def _octile(a, b):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2.0) * min(dr, dc)


def astar(costmap, start, goal, allow_unknown=True, start_free_radius=2):
    """격자 셀 경로 [(r, c), ...] 또는 None."""
    rows, cols = costmap.shape
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None
    if costmap[goal] >= INSCRIBED and costmap[goal] != UNKNOWN:
        return None
    if costmap[goal] == UNKNOWN and not allow_unknown:
        return None

    g = np.full((rows, cols), np.inf, dtype=np.float32)
    came = {}
    closed = np.zeros((rows, cols), dtype=bool)
    g[start] = 0.0
    heap = [(_octile(start, goal), start)]

    sr, sc = start
    while heap:
        f, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        if closed[cur]:
            continue
        closed[cur] = True
        cr, cc = cur
        for dr, dc, step in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if closed[nr, nc]:
                continue
            v = costmap[nr, nc]
            near_start = abs(nr - sr) <= start_free_radius and abs(nc - sc) <= start_free_radius
            if v == LETHAL:
                continue
            if v == INSCRIBED and not near_start:
                continue
            if v == UNKNOWN:
                if not allow_unknown:
                    continue
                cost = step * UNKNOWN_STEP_COST
            elif v == INSCRIBED:
                cost = step * 6.0
            else:
                cost = step * (1.0 + v / 64.0)
            # 대각선 이동 시 모서리 끼임 방지
            if dr != 0 and dc != 0:
                if costmap[cr + dr, cc] >= INSCRIBED and costmap[cr + dr, cc] != UNKNOWN:
                    continue
                if costmap[cr, cc + dc] >= INSCRIBED and costmap[cr, cc + dc] != UNKNOWN:
                    continue
            ng = g[cur] + cost
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                came[(nr, nc)] = cur
                heapq.heappush(heap, (ng + _octile((nr, nc), goal), (nr, nc)))
    return None


def _line_cells(a, b):
    """Bresenham. a, b는 (r, c)."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dc - dr
    r, c = r0, c0
    out = []
    while True:
        out.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return out


def _line_clear(costmap, a, b, max_cost=200):
    for (r, c) in _line_cells(a, b):
        v = costmap[r, c]
        if v == UNKNOWN:
            continue
        if v > max_cost:
            return False
    return True


def simplify(path, costmap, max_cost=200):
    """시선이 통하는 구간의 중간점을 제거한다."""
    if path is None or len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _line_clear(costmap, path[i], path[j], max_cost):
            j -= 1
        out.append(path[j])
        i = j
    return out


def to_world_path(path_rc, grid, spacing_m=0.2):
    """셀 경로를 미터 좌표로 바꾸고 일정 간격으로 다시 샘플링한다."""
    if not path_rc:
        return []
    pts = [grid.grid_to_world(r, c) for (r, c) in path_rc]
    out = [pts[0]]
    for k in range(1, len(pts)):
        x0, y0 = out[-1]
        x1, y1 = pts[k]
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-6:
            continue
        n = max(1, int(d / spacing_m))
        for s in range(1, n + 1):
            t = s / float(n)
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def path_blocked(path_world, grid, costmap, from_idx=0, pose=None, horizon_m=1.5, ignore_radius=0.30):
    """앞으로 horizon_m 구간의 경로가 새 장애물에 막혔는지.
    로봇 반경 ignore_radius 안의 셀은 팽창 구역일 수 있으므로 판정에서 제외한다."""
    if not path_world:
        return False
    acc = 0.0
    prev = None
    inscribed_run = 0
    for k in range(max(0, from_idx), len(path_world)):
        x, y = path_world[k]
        if pose is not None and math.hypot(x - pose[0], y - pose[1]) < ignore_radius:
            prev = (x, y)
            continue
        r, c = grid.world_to_grid(x, y)
        if grid.in_bounds(r, c):
            v = costmap[r, c]
            if v == LETHAL:
                return True
            if v == INSCRIBED:
                inscribed_run += 1
                if inscribed_run >= 3:
                    return True
            else:
                inscribed_run = 0
        if prev is not None:
            acc += math.hypot(x - prev[0], y - prev[1])
            if acc > horizon_m:
                break
        prev = (x, y)
    return False


def plan_world(grid, costmap, start_xy, goal_xy, allow_unknown=True, clear_cells=5):
    """미터 좌표 시작/목표로 계획하고 미터 waypoint 목록을 돌려준다. 실패 시 None."""
    s = grid.world_to_grid(*start_xy)
    g = grid.world_to_grid(*goal_xy)
    g2 = grid.snap_to_traversable(costmap, g[0], g[1], radius=8)
    if g2 is None:
        return None
    if not grid.in_bounds(*s):
        return None
    # 로봇이 실제로 서 있는 자리는 통행 가능해야 한다: 팽창 구역이나 가상 장애물이
    # 출발 셀을 둘러싸 A*가 시작조차 못 하는 상황을 막기 위해 반경 clear_cells를 비운다
    cm = costmap.copy()
    r0, c0 = s
    cc = clear_cells
    r_lo, r_hi = max(0, r0 - cc), min(cm.shape[0], r0 + cc + 1)
    c_lo, c_hi = max(0, c0 - cc), min(cm.shape[1], c0 + cc + 1)
    block = cm[r_lo:r_hi, c_lo:c_hi]
    block[(block == INSCRIBED) | (block == LETHAL)] = 200
    cells = astar(cm, s, g2, allow_unknown=allow_unknown)
    if cells is None:
        return None
    cells = simplify(cells, cm)
    return to_world_path(cells, grid)
