"""강화학습(PPO) 정책 섀도 평가.

팀이 rl/ 에서 학습한 PPO 정책(`rl/amr_win1_policy.pt`, Stable-Baselines3 MlpPolicy 가중치)을
미션 컨트롤러 안에서 **같은 관측으로 매 tick 추론만** 하고, 실제 바퀴 명령은 규칙 기반 Local Planner가 낸다.
정책이 제안한 (v, w)와 실제 명령의 일치도를 기록해 summary_final.json에 남긴다.

- 관측(40차원)은 rl/apartment_env.py 의 _get_obs 와 같은 형식:
    [36개 LiDAR 구간 최소 거리 / 3.5,  목표 거리/6,  sin(목표 방위),  cos(목표 방위),  출발점 거리/6]
- 행동: a ∈ [-1, 1]^2  →  v = v_min + (a0+1)/2·(v_max−v_min),  w = a1·2.0 rad/s  (rl/policy_meta.json)
- torch 가 없거나 가중치 파일이 없으면 조용히 비활성화된다. 미션 동작에는 영향이 없다.
"""
import json
import math
import os

import numpy as np

from config import CTRL_DIR, LIDAR_ANGLES, LIDAR_MIN

RL_DIR = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "rl"))
LIDAR_SECTORS = 36
LIDAR_MAX_RANGE = 3.5
MAX_ANGULAR_SPEED = 2.0


class ShadowPolicy:
    def __init__(self, cfg, log):
        self.enabled = False
        self.log = log
        self.v_min, self.v_max = 0.10, 0.22
        self.n = 0
        self.w_sign_agree = 0
        self.w_cmp = 0
        self.abs_dv = 0.0
        self.abs_dw = 0.0
        self.last = None
        weight = os.path.join(RL_DIR, str(cfg.get("rl_policy_file", "amr_win1_policy.pt")))
        if not os.path.exists(weight):                       # 브랜치에 따라 파일 이름이 다를 수 있음
            for alt in ("amr_win1_policy.pt", "amr_apartment_policy.pt", "amr_policy.pt"):
                if os.path.exists(os.path.join(RL_DIR, alt)):
                    weight = os.path.join(RL_DIR, alt)
                    break
        meta = os.path.join(RL_DIR, "policy_meta.json")
        try:
            import torch
            self.torch = torch
            sd = torch.load(weight, map_location="cpu", weights_only=False)
            self.W = [(sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]),
                      (sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]),
                      (sd["action_net.weight"], sd["action_net.bias"])]
            if os.path.exists(meta):
                with open(meta) as f:
                    m = json.load(f)
                self.v_min = float(m.get("v_min", self.v_min))
                self.v_max = float(m.get("v_max", self.v_max))
                self.meta = m
            else:
                self.meta = {}
            self.enabled = True
            self.weight_name = os.path.basename(weight)
            self.log(f"[RL] 섀도 정책 로드: {os.path.basename(weight)} "
                     f"({self.meta.get('timesteps', '?')} steps, v∈[{self.v_min:.2f},{self.v_max:.2f}])")
        except Exception as e:      # torch 미설치, 파일 없음 등
            self.log(f"[RL] 섀도 정책 비활성화 ({e.__class__.__name__}: {e})")

    # ---------- 관측 ----------
    def observation(self, ranges, pose, goal, home=(0.0, 0.0)):
        r = np.asarray(ranges, dtype=np.float32)
        r = np.where(np.isfinite(r) & (r >= LIDAR_MIN), r, LIDAR_MAX_RANGE)
        # 전방(인덱스 180)부터 반시계 방향으로 10도씩 36구간 최소 거리
        rel = (LIDAR_ANGLES + 2 * math.pi) % (2 * math.pi)          # 0 = 전방, 반시계 양수
        sector = np.minimum((rel / (2 * math.pi / LIDAR_SECTORS)).astype(int), LIDAR_SECTORS - 1)
        sec_min = np.full(LIDAR_SECTORS, LIDAR_MAX_RANGE, dtype=np.float32)
        np.minimum.at(sec_min, sector, r)
        x, y, yaw = pose
        gx, gy = goal if goal is not None else home
        dist = math.hypot(gx - x, gy - y)
        bearing = math.atan2(gy - y, gx - x) - yaw
        obs = np.empty(LIDAR_SECTORS + 4, dtype=np.float32)
        obs[:LIDAR_SECTORS] = np.clip(sec_min / LIDAR_MAX_RANGE, 0.0, 1.0)
        obs[LIDAR_SECTORS] = min(1.0, dist / 6.0)
        obs[LIDAR_SECTORS + 1] = math.sin(bearing)
        obs[LIDAR_SECTORS + 2] = math.cos(bearing)
        obs[LIDAR_SECTORS + 3] = min(1.0, math.hypot(home[0] - x, home[1] - y) / 6.0)
        return obs

    # ---------- 추론 ----------
    def act(self, obs):
        t = self.torch
        with t.no_grad():
            h = t.as_tensor(obs)
            for i, (W, b) in enumerate(self.W):
                h = h @ W.T + b
                if i < 2:
                    h = t.tanh(h)                     # SB3 MlpPolicy 기본 활성화
            a = t.clamp(h, -1.0, 1.0).numpy()
        v = self.v_min + (float(a[0]) + 1.0) / 2.0 * (self.v_max - self.v_min)
        w = float(a[1]) * MAX_ANGULAR_SPEED
        return v, w

    # ---------- 섀도 비교 ----------
    def step(self, ranges, pose, goal, cmd_v, cmd_w):
        if not self.enabled:
            return None
        try:
            v, w = self.act(self.observation(ranges, pose, goal))
        except Exception as e:
            self.log(f"[RL] 추론 오류로 섀도 비활성화 ({e})")
            self.enabled = False
            return None
        self.last = (v, w)
        if abs(cmd_v) > 0.02 or abs(cmd_w) > 0.2:      # 로봇이 실제로 움직일 때만 비교
            self.n += 1
            self.abs_dv += abs(v - max(cmd_v, 0.0))
            self.abs_dw += abs(w - cmd_w)
            if abs(cmd_w) > 0.2:
                self.w_cmp += 1
                if (w > 0) == (cmd_w > 0):
                    self.w_sign_agree += 1
        return (v, w)

    def summary(self):
        if not self.enabled or self.n == 0:
            return {"enabled": self.enabled, "samples": self.n}
        return {
            "enabled": True,
            "policy": getattr(self, "weight_name", "?"),
            "timesteps": self.meta.get("timesteps"),
            "samples": self.n,
            "turn_sign_agreement": round(self.w_sign_agree / max(1, self.w_cmp), 3),
            "mean_abs_dv": round(self.abs_dv / self.n, 3),
            "mean_abs_dw": round(self.abs_dw / self.n, 3),
        }
