"""Small, dependency-free RL interface for the Webots controller.

This file deliberately contains no Webots calls.  It turns the values already
available in the controller into a fixed-size observation and computes a simple
mission reward.  A future PPO/DQN runner can call these functions without
changing the sensor code.
"""

from dataclasses import dataclass
import math


@dataclass
class MissionState:
    """Information that the mission controller knows at one time step."""

    lidar: list[float]
    goal_distance: float
    goal_bearing: float
    start_distance: float
    target_found: bool = False
    at_start: bool = False
    collision: bool = False
    done: bool = False


def _clean(value: float, maximum: float = 10.0) -> float:
    if not math.isfinite(value):
        return maximum
    return max(0.0, min(maximum, value))


def observation(state: MissionState, sectors: int = 36) -> list[float]:
    """Return a fixed-size observation vector for a neural policy.

    LiDAR is downsampled into equally sized sectors, followed by normalized
    goal/start distances and goal bearing.  No GPS or compass is used.
    """

    if not state.lidar:
        scan = [10.0] * sectors
    else:
        scan = []
        for i in range(sectors):
            begin = i * len(state.lidar) // sectors
            end = max(begin + 1, (i + 1) * len(state.lidar) // sectors)
            scan.append(min(_clean(v) for v in state.lidar[begin:end]))
    return [v / 10.0 for v in scan] + [
        _clean(state.goal_distance) / 10.0,
        max(-1.0, min(1.0, state.goal_bearing / math.pi)),
        _clean(state.start_distance) / 10.0,
        float(state.target_found),
    ]


def reward(previous: MissionState, current: MissionState) -> float:
    """Simple dense reward: progress is good, collision/time wasting is bad."""

    if current.collision:
        return -100.0
    if current.at_start and current.target_found:
        return 200.0
    if current.target_found and not previous.target_found:
        return 50.0
    progress = previous.goal_distance - current.goal_distance
    if current.target_found:
        progress = previous.start_distance - current.start_distance
    return max(-1.0, min(1.0, progress)) - 0.01
