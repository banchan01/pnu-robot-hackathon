"""Webots Supervisor for repeatable search-and-rescue episodes with domain randomization.

Press R while the supervisor has keyboard focus to randomize obstacles, target, and robot.
Also listens on channel 8 for automatic episode reset requests from the AMR controller.
"""

import json
import math
import random

from controller import Keyboard, Supervisor

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
keyboard = Keyboard()
keyboard.enable(timestep)

robot = supervisor.getFromDef("AMR")
target = supervisor.getFromDef("RESCUE_TARGET")

obstacle_defs = [
    "RANDOM_OBSTACLE",
    "RANDOM_OBSTACLE_2",
    "RANDOM_OBSTACLE_3",
    "RANDOM_OBSTACLE_4",
    "RANDOM_OBSTACLE_5",
]
obstacles = [supervisor.getFromDef(name) for name in obstacle_defs]
obstacles = [obs for obs in obstacles if obs is not None]

emitter = supervisor.getDevice("mission emitter")
emitter.setChannel(7)

receiver = supervisor.getDevice("supervisor receiver")
if receiver:
    receiver.enable(timestep)

start_center = (-2.0, -1.5, 0.0)
rng = random.Random()


def is_in_fixed_obstacle(x, y, margin=0.35):
    # CENTER_WALL: (0.3, 0), size 0.2 x 2.2
    if (-0.1 - margin <= x <= 0.7 + margin) and (-1.1 - margin <= y <= 1.1 + margin):
        return True
    # EXTRA_WALL: (-0.8, -0.8), size 1.4 x 0.2
    if (-1.5 - margin <= x <= -0.1 + margin) and (-0.9 - margin <= y <= -0.7 + margin):
        return True
    # PILLAR_1: (1.2, -1.2), radius 0.25
    if math.hypot(x - 1.2, y - (-1.2)) < (0.25 + margin):
        return True
    return False


def randomize_episode():
    """Place the robot, multiple obstacles and rescue target in a safe, diverse layout."""

    # 1. Randomize robot near start
    start_x = start_center[0] + rng.uniform(-0.15, 0.15)
    start_y = start_center[1] + rng.uniform(-0.15, 0.15)
    start_pos = (start_x, start_y, 0.0)
    yaw = rng.uniform(-math.pi / 4, math.pi / 4)

    robot.getField("translation").setSFVec3f(list(start_pos))
    robot.getField("rotation").setSFRotation([0, 0, 1, yaw])
    robot.resetPhysics()

    # 2. Pick target position with good distance from start
    target_pos = None
    for _ in range(100):
        tx = rng.uniform(-1.0, 2.3)
        ty = rng.uniform(-2.0, 2.3)
        if math.hypot(tx - start_x, ty - start_y) >= 2.2 and not is_in_fixed_obstacle(tx, ty, margin=0.4):
            target_pos = (tx, ty, 0.16)
            break
    if target_pos is None:
        target_pos = (1.9, 1.8, 0.16)

    target.getField("translation").setSFVec3f(list(target_pos))

    # 3. Randomize each obstacle
    placed_positions = [(start_x, start_y), (target_pos[0], target_pos[1])]
    for obs in obstacles:
        placed = False
        for _ in range(60):
            ox = rng.uniform(-2.3, 2.3)
            oy = rng.uniform(-2.3, 2.3)
            if is_in_fixed_obstacle(ox, oy, margin=0.45):
                continue
            if math.hypot(ox - start_x, oy - start_y) < 0.85:
                continue
            if math.hypot(ox - target_pos[0], oy - target_pos[1]) < 0.65:
                continue
            if any(math.hypot(ox - px, oy - py) < 0.65 for px, py in placed_positions):
                continue

            obs.getField("translation").setSFVec3f([ox, oy, 0.25])
            placed_positions.append((ox, oy))
            placed = True
            break

        if not placed:
            obs.getField("translation").setSFVec3f([0.0, 0.0, -10.0])

    msg = {
        "event": "reset",
        "target": target_pos,
        "start": start_pos,
        "start_yaw": yaw,
    }
    emitter.send(json.dumps(msg).encode())
    print(
        f"[SUPERVISOR] Reset episode: robot=({start_x:.2f}, {start_y:.2f}) "
        f"target=({target_pos[0]:.2f}, {target_pos[1]:.2f}) "
        f"obstacles={len(placed_positions)-2}"
    )


randomize_episode()
while supervisor.step(timestep) != -1:
    if receiver:
        while receiver.getQueueLength() > 0:
            req = json.loads(receiver.getData().decode())
            receiver.nextPacket()
            if req.get("event") == "request_reset":
                randomize_episode()
                break

    if keyboard.getKey() in (ord("r"), ord("R")):
        randomize_episode()
