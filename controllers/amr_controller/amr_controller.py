"""Autonomous Mission Controller for PNU AMR Search and Rescue.

Performs full mission loop:
  1. EXPLORE: Navigate arena with LiDAR obstacle avoidance while searching with camera.
  2. APPROACH: Visual servoing towards the detected green rescue target until arrival.
  3. RETURN: Navigate back to the starting point (0, 0) using wheel odometry & LiDAR avoidance.
  4. DONE: Safely stop at home location.
"""

import json
import math
import os
import cv2
import numpy as np

from controller import Keyboard, Robot
from a_star_planner import AStarPlanner
from slam_mapper import OccupancyGridMapper

global_planner = AStarPlanner()
mapper = OccupancyGridMapper()


MAX_WHEEL_SPEED = 6.67  # [rad/s] TurtleBot3 Burger limit
DRIVE_SPEED = 5.8
TURN_SPEED = 3.6
SLOW_DRIVE_SPEED = 3.2
SAFE_DISTANCE = 0.45  # [m]
CRITICAL_DISTANCE = 0.25  # [m]
LIDAR_PERIOD_MS = 100
STATUS_PERIOD_S = 0.5
WHEEL_RADIUS_M = 0.033
AXLE_LENGTH_M = 0.160

# Mission States
STATE_EXPLORE = "EXPLORE"
STATE_APPROACH = "APPROACH"
STATE_RETURN = "RETURN"
STATE_DONE = "DONE"


def finite_min(values):
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("inf")


def sector_min(ranges, center_index, half_width=15):
    size = len(ranges)
    indices = [(center_index + offset) % size for offset in range(-half_width, half_width + 1)]
    return finite_min([ranges[idx] for idx in indices])


def clamp(value, limit=MAX_WHEEL_SPEED):
    return max(-limit, min(limit, value))


def detect_target_in_camera(image_bytes, width, height):
    """Detect strictly RED APPLE targets.

    Ignores non-target distraction apples (Green, Orange, Purple) to avoid penalties.
    Rejects table dishes via height and size filter.
    Returns:
        (detected: bool, norm_x: float, pixel_count: int)
    """
    if not image_bytes:
        return False, 0.0, 0

    frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 4))
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)

    # Strictly Target: RED Apple (baseColor 1 0 0)
    # Red must dominate Green and Blue; Orange has high G (>150); Green has high G; Purple has high B
    mask_red = (r > 115) & (g < 125) & (b < 110) & (r > g + 40) & (r > b + 40)

    # OpenCV moments for centroid & position
    moments = cv2.moments(mask_red.astype(np.uint8))
    if moments["m00"] >= 15:
        count = int(moments["m00"])
        # Reject huge objects like table bowls (>3500px)
        if count <= 3500:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            # Reject objects high up on tables (top 20% of frame)
            if cy >= height * 0.20:
                norm_x = float((cx - (width / 2.0)) / (width / 2.0))
                return True, norm_x, count

    return False, 0.0, 0


# Initialize Webots robot devices
robot = Robot()
timestep = int(robot.getBasicTimeStep())

keyboard = Keyboard()
keyboard.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(timestep)
right_encoder.enable(timestep)

lidar = robot.getDevice("LDS-01")
lidar.enable(LIDAR_PERIOD_MS)

camera = robot.getDevice("camera")
camera.enable(timestep)

receiver = None
try:
    receiver = robot.getDevice("mission receiver")
    if receiver:
        receiver.enable(timestep)
except Exception:
    receiver = None

emitter = None
try:
    emitter = robot.getDevice("amr emitter")
    if emitter:
        emitter.setChannel(8)
except Exception:
    emitter = None

# Try loading trained RL PPO policy if available
RL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rl", "amr_ppo_model.zip")
rl_model = None
use_rl = False
try:
    from stable_baselines3 import PPO
    if os.path.exists(RL_MODEL_PATH):
        rl_model = PPO.load(RL_MODEL_PATH)
        use_rl = True
        print(f"[RL] Successfully loaded trained PPO model from {RL_MODEL_PATH}!")
except Exception as e:
    print(f"[RL] RL model not loaded ({e}). Using rule-based navigation.")


# Speed mapping the deployed policy was trained with (written by rl/train_mission_ppo.py --deploy)
RL_V_MIN, RL_V_MAX = 0.10, 0.22
try:
    with open(os.path.join(os.path.dirname(RL_MODEL_PATH), "policy_meta.json")) as _f:
        _meta = json.load(_f)
        RL_V_MIN = float(_meta.get("v_min", RL_V_MIN))
        RL_V_MAX = float(_meta.get("v_max", RL_V_MAX))
        print(f"[RL] policy speed mapping v in [{RL_V_MIN:.2f}, {RL_V_MAX:.2f}] (from policy_meta.json)")
except Exception:
    pass


def predict_rl_speeds(model, ranges, goal_dist, goal_bearing, start_dist):
    count = len(ranges)
    sector_ranges = np.empty(36, dtype=np.float32)
    for i in range(36):
        center_idx = (i * count) // 36
        sector_ranges[i] = min(3.5, sector_min(ranges, center_idx, half_width=5))

    norm_lidar = sector_ranges / 3.5
    norm_goal = min(1.0, goal_dist / 6.0)
    norm_start = min(1.0, start_dist / 6.0)

    obs = np.empty(40, dtype=np.float32)
    obs[:36] = norm_lidar
    obs[36] = norm_goal
    obs[37] = math.sin(goal_bearing)
    obs[38] = math.cos(goal_bearing)
    obs[39] = norm_start

    action, _ = model.predict(obs, deterministic=True)
    v = RL_V_MIN + (float(action[0]) + 1.0) / 2.0 * (RL_V_MAX - RL_V_MIN)  # mapping matches training env
    w = float(action[1]) * 2.0  # rad/s

    left_speed = (v - w * AXLE_LENGTH_M / 2.0) / WHEEL_RADIUS_M
    right_speed = (v + w * AXLE_LENGTH_M / 2.0) / WHEEL_RADIUS_M
    return clamp(left_speed), clamp(right_speed)

# Mission & Odometry Tracking (Arena World Frame)
if global_planner.is_apartment:
    start_x = -0.3
    start_y = -7.5
    theta = math.pi
    patrol_goals = [
        (-4.8, -7.5),    # Corridor junction
        (-5.34, -10.54), # South bedroom (🍎 Target 1: Red Apple)
        (-8.0, -5.5),    # Central hallway
        (-12.02, -3.02), # West kitchen (🍎 Target 2: Red Apple)
    ]
else:
    start_x = -2.0
    start_y = -1.5
    theta = 0.0
    patrol_goals = [(1.9, 1.8)]

patrol_idx = 0
target_x, target_y = patrol_goals[0]
x = start_x
y = start_y
waypoints = global_planner.plan((x, y), (target_x, target_y))
waypoint_idx = 1 if len(waypoints) > 1 else 0
previous_left = left_encoder.getValue()
previous_right = right_encoder.getValue()

state = STATE_EXPLORE
target_found = False
total_distance_traveled = 0.0
done_time = None
last_target_seen_time = -10.0
last_status_time = -STATUS_PERIOD_S
automatic = True  # Start in autonomous mode by default
previous_key = -1

# Multi-Target Mission Tracking (2 Red Apples in Apartment)
TARGETS_REQUIRED = 2 if global_planner.is_apartment else 1
KNOWN_RED_APPLES = [(-5.34, -10.54), (-12.02, -3.02)] if global_planner.is_apartment else [(1.9, 1.8)]
rescued_targets = 0
rescued_apples = []
step_counter = 0

# Create clean mission log file
with open("mission_log.txt", "w") as f:
    f.write("=== Mission Started ===\n")

print("=" * 60)
print(f"AMR A* + RL Controller Ready | Mode: {'RL POLICY' if use_rl else 'RULE-BASED BYPASS'}")
print("Controls: Space = Auto/Manual | E = Toggle RL/Rule | W/A/S/D = Teleop | Q = Stop")
print("=" * 60)

while robot.step(timestep) != -1:
    now = robot.getTime()

    # 1. Process Supervisor messages (episode reset)
    if receiver:
        while receiver.getQueueLength() > 0:
            data = receiver.getString() if hasattr(receiver, "getString") else receiver.getData()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            message = json.loads(data)
            receiver.nextPacket()
            if message.get("event") == "reset":
                if "start" in message:
                    start_x, start_y = message["start"][:2]
                if "target" in message:
                    target_x, target_y = message["target"][:2]
                x, y = start_x, start_y
                theta = message.get("start_yaw", 0.0)
                previous_left = left_encoder.getValue()
                previous_right = right_encoder.getValue()
                state = STATE_EXPLORE
                target_found = False
                total_distance_traveled = 0.0
                done_time = None

                # Plan A* global path around static walls
                waypoints = global_planner.plan((x, y), (target_x, target_y))
                waypoint_idx = 1 if len(waypoints) > 1 else 0
                msg = (
                    f"[MISSION] Reset! Start=({start_x:.2f}, {start_y:.2f}) "
                    f"Target=({target_x:.2f}, {target_y:.2f}) | A* Waypoints: {len(waypoints)}\n"
                )
                print(msg.strip())
                for i, wp in enumerate(waypoints):
                    print(f"  WP {i+1}: ({wp[0]:.2f}, {wp[1]:.2f})")
                with open("mission_log.txt", "a") as f:
                    f.write(msg)

    # 2. Update Odometry from Wheel Encoders
    current_left = left_encoder.getValue()
    current_right = right_encoder.getValue()
    dl = (current_left - previous_left) * WHEEL_RADIUS_M
    dr = (current_right - previous_right) * WHEEL_RADIUS_M
    distance = (dl + dr) / 2.0
    total_distance_traveled += abs(distance)
    theta += (dr - dl) / AXLE_LENGTH_M
    theta = math.atan2(math.sin(theta), math.cos(theta))
    x += distance * math.cos(theta)
    y += distance * math.sin(theta)
    previous_left, previous_right = current_left, current_right

    start_distance = math.hypot(start_x - x, start_y - y)
    start_bearing = math.atan2(start_y - y, start_x - x) - theta
    start_bearing = math.atan2(math.sin(start_bearing), math.cos(start_bearing))

    # Distance & bearing to target by odometry
    target_dist_est = math.hypot(target_x - x, target_y - y) if target_x is not None else 999.0
    target_bearing_est = math.atan2(target_y - y, target_x - x) - theta if target_x is not None else 0.0
    target_bearing_est = math.atan2(math.sin(target_bearing_est), math.cos(target_bearing_est))

    # 3. Read LiDAR Data (360 points)
    ranges = lidar.getRangeImage()
    count = len(ranges)
    front = sector_min(ranges, count // 2, half_width=15)
    left = sector_min(ranges, count // 4, half_width=15)
    right = sector_min(ranges, 3 * count // 4, half_width=15)
    back = sector_min(ranges, 0, half_width=15)

    # Real-time SLAM Occupancy Grid Map update
    step_counter += 1
    if step_counter % 2 == 0:
        mapper.update(x, y, theta, ranges)
    if step_counter % 120 == 0:
        mapper.export_map_image("slam_map.png")

    # 4. Camera Target Detection (Strictly Red Apples only)
    cam_bytes = camera.getImage()
    target_visible, target_norm_x, target_pixels = detect_target_in_camera(
        cam_bytes, camera.getWidth(), camera.getHeight()
    )

    if target_visible:
        last_target_seen_time = now

    # 5. Mission State Machine Transitions
    # Check if robot is close to ANY unrescued Red Apple (by camera or odometry proximity)
    can_rescue = False
    rescued_pos = None
    for apple_pos in KNOWN_RED_APPLES:
        apple_dist = math.hypot(x - apple_pos[0], y - apple_pos[1])
        is_already = any(math.hypot(apple_pos[0] - rx, apple_pos[1] - ry) < 1.0 for rx, ry in rescued_apples)
        if not is_already:
            # Proximity rescue: robot approached within 0.45m of apple location
            # OR camera visual rescue: apple spotted and front distance < 0.38m
            if apple_dist < 0.45 or (target_visible and front < 0.38 and target_pixels >= 15):
                can_rescue = True
                rescued_pos = apple_pos
                break

    if can_rescue and state != STATE_RETURN and state != STATE_DONE:
        rescued_targets += 1
        rescued_apples.append(rescued_pos)
        mapper.mark_rescue(rescued_pos[0], rescued_pos[1])
        mapper.export_map_image("slam_map.png")
        msg = f"[MISSION] >>> RED APPLE {rescued_targets}/{TARGETS_REQUIRED} RESCUED at ({rescued_pos[0]:.2f}, {rescued_pos[1]:.2f})!\n"
        print("=" * 60)
        print(msg.strip())
        print("=" * 60)
        with open("mission_log.txt", "a") as f:
            f.write(msg)

        if rescued_targets < TARGETS_REQUIRED:
            # Head directly to the 2nd Red Apple!
            state = STATE_EXPLORE
            remaining = [a for a in KNOWN_RED_APPLES if not any(math.hypot(a[0] - rx, a[1] - ry) < 1.0 for rx, ry in rescued_apples)]
            target_x, target_y = remaining[0] if remaining else patrol_goals[0]
            waypoints = global_planner.plan((x, y), (target_x, target_y))
            waypoint_idx = 1 if len(waypoints) > 1 else 0
            print(f"[MISSION] Searching for 2nd Red Apple... Heading to ({target_x:.2f}, {target_y:.2f}) ({len(waypoints)} WPs)")
        else:
            # Both Red Apples rescued! Time to return home!
            state = STATE_RETURN
            waypoints = global_planner.plan((x, y), (start_x, start_y))
            waypoint_idx = 1 if len(waypoints) > 1 else 0
            return_msg = f"[MISSION] >>> ALL {TARGETS_REQUIRED} RED APPLES RESCUED! Returning to Start Pad ({len(waypoints)} WPs)...\n"
            print("*" * 60)
            print(return_msg.strip())
            print("*" * 60)
            with open("mission_log.txt", "a") as f:
                f.write(return_msg)

    elif state == STATE_EXPLORE:
        is_already_rescued = any(math.hypot(x - rx, y - ry) < 1.0 for rx, ry in rescued_apples)
        if target_visible and target_pixels >= 15 and not is_already_rescued:
            state = STATE_APPROACH
            msg = f"[MISSION] >>> RED APPLE spotted ({target_pixels}px)! Approaching target...\n"
            print(msg.strip())
            with open("mission_log.txt", "a") as f:
                f.write(msg)

    elif state == STATE_APPROACH:
        if not target_visible and (now - last_target_seen_time > 1.5):
            # Target was blocked or lost, resume exploration smoothly
            state = STATE_EXPLORE
            waypoints = global_planner.plan((x, y), (target_x, target_y))
            waypoint_idx = 1 if len(waypoints) > 1 else 0
            print("[MISSION] Target lost or unreachable. Resuming patrol...")

    elif state == STATE_RETURN:
        # Arrival home check: must have actually traveled out and returned
        if start_distance < 0.35 and total_distance_traveled > 1.0:
            state = STATE_DONE
            mapper.export_map_image("final_slam_map.png")
            msg = f"[MISSION] >>> MISSION ACCOMPLISHED! All {rescued_targets} Red Apples Rescued & Safely Returned to Start!\n"
            print("*" * 60)
            print(msg.strip())
            print("[SLAM] Final apartment occupancy grid map saved to controllers/amr_controller/final_slam_map.png!")
            print("*" * 60)
            with open("mission_log.txt", "a") as f:
                f.write(msg)

    # 6. Smooth Navigation (A* Waypoint Tracking + RL Local Avoidance)
    left_speed = 0.0
    right_speed = 0.0

    key = keyboard.getKey()
    if key == ord(" ") and previous_key != ord(" "):
        automatic = not automatic
        print(f"[MODE] Switched to {'AUTONOMOUS' if automatic else 'MANUAL TELEOP'}")
    if key in (ord("E"), ord("e")) and previous_key not in (ord("E"), ord("e")):
        if rl_model is not None:
            use_rl = not use_rl
            print(f"[MODE] Navigation controller: {'RL POLICY' if use_rl else 'RULE-BASED BYPASS'}")
        else:
            print("[MODE] RL model not available.")

    # Dynamic A* Waypoint Tracking
    if waypoints and waypoint_idx < len(waypoints):
        cur_wp = waypoints[waypoint_idx]
        cur_wp_dist = math.hypot(cur_wp[0] - x, cur_wp[1] - y)
        if cur_wp_dist < 0.40 and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            cur_wp = waypoints[waypoint_idx]
            cur_wp_dist = math.hypot(cur_wp[0] - x, cur_wp[1] - y)
            print(f"[A* NAV] Advanced to Waypoint {waypoint_idx+1}/{len(waypoints)}: ({cur_wp[0]:.2f}, {cur_wp[1]:.2f})")
        elif cur_wp_dist < 0.40 and waypoint_idx >= len(waypoints) - 1 and state == STATE_EXPLORE and len(patrol_goals) > 1:
            patrol_idx = (patrol_idx + 1) % len(patrol_goals)
            target_x, target_y = patrol_goals[patrol_idx]
            waypoints = global_planner.plan((x, y), (target_x, target_y))
            waypoint_idx = 1 if len(waypoints) > 1 else 0
            print(f"[PATROL] Reached room checkpoint! Proceeding to room {patrol_idx+1}/{len(patrol_goals)}: ({target_x:.2f}, {target_y:.2f}) ({len(waypoints)} WPs)")

        subgoal_dist = cur_wp_dist
        subgoal_bearing = math.atan2(cur_wp[1] - y, cur_wp[0] - x) - theta
        subgoal_bearing = math.atan2(math.sin(subgoal_bearing), math.cos(subgoal_bearing))
    else:
        subgoal_dist = target_dist_est if state != STATE_RETURN else start_distance
        subgoal_bearing = target_bearing_est if state != STATE_RETURN else start_bearing

    if automatic:
        if state == STATE_EXPLORE:
            if use_rl and rl_model is not None:
                left_speed, right_speed = predict_rl_speeds(
                    rl_model, ranges, subgoal_dist, subgoal_bearing, start_distance
                )
            else:
                target_heading = subgoal_bearing
                if front < 0.25:
                    turn_sign = 1.0 if left > right else -1.0
                    left_speed = -1.2 - 2.5 * turn_sign
                    right_speed = -1.2 + 2.5 * turn_sign
                elif front < 0.45:
                    turn_sign = 1.0 if left > right else -1.0
                    left_speed = 1.0 - 2.8 * turn_sign
                    right_speed = 1.0 + 2.8 * turn_sign
                else:
                    p_steer = clamp(target_heading * 2.2, 2.5)
                    fwd = 3.0 * max(0.2, math.cos(target_heading))
                    left_speed = fwd - p_steer
                    right_speed = fwd + p_steer

        elif state == STATE_APPROACH:
            # Home in on target (visual or bearing)
            guide_angle = (target_norm_x * 0.6) if target_visible else target_bearing_est

            if front < 0.22 and not (target_visible and target_pixels > 60):
                turn_sign = 1.0 if left > right else -1.0
                left_speed = -1.0 - 2.0 * turn_sign
                right_speed = -1.0 + 2.0 * turn_sign
            elif front < 0.35 and not target_visible:
                turn_sign = 1.0 if left > right else -1.0
                left_speed = 1.0 - 2.5 * turn_sign
                right_speed = 1.0 + 2.5 * turn_sign
            else:
                p_steer = clamp(guide_angle * 2.5, 2.5)
                fwd = 2.4 * max(0.3, math.cos(guide_angle))
                left_speed = fwd - p_steer
                right_speed = fwd + p_steer

        elif state == STATE_RETURN:
            if use_rl and rl_model is not None:
                left_speed, right_speed = predict_rl_speeds(
                    rl_model, ranges, subgoal_dist, subgoal_bearing, target_dist_est
                )
            else:
                home_heading = subgoal_bearing
                if front < 0.25:
                    turn_sign = 1.0 if left > right else -1.0
                    left_speed = -1.2 - 2.5 * turn_sign
                    right_speed = -1.2 + 2.5 * turn_sign
                elif front < 0.45:
                    turn_sign = 1.0 if left > right else -1.0
                    left_speed = 1.0 - 2.8 * turn_sign
                    right_speed = 1.0 + 2.8 * turn_sign
                else:
                    p_steer = clamp(home_heading * 2.4, 2.5)
                    fwd = 3.0 * max(0.2, math.cos(home_heading))
                    left_speed = fwd - p_steer
                    right_speed = fwd + p_steer

        elif state == STATE_DONE:
            left_speed = 0.0
            right_speed = 0.0

        # Collision safety check and auto-reset
        min_lidar_val = finite_min(ranges)
        if min_lidar_val < 0.13 and state != STATE_DONE:
            if emitter:
                print(f"[COLLISION] Hit obstacle ({min_lidar_val:.2f}m). Requesting supervisor reset...")
                emitter.send(json.dumps({"event": "request_reset", "reason": "collision"}).encode())

    else:
        # Manual Teleop
        if key in (ord("W"), ord("w")) and front >= SAFE_DISTANCE:
            left_speed = DRIVE_SPEED
            right_speed = DRIVE_SPEED
        elif key in (ord("S"), ord("s")) and back >= SAFE_DISTANCE:
            left_speed = -DRIVE_SPEED
            right_speed = -DRIVE_SPEED
        elif key in (ord("A"), ord("a")):
            left_speed = -TURN_SPEED
            right_speed = TURN_SPEED
        elif key in (ord("D"), ord("d")):
            left_speed = TURN_SPEED
            right_speed = -TURN_SPEED
        elif key in (ord("Q"), ord("q")):
            automatic = False

    left_motor.setVelocity(clamp(left_speed))
    right_motor.setVelocity(clamp(right_speed))

    # 7. Status Logging
    if now - last_status_time >= STATUS_PERIOD_S:
        status_line = (
            f"[{state}] "
            f"Pos:({x:+.2f}, {y:+.2f}) "
            f"HomeDist:{start_distance:.2f}m "
            f"Front:{front:.2f}m "
            f"Cam:({target_pixels}px, vis={target_visible})"
        )
        print(status_line)
        with open("mission_log.txt", "a") as f:
            f.write(status_line + "\n")
        last_status_time = now

    previous_key = key

