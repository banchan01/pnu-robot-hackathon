"""Minimal Webots controller for the PNU AMR hackathon.

This first version intentionally uses no compass and no GPS/GNSS. It verifies
the allowed sensor/control path before mapping and planning are added.
"""

import math

from controller import Keyboard, Robot


MAX_WHEEL_SPEED = 6.67  # TurtleBot3 Burger motor limit [rad/s]
DRIVE_SPEED = 3.0
TURN_SPEED = 2.0
SAFE_DISTANCE = 0.35  # [m]
LIDAR_PERIOD_MS = 100
STATUS_PERIOD_S = 0.5


def finite_min(values):
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else float("inf")


def sector_min(ranges, center_index, half_width=12):
    size = len(ranges)
    indices = [(center_index + offset) % size for offset in range(-half_width, half_width + 1)]
    return finite_min([ranges[index] for index in indices])


def clamp(value, limit=MAX_WHEEL_SPEED):
    return max(-limit, min(limit, value))


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

automatic = False
last_status_time = -STATUS_PERIOD_S
previous_key = -1

print("AMR controller ready")
print("W/A/S/D: drive | Space: toggle safe wander | Q: stop")

while robot.step(timestep) != -1:
    ranges = lidar.getRangeImage()
    count = len(ranges)

    # LDS-01 ordering in the provided Webots setup: back=0, left=1/4,
    # front=1/2, right=3/4 of the scan array.
    back = sector_min(ranges, 0)
    left = sector_min(ranges, count // 4)
    front = sector_min(ranges, count // 2)
    right = sector_min(ranges, 3 * count // 4)

    key = keyboard.getKey()
    if key == ord(" ") and previous_key != ord(" "):
        automatic = not automatic
        print(f"mode={'SAFE_WANDER' if automatic else 'TELEOP'}")

    left_speed = 0.0
    right_speed = 0.0

    if automatic:
        if front < SAFE_DISTANCE:
            # Rotate toward the side with more free space.
            turn_direction = 1.0 if left > right else -1.0
            left_speed = -TURN_SPEED * turn_direction
            right_speed = TURN_SPEED * turn_direction
        else:
            left_speed = DRIVE_SPEED
            right_speed = DRIVE_SPEED
    else:
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

    now = robot.getTime()
    if now - last_status_time >= STATUS_PERIOD_S:
        print(
            f"mode={'AUTO' if automatic else 'MANUAL'} "
            f"lidar_points={count} "
            f"front={front:.2f} left={left:.2f} right={right:.2f} back={back:.2f} "
            f"encoder=({left_encoder.getValue():.3f}, {right_encoder.getValue():.3f}) "
            f"camera={camera.getWidth()}x{camera.getHeight()}"
        )
        last_status_time = now

    previous_key = key
