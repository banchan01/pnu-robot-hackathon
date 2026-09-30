"""탐지 검사 리그: Supervisor로 로봇을 사과 앞 여러 거리에 놓고 카메라 프레임을 찍어 AppleDetector를 돌린다."""
import math, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tb3_mission"))
os.environ.setdefault("TB3_HEADLESS", "1")
from controller import Supervisor
import cv2
from config import load_mission
from perception import AppleDetector

OUT = "/tmp/tb3_captest"; os.makedirs(OUT, exist_ok=True)
robot = Supervisor(); ts = int(robot.getBasicTimeStep())
cam = robot.getDevice("camera"); cam.enable(ts)
node = robot.getSelf(); tf = node.getField("translation"); rf = node.getField("rotation")
for m in ("left wheel motor", "right wheel motor"):
    d = robot.getDevice(m); d.setPosition(float("inf")); d.setVelocity(0.0)
cfg = load_mission(); det = AppleDetector(cam, cfg)

# (이름, 사과 world 좌표, 접근 방향 각도(world, 사과에서 로봇 쪽))
cases = [("red2_bathroom", (-5.34, -10.54), [math.radians(a) for a in (60, 90, 30)]),
         ("red1_garden",   (-12.02, -3.02), [math.radians(a) for a in (0, 30, -30)])]
results = []
for name, (ax, ay), dirs in cases:
    for ang in dirs:
        for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            rx, ry = ax + d * math.cos(ang), ay + d * math.sin(ang)
            yaw = math.atan2(ay - ry, ax - rx)          # 사과를 향함
            tf.setSFVec3f([rx, ry, 0.0]); rf.setSFRotation([0, 0, 1, yaw]); node.resetPhysics()
            for _ in range(6):
                robot.step(ts)
            dets = det.detect((0.0, 0.0, 0.0), robot.getTime())
            reds = [x for x in dets if x["color"] == "red"]
            tag = f"{name}_a{math.degrees(ang):.0f}_d{d:.1f}"
            img = det.overlay()
            if img is not None:
                cv2.imwrite(os.path.join(OUT, tag + ".jpg"), img)
            r = {"case": tag, "true_d": d, "n_red": len(reds),
                 "est_d": round(reds[0]["dist"], 2) if reds else None,
                 "px": [round(v) for v in reds[0]["pixel"]] if reds else None,
                 "rejected_ground": det.rejected_ground}
            det.rejected_ground = 0
            results.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("CAPTEST_DONE", flush=True)
