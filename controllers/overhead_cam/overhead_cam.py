"""로봇 상공 추적 카메라 (웹 지휘 콘솔의 '상공 시점').

물리가 없는 Supervisor Robot 노드에 아래를 향한 카메라를 달고, 매 step마다 TurtleBot3 위 일정 높이로
자기 위치를 옮겨 드론처럼 따라간다. 화면 위쪽이 로봇 진행 방향이 되도록 yaw 도 맞춘다.
프레임은 controllers/tb3_mission/output/live_top.jpg 로 저장되고 web/server.py 가 그대로 보여준다.
이 노드가 월드에 없어도 미션 컨트롤러와 웹 콘솔은 정상 동작한다 (상공 시점만 비활성).
"""
import math
import os

import cv2
import numpy as np
from controller import Supervisor

HEIGHT = 2.2                 # 로봇 위 카메라 높이 [m] (천장 2.4 m 아래)
SAVE_EVERY = 2               # step 몇 번마다 저장할지 (64 ms 기준 약 8 fps)
CTRL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tb3_mission")
LIVE_POINTER = os.path.join(CTRL_DIR, "live_output_dir.txt")


def output_path():
    """tb3_mission 이 실제로 쓰는 출력 폴더 (환경 변수로 바뀔 수 있음) 를 포인터 파일에서 읽는다."""
    d = os.environ.get("TB3_OUTPUT_DIR")
    if not d:
        try:
            with open(LIVE_POINTER, "r", encoding="utf-8") as f:
                d = f.read().strip()
        except OSError:
            d = ""
    if not d:
        d = os.path.join(CTRL_DIR, "output")
    return os.path.join(d, "live_top.jpg")

sup = Supervisor()
ts = int(sup.getBasicTimeStep())
cam = sup.getDevice("top_camera")
cam.enable(ts)
W, H = cam.getWidth(), cam.getHeight()

me = sup.getSelf()
f_tr = me.getField("translation")
f_rot = me.getField("rotation")


def find_robot():
    root = sup.getRoot().getField("children")
    for i in range(root.getCount()):
        n = root.getMFNode(i)
        if n is not None and n.getTypeName() == "TurtleBot3Burger":
            return n
    return None


robot = find_robot()
if robot is None:
    print("[overhead_cam] TurtleBot3Burger 노드를 찾지 못했습니다. 정지 카메라로 동작합니다.", flush=True)

step = 0
while sup.step(ts) != -1:
    step += 1
    if robot is not None:
        p = robot.getPosition()
        o = robot.getOrientation()
        yaw = math.atan2(o[3], o[0])
        f_tr.setSFVec3f([p[0], p[1], p[2] + HEIGHT])
        f_rot.setSFRotation([0.0, 0.0, 1.0, yaw])
    if step % SAVE_EVERY == 0:
        OUT = output_path()
        img = cam.getImage()
        if img is None:
            continue
        frame = np.frombuffer(img, np.uint8).reshape((H, W, 4))
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # 로봇 위치(화면 중앙)에 진행 방향 표시
        cx, cy = W // 2, H // 2
        cv2.circle(bgr, (cx, cy), 10, (0, 140, 255), 2)
        cv2.line(bgr, (cx, cy), (cx, cy - 26), (0, 140, 255), 2)
        try:
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                tmp = OUT + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(buf.tobytes())
                os.replace(tmp, OUT)
        except OSError:
            pass
