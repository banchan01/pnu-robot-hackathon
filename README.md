# PNU Robot Hackathon

Webots R2025a와 TurtleBot3 Burger를 사용하는 Autonomous Search and Rescue 프로젝트입니다.

현재 단계의 목표는 다음 세 가지입니다.

1. Webots에서 로봇과 간단한 장애물 월드를 실행한다.
2. 2D LiDAR, wheel encoder, camera 입력을 확인한다.
3. 키보드 조종과 최소 안전 정지 로직을 기반으로 자율주행 코드를 확장한다.

## 실행

1. Webots R2025a를 설치합니다.
2. Webots에서 `worlds/sensor_playground.wbt`를 엽니다.
3. 상단 재생 버튼을 누릅니다.
4. 3D 화면을 클릭한 뒤 `W/A/S/D`로 로봇을 움직입니다.
5. `Space`로 수동/자동 안전주행 모드를 전환하고 `Q`로 정지합니다.

컨트롤러 콘솔에는 LiDAR 전후좌우 거리, 좌우 encoder 값, 카메라 해상도와 현재 모드가 출력됩니다.

## 센서 정책

- 필수: 360도 2D LiDAR, wheel encoder
- 객체 탐지 전용: camera
- 선택: accelerometer, gyro
- 사용하지 않음: compass, GPS/GNSS

## 구조

```text
controllers/amr_controller/amr_controller.py  # Webots 로봇 제어 진입점
worlds/sensor_playground.wbt                   # 자체 제작 최소 테스트 월드
docs/architecture.md                           # 확장할 전체 파이프라인
```

## 다음 구현 순서

1. Odometry
2. Occupancy grid mapping
3. Frontier exploration
4. Object detection
5. A* global planner
6. Dynamic obstacle avoidance/local planner
7. Return-to-start mission state machine

