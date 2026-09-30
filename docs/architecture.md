# Architecture

```text
LiDAR + wheel encoder + camera
              |
              v
Perception / Localization / Mapping
              |
              v
Mission state: EXPLORE -> APPROACH -> RESCUE -> RETURN -> DONE
              |
              v
Global planner + local collision avoidance
              |
              v
left/right wheel velocity
```

초기 컨트롤러는 센서 연결, 수동 조종, 간단한 LiDAR 기반 안전 정지만 담당합니다. 이후 각 기능을 독립 모듈로 분리합니다.

강화학습을 적용한다면 전체 시스템을 대체하기보다 local planner 후보로 연결합니다. 관측값은 축약한 LiDAR 거리와 목표 방향, 행동은 좌우 바퀴 속도, 보상은 충돌·안전거리·진행도·목표 도달을 기준으로 설계합니다.

