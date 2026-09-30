# tb3_mission — TurtleBot3 Burger 자율 Search & Rescue 컨트롤러

2026 부산대학교 TECH WEEK Physical AI 해커톤 제출물입니다.
사전 지도와 자기 위치를 모르는 아파트(`apartment.wbt`)에서 LiDAR와 휠 엔코더만으로
지도를 만들며 탐색하고, 카메라로 **빨간 사과 2개**를 찾아 접근한 뒤 시작 지점으로 복귀합니다.
다른 색 사과에는 접근하지 않으며, 정적 장애물과 움직이는 보행자를 회피합니다.

> 컨셉 시나리오와 확장성 서술은 아래 [7. 창의성](#7-창의성-사람과의-소통)과 [8. 확장성](#8-확장성)에 있습니다.

---

## 1. 실행 방법

### 1-1. 요구 사항

| 항목 | 버전 |
|---|---|
| Webots | R2025a |
| Python | 3.10 (3.9 ~ 3.11 동작 확인 필요 시 3.10 권장) |
| 추가 라이브러리 | numpy, opencv-python, scipy (아래 설치 명령 참고) |

### 1-2. 설치

```bash
# 1) 가상환경 (uv 사용 예시. venv/conda도 동일)
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python numpy==1.23.5 opencv-python==4.8.0.74 scipy

# 2) Webots가 이 Python으로 컨트롤러를 실행하도록 설정
#    Webots 메뉴 Tools > Preferences > General > Python command 에 .venv/bin/python 절대 경로 입력
#    (macOS는 아래 한 줄로도 가능)
defaults write com.cyberbotics.Webots-R2025a General.pythonCommand -string "$(pwd)/.venv/bin/python"
```

Webots가 처음 `apartment.wbt`를 열 때 텍스처와 메시 자원을 내려받다가 멈추면,
자원 묶음을 캐시에 직접 풀어 넣으면 됩니다.

```bash
curl -L -C - -o /tmp/assets-R2025a.zip https://github.com/cyberbotics/webots/releases/download/R2025a/assets-R2025a.zip
mkdir -p ~/Library/Caches/Cyberbotics/Webots/assets          # Linux: ~/.cache/Cyberbotics/Webots/assets
unzip -oq /tmp/assets-R2025a.zip -d ~/Library/Caches/Cyberbotics/Webots/assets
```

### 1-3. 실행

1. `controllers/tb3_mission/` 폴더를 프로젝트의 `controllers/` 아래에 둡니다.
2. `apartment.wbt`에서 `TurtleBot3Burger`의 `controller` 필드를 `"tb3_mission"`으로 바꿉니다
   (저장소의 `worlds/apartment_mission.wbt`는 이미 바뀌어 있습니다).
3. 시뮬레이션을 시작하면 로봇이 자동으로 탐색을 시작합니다. 콘솔에 상태 로그가 찍히고,
   지도와 카메라 창이 열립니다 (창을 끄려면 `mission.json`의 `show_window`를 `false`).
4. 미션이 끝나면 `output/` 폴더에 지도 이미지, 궤적, 요약 JSON이 저장됩니다.

헤드리스 자동 테스트: `./run_headless.sh ../../worlds/apartment_mission.wbt 300 /tmp/tb3.log`

### 1-4. 설정

조정 가능한 값은 모두 `mission.json`에 있습니다. 대표 항목:

| 키 | 의미 | 기본값 |
|---|---|---|
| `target_colors`, `target_count` | 목표 색과 개수 | `["red"]`, 2 |
| `forbidden_colors` | 접근 금지 색 | green, purple, orange |
| `time_limit_s`, `return_at_ratio` | 제한 시간, 강제 복귀 시점 비율 | 900, 0.7 |
| `v_max`, `w_max`, `lookahead_m` | 주행 속도와 Look-ahead 거리 | 0.18, 1.5, 0.35 |
| `stop_dist_m`, `slow_dist_m` | 전방 정지·감속 거리 | 0.25, 0.40 |
| `inflate_cells`, `soft_cells` | Costmap 팽창 반경(셀) | 3, 7 |
| `hsv` | 색상별 HSV 범위 | 파일 참고 |

---

## 2. 시스템 구조

```
센서 읽기 ─▶ 위치 추정 ─▶ Bayesian 지도 갱신 ─▶ 사과 탐지 ─▶ 명령 수신 ─▶ Behavior Tree tick ─▶ 바퀴 속도
 (LiDAR, 엔코더,   localization.py   mapping.py         perception.py   commands.py    bt_mission.py       motion.py
  나침반, 카메라)
```

강의에서 다룬 인지 → 계획 → 행동 파이프라인과 코드의 대응은 다음과 같습니다.

| 강의 항목 | 구현 | 파일 · 함수 |
|---|---|---|
| 휠 엔코더 odometry | 좌우 회전각 증분 → 이동 거리 → 방향 분해 적분 | `localization.py` `Localizer.update` |
| 방향 추정 | 나침반 상대 각도 (시작 방향 = 0) | `localization.py` `_compass_yaw` |
| LiDAR 360도 스캔 | 360빔, 0.12 ~ 3.5 m, 인덱스 180 = 전방 | `config.py` `LIDAR_ANGLES` |
| **Bayesian Occupancy Grid** | log-odds 갱신 `L ← L + l(z)`, 자유 −0.4 / 점유 +0.9, 클램프 ±5 | `mapping.py` `OccupancyGrid.update` |
| **Costmap** | 점유 254, 내접 253, 팽창 1~252, 미관측 255 | `mapping.py` `OccupancyGrid.costmap` |
| Frontier 탐사 | 자유 셀 중 미관측 인접 셀 → 연결 성분 → 크기/경로거리 점수 | `mapping.py` `OccupancyGrid.frontiers` |
| Global Planner | 8방향 A* + Costmap 비용 + 시선 단축 | `planner.py` `astar`, `simplify` |
| Local Planner | Look-ahead 곡률 제어 + 안전 계층(정지·감속·측면 보정) | `motion.py` `Motion.follow`, `apply_safety` |
| RGBA → BGR 전처리 | `cv2.cvtColor(BGRA2BGR)` | `perception.py` `AppleDetector.detect` |
| HSV 색상 분할, 이진 마스크 | `inRange` + Opening/Closing | `perception.py` `_mask` |
| 컨투어, 최소 외접원, 무게중심 | 원형도·종횡비 필터, 외접원으로 거리 추정 | `perception.py` `detect` |
| 동적 환경 대응 | 지도 실시간 갱신, 경로 막힘 감지 시 재계획, 보행자 대기 | `planner.py` `path_blocked`, `bt_mission.py` `front_clear` |
| **Behavior Tree** | Selector/Sequence/Parallel(노트북) + Reactive·Recovery 노드 | `bt_core.py`, `bt_mission.py` |

IMU(가속도계·자이로)는 사용하지 않았습니다. 나침반이 drift 없는 방향을 주므로 자이로 적분이 필요 없었습니다.

---

## 3. 인지 (Perception)

### 위치 추정
- 엔코더: `d = R·Δφ`, `Δs = (d_r + d_l)/2`, `x += Δs·cos(θ)`, `y += Δs·sin(θ)`
- 방향: 나침반 벡터 각도의 시작 대비 변화량. 회전 방향 부호는 Supervisor ground truth로 검증했습니다.

### Bayesian Occupancy Grid
- 18 × 18 m, 5 cm 해상도 (360 × 360 셀). 시작 지점이 원점.
- 매 스캔마다 360개 빔을 2.5 cm 간격으로 샘플링하여 자유 셀에 `−l_free`, 반사 지점 셀에 `+l_occ`를 더합니다.
- 각속도가 1 rad/s를 넘는 tick은 빔이 번지므로 갱신을 건너뜁니다.
- 보행자가 지나간 자리는 이후 자유 관측이 누적되며 자연스럽게 지워집니다.

### 사과 탐지
1. BGRA → BGR → 가우시안 블러 → HSV
2. 색상 범위 마스크 (빨강은 Hue 0 부근 두 구간을 OR), Opening 1회, Closing 2회
3. 외곽 컨투어 → 면적 ≥ 50 px, 원형도 ≥ 0.55, 종횡비 0.6 ~ 1.7
4. **바닥 기하 일관성 검사**: 바닥 위 사과는 거리 `d`에서 화면 세로 위치가
   `cy = 240 + f·(h_cam − r_apple)/d` 근처에만 나타납니다. 이 값에서 30 px 이상 벗어나면
   탁자 위 과일, 벽의 그림, 크기가 다른 붉은 물체로 보고 기각합니다.
5. 최소 외접원 반지름으로 거리 `d = f·(2r_apple)/(2r_px)`, 방위 `−atan((cx − 320)/f)` → 로봇 pose로 지도 좌표 변환
6. 최근 7프레임 중 5프레임 이상 같은 위치(0.35 m)면 **확정**. 금지 색은 기록만 하고 절대 확정하지 않습니다.

---

## 4. 계획 (Planning)

- **Costmap**: 점유 셀을 로봇 반지름에 맞춰 3셀(15 cm) 팽창하여 내접 구역(253)으로, 그 밖 7셀까지 비용 경사를 둡니다.
- **Frontier 선택**: 연결 성분 크기 ÷ (로봇에서의 BFS 경로 거리 + 1)이 큰 후보. 도달 불가나 실패한 후보는 60초 블랙리스트.
- **A\***: 8방향, 대각선 모서리 끼임 방지, 미관측 셀은 탐색 목표로 갈 때만 비용 3배로 허용, 복귀 시에는 금지.
- **재계획 조건**: 목표 변경, 앞 1.5 m 경로가 새 점유 셀에 막힘, 전방 장애물 5초 지속, 끼임 복구 후.

---

## 5. 행동 (Action)

- **Look-ahead 제어**: 경로 거리 0.35 m 앞 지점을 로봇 프레임으로 변환, 곡률 `κ = 2y/(x²+y²)`, `ω = v·κ`,
  `v_r = v + ωL/2`, `v_l = v − ωL/2`. 방향 오차가 60도를 넘으면 제자리 회전.
- **안전 계층**: 전방 ±30도 최소 거리 0.40 m 이하 감속, 0.25 m 이하 정지. 측면 0.16 m 이하면 반대쪽으로 보정.
- **보행자 대응**: 정지 후 최대 5초 대기(지나가기를 기다림), 그래도 막히면 지도에 반영된 장애물로 보고 재계획.
- **끼임 복구**: 3초간 5 cm 미만 이동이면 후진 2초 → 회전 1.2초 → 재계획.
- **접근**: 사과 앞 0.35 m 지점을 A*로 가고, 시각 서보로 화면 중앙 정렬 후 0.25 m까지 접근. 사과 주변은 가상 장애물로 보호.
- **복귀**: 원점으로 A*(미관측 금지) → 도착 → 시작 방향으로 정렬 → 보고서 저장.

---

## 6. Behavior Tree

```
Root (ReactiveSequence)
├─ SafetyGate (ReactiveSequence)           매 tick 재평가 → 아래 Mission보다 항상 우선
│   ├─ NotExternallyStopped                외부 정지 명령이면 정지 유지
│   ├─ StuckRecovery (Recovery)            끼임 → Escape(후진·회전) → 재계획
│   └─ FrontClear                          전방 막힘 → 정지 대기 → 5초 후 재계획 요청
└─ Mission (ReactiveSelector)              우선순위: 강제복귀 > 접근 > 탐색 > 복귀 > 대기
    ├─ ForcedReturn (Sequence)  [return 명령 또는 제한시간 70%] → PlanHome → FollowPath → Align → Finish
    ├─ Approach     (Sequence)  [목표 확정] → ProtectTarget → PlanToStandoff → FollowPath → VisualApproach → MarkRescued
    ├─ Explore      (Sequence)  [남은 목표 > 0] → SelectFrontier → PlanToGoal → FollowPath
    ├─ ReturnHome   (Sequence)  [목표 없음 또는 frontier 소진] → PlanHome → FollowPath → Align → Finish
    └─ Idle
```

- `bt_core.py`: 노트북의 `Status`, `Node`, `Selector`, `Sequence`, `Parallel`, `Blackboard`, `BehaviorTree`에
  `ReactiveSequence`, `ReactiveSelector`, `Recovery`, `Condition`, `Action`, `Inverter`를 추가했습니다.
- 모든 잎 노드는 `Blackboard`만 읽고 씁니다 (`pose`, `costmap`, `path`, `goal`, `detection`, `targets_left`, `command_state` …).
- 우선순위가 높은 서브트리가 살아나면 낮은 서브트리는 자동으로 `reset`되어 stale 경로를 이어가지 않습니다.

---

## 7. 창의성: 사람과의 소통

_(컨셉 시나리오는 확정 후 이 절에 기술)_

구현된 소통 채널:

| 채널 | 명령 | 동작 |
|---|---|---|
| 텍스트 파일 `commands.txt` | `stop` / `resume` / `return` / `target <색> <개수>` / `status` | 한 줄 쓰면 즉시 반영 |
| 키보드 (3D 뷰 클릭 후) | `P` 정지, `R` 재개, `H` 즉시 복귀, `S` 상태 출력 | 동일 |

- 정지 명령은 Behavior Tree의 SafetyGate 첫 노드에서 처리되어 어떤 상태에서도 즉시 멈춥니다.
- `target green 1`처럼 목표를 실행 중에 바꿀 수 있습니다.
- 보행자를 만나면 정지하고 양보하며, 만난 횟수와 최소 거리를 보고서에 기록합니다.
- 미션 종료 시 `output/summary_final.json`에 구출 위치, 탐색률, 이동 거리, 복귀 오차, 이벤트 로그를 남깁니다.

---

## 8. 확장성

_(현실 시나리오별 대응과 적용처는 컨셉 확정 후 기술)_

설계상 확장 지점:
- 목표 사양은 `mission.json`에서 색과 개수만 바꾸면 되며, 탐지기는 색 목록을 받아 동작합니다.
- Local Planner(`Motion.follow`)는 `(pose, path) → 바퀴 속도` 인터페이스라 학습 기반 회피 정책으로 교체할 수 있습니다.
- 지도는 `output/logodds_final.npy`로 저장되어 다음 실행의 초기 지도로 재사용할 수 있습니다.
- Blackboard와 잎 노드 구조 덕분에 배터리 모델, 다중 로봇 영역 분담 같은 노드를 추가해도 기존 노드를 고칠 필요가 없습니다.

---

## 9. 실험 결과

_(테스트 후 기입: 완주 시간, 탐색률, 구출 좌표, 복귀 오차, 지도 이미지, 환경 변경 테스트)_

---

## 10. 한계와 알려진 문제

- 사과는 LiDAR 높이(17 cm) 아래에 있어 LiDAR로 보이지 않으므로, 접근 단계는 카메라에만 의존합니다.
- 탁자처럼 LiDAR 높이 위에 걸쳐 있는 장애물은 지도에 다리만 기록됩니다.
- 연기·조명 변화 같은 카메라 무력화 상황은 시뮬레이션에서 다루지 않았습니다.

---

## 파일 구성

```
tb3_mission/
├─ tb3_mission.py   진입점, 메인 루프
├─ config.py        상수, mission.json 로더
├─ mission.json     조정값
├─ localization.py  엔코더 + 나침반 위치 추정
├─ mapping.py       Bayesian Occupancy Grid, Costmap, Frontier
├─ planner.py       A*, 경로 단축, 막힘 감지
├─ motion.py        Look-ahead 제어, 안전 계층, 시각 서보
├─ perception.py    HSV 색상 분할 사과 탐지
├─ bt_core.py       Behavior Tree 노드
├─ bt_mission.py    미션 잎 노드와 트리 조립
├─ commands.py      외부 명령 수신
├─ report.py        디버그 화면, 결과 저장
└─ run_headless.sh  자동 테스트 스크립트
```
