# AMR 강화학습 (PPO) 내비게이션 시스템

Webots TurtleBot3 Burger의 360도 2D LiDAR 센서와 엔코더 위치 추정(Odometry)을 활용하여, 무작위로 생성되는 다양한 장애물(Domain Randomization) 환경에서 목표 지점까지 안전하고 빠르게 자율주행하는 PPO 강화학습 파이프라인입니다.

---

## 1. 시스템 구조

```text
[Mission Supervisor (Webots)]
  - 매 에피소드마다 5개 이상의 장애물(상자, 원기둥)을 겹치지 않게 무작위 배치
  - 로봇 및 조난자(Target) 위치 랜덤화
       │
       ▼
[Gymnasium Env (rl/fast_env.py)]
  - State: 36개 LiDAR 거리 (정규화 [0, 1]) + 목표 거리/방향(sin, cos) + 출발점 거리 = 40차원 벡터
  - Action: [전진 속도 v, 회전 각속도 w] ∈ [-1, 1]
  - Reward: 목표 접근 보상(+), 안전거리 유지 보상, 충돌 페널티(-100), 목표 도착 보상(+150)
       │
       ▼
[PPO Policy (Stable-Baselines3)]
  - Actor-Critic MLP (128 x 128)
  - 학습 완료 모델: rl/amr_ppo_model.zip & rl/amr_policy.pt
       │
       ▼
[실제 로봇 컨트롤러 (controllers/amr_controller/amr_controller.py)]
  - 학습된 PPO 정책 모델을 로드하여 Webots 환경에서 실시간 추론(Inference) 주행
```

---

## 2. 파일 구성

- **`rl/fast_env.py`**: Webots 월드와 정확히 일치하는 6x6m 경기장, 벽, 5개 이상의 무작위 장애물, 36-ray LiDAR 레이캐스팅을 구현한 초고속 2D 키네마틱 Gymnasium 환경 (>5,000 steps/s).
- **`rl/train_ppo.py`**: Stable-Baselines3 PPO 알고리즘을 이용한 강화학습 훈련 스크립트.
- **`rl/evaluate.py`**: 훈련된 모델을 3개, 5개, 7개 장애물 밀도 환경에서 스트레스 테스트하고 성공률(%)과 충돌률(%)을 측정하는 평가 스크립트.
- **`controllers/mission_supervisor/mission_supervisor.py`**: Webots 상에서 다양한 크기의 장애물을 자동으로 무작위 재배치하는 도메인 랜덤화 수퍼바이저.

---

## 3. 로컬 실행 방법

### ① 강화학습 훈련
```bash
python3 rl/train_ppo.py
```
* Mac CPU / Apple Silicon (MPS) 환경에서 약 1~2분 만에 300,000 스텝 훈련이 완료됩니다.
* 훈련 완료 시 `rl/amr_ppo_model.zip` 및 `rl/amr_policy.pt`가 저장됩니다.

### ② 모델 평가 및 스트레스 테스트
```bash
python3 rl/evaluate.py
```
* 장애물 개수를 바꿔가며 성공률과 충돌률을 검증합니다.

### ③ Webots에서 실시간 자율주행 확인
1. Webots에서 `worlds/sensor_playground.wbt`를 엽니다.
2. 재생(Play) 버튼을 누릅니다.
3. `amr_controller`가 학습된 PPO 정책을 자동으로 인식하여 장애물을 능동적으로 회피하며 주행합니다.
4. 키보드 `E` 키로 RL 정책 / 룰 기반 회피 모드를 전환할 수 있습니다.

---

## 4. 학교 GPU 서버로 이전 및 대규모 학습 가이드

로컬에서 동작이 검증되었으므로, 학교 GPU 서버(Linux)로 옮겨 수백만 스텝의 초대규모 학습을 돌릴 수 있습니다:

1. **저장소 복사**: 프로젝트 폴더를 서버로 복사 (scp 또는 git).
2. **가상환경 패키지 설치**:
   ```bash
   pip install torch gymnasium stable-baselines3
   ```
3. **병렬 환경(SubprocVecEnv) 학습**:
   * `rl/train_ppo.py`에서 `SubprocVecEnv`를 사용하여 여러 CPU 코어로 동시 수집하면 1,000,000 스텝도 수 분 내에 학습 가능합니다.
4. **학습된 가중치 복사**:
   * 서버에서 학습 완료된 `amr_ppo_model.zip` 파일을 로컬 Mac의 `rl/` 디렉토리로 가져오면 Webots에서 바로 동작합니다.
