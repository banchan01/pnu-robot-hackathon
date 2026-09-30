# Windows GPU 머신에서 미션 PPO 학습 돌리기

학습 대상은 고정된 아파트 월드에서의 실제 미션입니다.

```
출발 (-0.30, -7.50) -> 빨간 사과 A (-5.34, -10.54) -> 빨간 사과 B (-12.02, -3.02) -> 출발점 복귀
```

- 환경: `rl/mission_env.py` (팀원이 만든 `apartment_env.py` 위에 미션 구간을 얹은 것. 보행자 포함)
- 학습: `rl/train_mission_ppo.py` (시간 예산으로 종료, 기존 3M 모델에서 이어서 학습)
- 평가: `rl/eval_mission.py` (전체 미션 성공률과 궤적 그림)
- 관측·행동 형식은 `controllers/amr_controller`가 쓰는 40차원 그대로라서, 결과 모델을 `rl/amr_ppo_model.zip`에 넣으면 컨트롤러 수정 없이 바로 동작합니다.

## 1. 윈도우에서 준비 (한 번만)

PowerShell을 열고 순서대로 실행합니다. Python 3.10~3.12 와 NVIDIA 드라이버가 있어야 합니다.

```powershell
git clone https://github.com/banchan01/pnu-robot-hackathon
cd pnu-robot-hackathon
git fetch origin
git checkout windows-gpu-train
powershell -ExecutionPolicy Bypass -File rl\setup_windows.ps1
```

마지막 줄에 `cuda available: True` 와 GPU 이름이 나오면 준비가 끝난 것입니다.

## 2. 학습 실행 (15분 안에 끝남)

```powershell
powershell -ExecutionPolicy Bypass -File rl\train_windows.ps1 -Minutes 13 -Tag win1 -Deploy
```

- 13분 학습 후 자동 저장하고, 전체 미션 10회 평가를 거쳐 최종 모델과 학습 중 최고 모델 가운데 더 좋은 쪽을 고릅니다.
- `-Deploy` 를 주면 고른 모델을 `rl\amr_ppo_model.zip` 에 복사합니다. 컨트롤러가 이 파일을 읽습니다.
- 로그는 `rl\logs\mission_<시각>.log`, 결과물은 `rl\amr_win1_ppo_model.zip`, `rl\best_win1_model\best_model.zip` 입니다.
- 20초마다 진행 상황이 출력됩니다. `success` 는 에피소드 성공률, `legs/ep` 는 에피소드당 완료한 구간 수(최대 3)입니다.

궤적을 그림으로 확인하려면:

```powershell
.venv\Scripts\python.exe rl\eval_mission.py --model rl\amr_ppo_model.zip --episodes 20 --plot
```

`rl\mission_eval.png` 가 만들어집니다. 초록 선이 성공한 주행, 붉은 선이 실패한 주행입니다.

## 3. 결과를 팀 저장소로 보내기

```powershell
git add rl\amr_ppo_model.zip rl\amr_win1_ppo_model.zip rl\amr_win1_policy.pt rl\best_win1_model
git commit -m "feat(rl): windows gpu mission model"
git push origin windows-gpu-train
```

푸시에는 GitHub 로그인이 필요합니다. `winget install GitHub.cli` 후 `gh auth login` 을 한 번 해두면 됩니다.

## 4. 맥에서 원격으로 조작하기 (선택)

맥의 Claude 세션이 윈도우 머신에 직접 명령을 보내려면 윈도우에 OpenSSH 서버가 켜져 있어야 합니다.
Windows 설정 > 시스템 > 선택적 기능에서 "OpenSSH 서버"를 추가하고, 서비스 `sshd` 를 시작한 뒤,
맥 쪽 공개키(`cat ~/.ssh/id_ed25519.pub` 출력)를 윈도우 계정의 `authorized_keys` 에 등록합니다.
그 다음 윈도우의 IPv4 주소와 사용자 이름을 맥 쪽에 알려주면 됩니다. 두 컴퓨터가 같은 Wi-Fi 에 있어야 합니다.

## 참고

- 환경 시뮬레이션은 CPU 에서 돌고 GPU 는 정책 업데이트에 쓰입니다. 속도는 병렬 환경 수(CPU 코어 수)에 더 좌우됩니다. 맥 CPU 6개 환경으로 2분에 126만 스텝이 나왔습니다.
- 학습 스크립트는 보행자 출발 위치를 매 에피소드 무작위로 바꾸므로, 보행자 타이밍이 실제와 달라도 대응합니다.
- 컨트롤러 배포 모델을 건드리지 않고 실험만 하려면 `-Deploy` 를 빼면 됩니다.
