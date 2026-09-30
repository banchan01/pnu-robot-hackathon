# Windows GPU 머신에서 PPO 학습 돌리기

## 1. 윈도우에서 준비 (한 번만)

PowerShell을 열고 순서대로 실행합니다.

```powershell
git clone https://github.com/banchan01/pnu-robot-hackathon
cd pnu-robot-hackathon
git checkout windows-gpu-train
powershell -ExecutionPolicy Bypass -File rl\setup_windows.ps1
```

마지막 줄에 `cuda available: True` 와 GPU 이름이 나오면 준비가 끝난 것입니다.
`False` 가 나오면 `nvidia-smi` 가 동작하는지, Python 이 3.10~3.12 인지 확인합니다.

## 2. 학습 실행

```powershell
powershell -ExecutionPolicy Bypass -File rl\train_windows.ps1 -Steps 5000000 -Envs 16 -Tag win1
```

- `-Envs` 는 CPU 코어 수보다 2 작게 두는 것이 좋습니다. 환경 시뮬레이션은 CPU 에서 돌고, GPU 는 정책 업데이트에만 쓰입니다.
- 결과물은 `rl/amr_apartment_ppo_model_<tag>.zip`, `rl/amr_apartment_policy_<tag>.pt`, `rl/best_apartment_model/best_model.zip` 입니다.
- 로그는 `rl/logs/train_<시각>.log` 에 남습니다.

학습된 모델을 팀 저장소로 보내려면:

```powershell
git add rl\amr_apartment_ppo_model_win1.zip rl\amr_apartment_policy_win1.pt rl\best_apartment_model
git commit -m "feat(rl): windows gpu 5M step model"
git push origin windows-gpu-train
```

## 3. 맥에서 원격으로 조작하기 (선택)

맥의 Claude 세션이 윈도우 머신에 직접 명령을 보내려면 윈도우에 OpenSSH 서버가 켜져 있어야 합니다.
Windows 설정 > 시스템 > 선택적 기능에서 "OpenSSH 서버"를 추가하고, 서비스 `sshd` 를 시작한 뒤,
맥 쪽 공개키(`cat ~/.ssh/id_ed25519.pub` 출력)를 윈도우 계정의 `authorized_keys` 에 등록합니다.
그 다음 윈도우의 IPv4 주소와 사용자 이름을 맥 쪽에 알려주면 됩니다.

맥과 윈도우가 같은 Wi-Fi 에 있어야 합니다. 다른 네트워크라면 양쪽에 Tailscale 을 설치하면 됩니다.

맥에서 확인:

```bash
ssh <윈도우사용자>@<윈도우IP> nvidia-smi
```

SSH 없이도 됩니다. 윈도우에서 직접 Claude Code 를 실행해도 되고, 위 2번처럼 학습만 돌리고 결과를 git push 해도 됩니다.
