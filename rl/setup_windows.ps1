# Windows GPU 학습 환경 준비 스크립트
# 사용법 (PowerShell, 저장소 루트에서):  powershell -ExecutionPolicy Bypass -File rl\setup_windows.ps1
# 요구 사항: Python 3.10 ~ 3.12, NVIDIA 드라이버(nvidia-smi 동작)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "[1/3] .venv 생성"
    python -m venv .venv
}
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "[2/3] 패키지 설치 (CUDA 12.8 빌드 torch)"
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install "numpy<2.3" gymnasium "stable-baselines3>=2.3" tensorboard

Write-Host "[3/3] CUDA 확인"
& $py -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
Write-Host "완료. 학습 실행:  powershell -ExecutionPolicy Bypass -File rl\train_windows.ps1"
