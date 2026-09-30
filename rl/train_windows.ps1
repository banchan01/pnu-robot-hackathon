# Windows GPU에서 아파트 환경 PPO 학습 실행
# 사용법:  powershell -ExecutionPolicy Bypass -File rl\train_windows.ps1 [-Steps 5000000] [-Envs 16] [-Tag run1]
param(
    [int]$Steps = 5000000,
    [int]$Envs = [Math]::Max(4, [Environment]::ProcessorCount - 2),
    [string]$Tag = "",
    [string]$Device = "auto"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw ".venv 이 없습니다. 먼저 rl\setup_windows.ps1 을 실행하세요." }

New-Item -ItemType Directory -Force -Path "rl\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "rl\logs\train_$stamp.log"
Write-Host "steps=$Steps envs=$Envs device=$Device tag='$Tag'  log=$log"

& $py rl\train_apartment_ppo.py --steps $Steps --envs $Envs --subproc --device $Device --tag "$Tag" 2>&1 | Tee-Object -FilePath $log
