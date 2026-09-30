# Windows GPU에서 고정 미션(출발 -> 빨간 사과 A -> 빨간 사과 B -> 복귀) PPO 학습 실행
# 사용법:  powershell -ExecutionPolicy Bypass -File rl\train_windows.ps1 [-Minutes 13] [-Envs 14] [-Tag win1] [-Deploy]
#   -Minutes : 학습 시간 예산(분). 시간이 다 되면 자동 저장 후 종료. 평가까지 포함해 약 2분 더 걸림.
#   -Envs    : 병렬 환경 수. CPU 코어 수 - 2 가 기본값.
#   -Deploy  : 결과를 rl\amr_ppo_model.zip 에 덮어써서 amr_controller 가 바로 쓰게 함.
param(
    [double]$Minutes = 13,
    [int]$Envs = [Math]::Max(4, [Environment]::ProcessorCount - 2),
    [string]$Tag = "win",
    [string]$Device = "auto",
    [switch]$Deploy
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw ".venv 이 없습니다. 먼저 rl\setup_windows.ps1 을 실행하세요." }

New-Item -ItemType Directory -Force -Path "rl\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "rl\logs\mission_$stamp.log"
Write-Host "minutes=$Minutes envs=$Envs device=$Device tag='$Tag' deploy=$Deploy  log=$log"

$extra = @()
if ($Deploy) { $extra += "--deploy" }
& $py rl\train_mission_ppo.py --minutes $Minutes --envs $Envs --device $Device --tag "$Tag" @extra 2>&1 | Tee-Object -FilePath $log

Write-Host ""
Write-Host "궤적 그림:  $py rl\eval_mission.py --model rl\best_${Tag}_model\best_model.zip --episodes 20 --plot"
