# =========================
# worker.ps1
# =========================
param(
    [int]$Offset
)
# ====== conda 环境 ======
# & "C:\Users\taizun\miniconda3\shell\condabin\conda-hook.ps1"
# conda activate rl_torch
# ====== 配置区 ======
# 实验命令列表
# $experiments = @(
#     '.\target\release\SubgamePrune0923.exe "cfr/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo cfr',
#     '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo sp',  
#     '.\target\release\SubgamePrune0923.exe "cfr-sp-general/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo sp-general',    
#     # '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only_c/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo spc',    
#     # '.\target\release\SubgamePrune0923.exe "cfr-sp-general_c/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo sp-general-c',
#     '.\target\release\SubgamePrune0923.exe "cfr-plus/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo cfr --cfr-plus',
#     '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo sp --cfr-plus',
#     '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo sp-general --cfr-plus'
#     # '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only_c/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo spc --cfr-plus',
#     # '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general_c/leduc5-3-2" "run time test" --env leduc5-3-2 --epoch 100000 --algo sp-general-c --cfr-plus'
#  )
# $experiments = @(
# '.\target\release\SubgamePrune0923.exe "cfr/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo cfr',    
# '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp',    
# '.\target\release\SubgamePrune0923.exe "cfr-sp-general/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp-general',    
#  '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp --cfr-plus',
#   '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp-general --cfr-plus',
#    '.\target\release\SubgamePrune0923.exe "cfr-plus/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo cfr --cfr-plus'    
# )
# $experiments = @(
#     '.\target\release\SubgamePrune0923.exe "cfr/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo cfr',
#     '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp',
#     '.\target\release\SubgamePrune0923.exe "cfr-sp-general/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp-general',   
#     '.\target\release\SubgamePrune0923.exe "cfr-plus/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo cfr --cfr-plus',
#     '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp --cfr-plus',
#     '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp-general --cfr-plus'
# )
$experiments = @(
    '.\target\release\SubgamePrune0923.exe "cfr/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 40000 --algo cfr',
    '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo sp',
    '.\target\release\SubgamePrune0923.exe "cfr-sp-general/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo sp-general',
    # '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only_c/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo spc',
    # '.\target\release\SubgamePrune0923.exe "cfr-sp-general_c/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo sp-general-c',
    # '.\target\release\SubgamePrune0923.exe "cfr-plus/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 40000 --algo cfr --cfr-plus',
    '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo sp --cfr-plus',
    '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo sp-general --cfr-plus',
    # '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only_c/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo spc --cfr-plus',
    # '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general_c/tic_tac_toe" "测时实验" --env tic_tac_toe --epoch 100000 --algo sp-general-c --cfr-plus'
    # '.\target\release\SubgamePrune0923.exe "cfr/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 200000 --algo cfr',
    '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo sp',    
    '.\target\release\SubgamePrune0923.exe "cfr-sp-general/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo sp-general',    
    # '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only_c/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo spc'    
    # '.\target\release\SubgamePrune0923.exe "cfr-sp-general_c/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo sp-general-c'
    # '.\target\release\SubgamePrune0923.exe "cfr-plus/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 200000 --algo cfr --cfr-plus',
    '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo sp --cfr-plus',
    '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo sp-general --cfr-plus'
    # '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only_c/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo spc --cfr-plus',
    # '.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general_c/tiny_bridge_2p" "测时实验" --env tiny_bridge_2p --epoch 1000000 --algo sp-general-c --cfr-plus'
)

# 总运行时间（小时）
$totalHours = 16
$endTime = (Get-Date).AddHours($totalHours)

# 日志目录
$logDir = "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# ====== 主循环 ======

$idx = $Offset % $experiments.Count

Write-Host "Worker started. Offset=$Offset  EndTime=$endTime" -ForegroundColor Green

while ((Get-Date) -lt $endTime) {

    $cmd = $experiments[$idx % $experiments.Count]
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $logDir "offset${Offset}_${ts}.log"

    Write-Host ""
    Write-Host "[Run] $cmd" -ForegroundColor Cyan

    # 👉 同时输出到窗口 + 日志
    Invoke-Expression $cmd 2>&1 |
    Tee-Object -FilePath $log -Append |
    Out-Host

    $idx++
    sleep 1
}

Write-Host "Time reached. Worker exiting." -ForegroundColor Yellow
