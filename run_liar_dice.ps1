# =========================================
# 🚀 一键运行多个 cargo 进程，每个绑定一个 CPU 核心
# powershell -ExecutionPolicy Bypass -File .\run_liar_dice.ps1
# =========================================

# 你的实验命令列表（6 个）
$experiments = @(
    @{Title="Exp1_CFR";   Cmd='cargo run "cfr/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo cfr'},
    @{Title="Exp3";  Cmd='cargo run "cfr-sp-check-free-only/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp'},
    @{Title="Exp5"; Cmd='cargo run "cfr-sp-general/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp-general'},
    @{Title="Exp7";  Cmd='cargo run "cfr-sp-check-free-only_c/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo spc'},
    @{Title="Exp9"; Cmd='cargo run "cfr-sp-general_c/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp-general-c'}
    # @{Title="Exp2_CFR+";  Cmd='cargo run "cfr-plus/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo cfr --cfr-plus'},
    # @{Title="Exp4"; Cmd='cargo run "cfr-plus-sp-check-free-only/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp --cfr-plus'},
    # @{Title="Exp6";  Cmd='cargo run "cfr-plus-sp-general/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp-general --cfr-plus'},
    # @{Title="Exp8"; Cmd='cargo run "cfr-plus-sp-check-free-only_c/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo spc --cfr-plus'},
    # @{Title="Exp10";  Cmd='cargo run "cfr-plus-sp-general_c/liars_dice" "测时实验" --env liars_dice --epoch 100000 --algo sp-general-c --cfr-plus'}
)

# 每个进程绑定一个核心注意这里的编号是逻辑核心编号
# 每个进程绑定到不同核心（掩码为16进制位）
# $masks = @("1", "2", "4", "8", "10", "20")
$masks = @("1", "4", "10", "40","100","400","1000","4000")

# 创建日志目录
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

for ($i = 0; $i -lt $experiments.Count; $i++) {
    $mask = $masks[$i]
    $cmd = $experiments[$i].Cmd
    $log = "logs\Exp$($i+1).log"

    Write-Host "run Exp$($i+1) core 0x$mask..." -ForegroundColor Cyan

    # ✅ 不设置标题，直接启动独立窗口 + 锁核 + 输出日志
    # cmd /c start "" /affinity $mask powershell -NoExit -Command "$cmd *> '$log' 2>&1"
    cmd /c start "" /affinity $mask powershell -NoExit -Command "$cmd"
}