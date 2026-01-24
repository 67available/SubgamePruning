# =========================================
# 🚀 一键运行多个 cargo 进程，每个绑定一个 CPU 核心 
#  powershell -ExecutionPolicy Bypass -File .\run_leduc.ps1
# =========================================

# 你的实验命令列表（6 个）
$experiments = @(
    @{Title = "Exp1_CFR"; Cmd = '.\target\release\SubgamePrune0923.exe "cfr/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo cfr' },    
    @{Title = "Exp3"; Cmd = '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp' },    
    @{Title = "Exp5"; Cmd = '.\target\release\SubgamePrune0923.exe "cfr-sp-general/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp-general' },
    @{Title = "Exp7"; Cmd = '.\target\release\SubgamePrune0923.exe "cfr-sp-check-free-only_c/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo spc' },    
    @{Title = "Exp9"; Cmd = '.\target\release\SubgamePrune0923.exe "cfr-sp-general_c/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp-general-c' }
    # @{Title="Exp4"; Cmd='.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp --cfr-plus'},
    # @{Title="Exp6";  Cmd='.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp-general --cfr-plus'},
    # @{Title="Exp2_CFR+";  Cmd='.\target\release\SubgamePrune0923.exe "cfr-plus/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo cfr --cfr-plus'},
    # @{Title="Exp8"; Cmd='.\target\release\SubgamePrune0923.exe "cfr-plus-sp-check-free-only_c/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo spc --cfr-plus'},
    # @{Title="Exp10";  Cmd='.\target\release\SubgamePrune0923.exe "cfr-plus-sp-general_c/leduc_poker" "测时实验" --env leduc_poker --epoch 1000000 --algo sp-general-c --cfr-plus'}
)

# 每个进程绑定一个核心注意这里的编号是逻辑核心编号
# 每个进程绑定到不同核心（掩码为16进制位）
# $masks = @("1", "2", "4", "8", "10", "20")
$masks = @("1", "4", "10", "40", "1000", "4000")

# 创建日志目录
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
foreach ($k in 1..10) {
    for ($i = 0; $i -lt $experiments.Count; $i++) {
        $mask = $masks[$i]
        $cmd = $experiments[$i].Cmd
        $log = "logs\Exp$($i+1).log"

        Write-Host "run Exp$($i+1) core 0x$mask..." -ForegroundColor Cyan

        # ✅ 不设置标题，直接启动独立窗口 + 锁核 + 输出日志
        # cmd /c start "" /affinity $mask powershell -NoExit -Command "$cmd *> '$log' 2>&1"
        cmd /c start "" /affinity $mask powershell -NoExit -Command "$cmd"
    }
    sleep 3000
}
