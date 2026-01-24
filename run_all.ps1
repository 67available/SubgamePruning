# =========================
# run_all.ps1
# =========================

# 逻辑核心掩码（你原来能跑的那组）
# $masks = @("1", "4", "10", "40", "1000", "4000")
$masks = @("4", "10")

for ($i = 0; $i -lt $masks.Count; $i++) {

    $offset = $i       # ⭐ 起始偏移
    $mask = $masks[$i]

    Write-Host "Start worker offset=$offset core=0x$mask" -ForegroundColor Cyan

    cmd /c start "" /affinity $mask powershell.exe `
        -ExecutionPolicy Bypass `
        -NoExit `
        -File "$PSScriptRoot\worker.ps1" `
        -Offset $offset
    
    sleep 1
}
# 总运行时间（小时）
$totalHours = 16
$endTime = (Get-Date).AddHours($totalHours)
while ((Get-Date) -lt $endTime) {
    sleep 10
}
Get-CimInstance Win32_Process | ? { $_.ExecutablePath -like "*\SubgamePrune0923.exe" } | % { Stop-Process -Id $_.ProcessId -Force }
Write-Host "END" -ForegroundColor Cyan