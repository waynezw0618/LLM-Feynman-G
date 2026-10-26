# runner_sm_resume.ps1 — 断链续跑 (2026-10-15): 剩余 runs 由本脚本直接串行执行
# (不依赖原 runner 的 Start-Process 链; 在本进程内同步跑, 断点续跑由 result.json SKIP)
$ErrorActionPreference = 'Stop'
Set-Location 'd:/AI/LLM-Feynman-G'

$cfg = 'C:/Users/lenovo/.config/evoscientist/config.yaml'
$key = $null
if (Test-Path $cfg) {
    foreach ($line in Get-Content $cfg) {
        if ($line -match '^\s*minimax_api_key:\s*"?([A-Za-z0-9_\-\.]+)"?') { $key = $Matches[1]; break }
    }
}
if ($key) { $env:LGF_LLM_API_KEY = $key }
$env:PYTHONIOENCODING = 'utf-8'
$env:LGF_LLM_DEBUG = '1'
foreach ($v in @('HTTPS_PROXY','https_proxy','HTTP_PROXY','http_proxy','ALL_PROXY','all_proxy')) {
    if ([Environment]::GetEnvironmentVariable($v) -ne $null) { [Environment]::SetEnvironmentVariable($v, $null, 'Process') }
}

$py = 'C:/Users/lenovo/AppData/Local/Programs/Python/Python313/python.exe'

# 完整 23 任务清单 (与 runner_sm.ps1 一致); 已完成的会被 SKIP
$jobs = @()
foreach ($v in @('v1','v2','v3','v4')) {
    foreach ($eq in @('Y','N')) {
        foreach ($n in @('0.0','0.03')) {
            $jobs += @{ v=$v; eq=$eq; n=$n; seed='0'; arm='gponly'; be='heuristic'; budget='180'; data='' }
        }
    }
}
foreach ($v in @('v1','v2')) {
    foreach ($n in @('0.0','0.03')) {
        $jobs += @{ v=$v; eq='Y'; n=$n; seed='0'; arm='llm'; be='glm53'; budget='300'; data='' }
    }
}
foreach ($seed in @('1','2')) {
    $jobs += @{ v='v2'; eq='Y'; n='0.0'; seed=$seed; arm='gponly'; be='heuristic'; budget='180'; data='' }
}
$jobs += @{ v='v2'; eq='Y'; n='0.0'; seed='0'; arm='gponly'; be='heuristic'; budget='180'; data='v1' }

$total = $jobs.Count; $i = 0; $skipped = 0
foreach ($j in $jobs) {
    $i++
    $tag = "" ; if ($j.data) { $tag = "$($j.v)pool_$($j.data)data_" }
    $rd = "outputs/shipman/runs_sm/${tag}$($j.v)_$($j.eq)_n$($j.n)_s$($j.seed)_$($j.arm)"
    if (Test-Path "$rd/result.json") { $skipped++; Write-Host "[$i/$total] SKIP $rd"; continue }
    $argList = @('-X','utf8','shipman/rc_sm_run.py','--variant',$j.v,'--eq',$j.eq,
                 '--noise',$j.n,'--seed',$j.seed,'--arm',$j.arm,'--backend',$j.be,
                 '--budget',$j.budget,'--pop','60','--gens','40')
    if ($j.data) { $argList += @('--data',$j.data) }
    Write-Host "[$i/$total] $($j.v) $($j.eq) n=$($j.n) s=$($j.seed) $($j.arm) running..."
    & $py @argList 2>&1 | Select-Object -Last 2 | Write-Host
}
Write-Host "RESUME DONE: $total jobs, $skipped skipped, $($total-$skipped) executed"
