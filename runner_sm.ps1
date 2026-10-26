# runner_sm.ps1 — shipman 收尾实验矩阵后台跑批 (A1-A4, 断点续跑)
# 用法: powershell -ExecutionPolicy Bypass -File runner_sm.ps1
$ErrorActionPreference = 'Stop'
Set-Location 'd:/AI/LLM-Feynman-G'

# API key (llm 臂需要; gponly 臂不用): 沿用 asm runner 机制
$cfg = 'C:/Users/lenovo/.config/evoscientist/config.yaml'
$key = $null
if (Test-Path $cfg) {
    foreach ($line in Get-Content $cfg) {
        if ($line -match '^\s*minimax_api_key:\s*"?([A-Za-z0-9_\-\.]+)"?') { $key = $Matches[1]; break }
    }
}
if ($key) { $env:LGF_LLM_API_KEY = $key; Write-Host "API key loaded" } else { Write-Host "WARN: no API key, llm arm will fail" }
$env:PYTHONIOENCODING = 'utf-8'
$env:LGF_LLM_DEBUG = '1'
# 清理可能损坏的代理环境变量 (Start-Process ArgumentException 根因)
foreach ($v in @('HTTPS_PROXY','https_proxy','HTTP_PROXY','http_proxy','ALL_PROXY','all_proxy','NO_PROXY','no_proxy')) {
    if ([Environment]::GetEnvironmentVariable($v) -ne $null) { [Environment]::SetEnvironmentVariable($v, $null, 'Process') }
}

$py = 'C:/Users/lenovo/AppData/Local/Programs/Python/Python313/python.exe'
$log = 'outputs/shipman/_runner_sm.log'
$err = 'outputs/shipman/_runner_sm.err.log'

# 任务清单: A1 核心 16 → A2 消融 4 → A3 多种子 2 → A4 误吸收 1
$jobs = @()
foreach ($v in @('v1','v2','v3','v4')) {
    foreach ($eq in @('Y','N')) {
        foreach ($n in @('0.0','0.03')) {
            $jobs += @{ v=$v; eq=$eq; n=$n; seed='0'; arm='gponly'; be='heuristic'; budget='180'; pop='60'; gens='40'; data='' }
        }
    }
}
foreach ($v in @('v1','v2')) {
    foreach ($n in @('0.0','0.03')) {
        $jobs += @{ v=$v; eq='Y'; n=$n; seed='0'; arm='llm'; be='glm53'; budget='300'; pop='60'; gens='40'; data='' }
    }
}
foreach ($seed in @('1','2')) {
    $jobs += @{ v='v2'; eq='Y'; n='0.0'; seed=$seed; arm='gponly'; be='heuristic'; budget='180'; pop='60'; gens='40'; data='' }
}
$jobs += @{ v='v2'; eq='Y'; n='0.0'; seed='0'; arm='gponly'; be='heuristic'; budget='180'; pop='60'; gens='40'; data='v1' }

$total = $jobs.Count
$i = 0
foreach ($j in $jobs) {
    $i++
    $argList = @('-X','utf8','shipman/rc_sm_run.py','--variant',$j.v,'--eq',$j.eq,
                 '--noise',$j.n,'--seed',$j.seed,'--arm',$j.arm,'--backend',$j.be,
                 '--budget',$j.budget,'--pop',$j.pop,'--gens',$j.gens)
    if ($j.data) { $argList += @('--data',$j.data) }
    $tag = "[$i/$total] v=$($j.v) eq=$($j.eq) n=$($j.n) s=$($j.seed) arm=$($j.arm)"
    Write-Host "$tag starting..."
    Add-Content $log "=== $tag $(Get-Date -Format 'HH:mm:ss') ==="
    $p = Start-Process -FilePath $py -ArgumentList $argList -RedirectStandardOutput $log `
        -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "$tag exit=$($p.ExitCode)"
}
Write-Host "ALL DONE ($total jobs)"
