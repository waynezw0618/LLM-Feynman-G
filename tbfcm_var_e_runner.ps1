# tbfcm_var_e_runner.ps1 — TBFCM e 变体 10 目标 minimax 后台跑批 (方案 2)
$ErrorActionPreference = 'Stop'
Set-Location 'd:\AI\LLM-Feynman-G'

# Read minimax API key from evoscientist config
$key = $null
foreach ($line in (Get-Content 'C:\Users\lenovo\.config\evoscientist\config.yaml')) {
    if ($line -match '^minimax_api_key:\s*(.+?)\s*$') {
        $key = $Matches[1]
        break
    }
}
if (-not $key) { throw 'minimax_api_key not found' }
$env:LGF_LLM_API_KEY = $key
$env:PYTHONIOENCODING = 'utf-8'
$env:LGF_LLM_DEBUG = '1'
Write-Host "[tbfcm_var_e] env set; key tail: ...$($key.Substring($key.Length-4))"

# Run all 10 targets of e variant in one process (serial); ~5-7 min/target -> 60-80 min total
$argList = @('-X', 'utf8', 'scripts\rc_asm_tbfcm_var_mix.py',
             '--variant', 'e',
             '--backend', 'minimax',
             '--gens', '30',
             '--pop', '50',
             '--budget', '300',
             '--sub_sample', '300',
             '--max_edges', '12',
             '--max_params', '6',
             '--seed', '0')
$proc = Start-Process -FilePath 'C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe' `
    -ArgumentList $argList `
    -RedirectStandardOutput 'd:\AI\LLM-Feynman-G\outputs\asm\_tbfcm_var_e_lgf_go.log' `
    -RedirectStandardError  'd:\AI\LLM-Feynman-G\outputs\asm\_tbfcm_var_e_lgf_go.err.log' `
    -WorkingDirectory 'd:\AI\LLM-Feynman-G' `
    -WindowStyle Hidden `
    -PassThru
Write-Host "[tbfcm_var_e] launched PID=$($proc.Id) at $(Get-Date -Format 'HH:mm:ss')"
Write-Host "[tbfcm_var_e] log: outputs\asm\_tbfcm_var_e_lgf_go.log"
Write-Host "[tbfcm_var_e] ETA: ~60-80 min (10 targets x 6-8 min)"