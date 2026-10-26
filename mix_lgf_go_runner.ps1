# mix_lgf_go_runner.ps1
$ErrorActionPreference = 'Stop'
Set-Location 'd:\AI\LLM-Feynman-G'

# Read API key from evoscientist config (simple line-parse)
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
Write-Host "[mix-lgf-go] env set; key tail: ...$($key.Substring($key.Length-4))"

# Launch in background
$argList = @('-X', 'utf8', 'scripts\rc_asm_true_mix.py',
             '--skip_pysr', '--skip_prep', '--lgf_gens', '40')
$proc = Start-Process -FilePath 'C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe' `
    -ArgumentList $argList `
    -RedirectStandardOutput 'd:\AI\LLM-Feynman-G\outputs\asm\_trueasm_mix_lgf_go.log' `
    -RedirectStandardError  'd:\AI\LLM-Feynman-G\outputs\asm\_trueasm_mix_lgf_go.err.log' `
    -WorkingDirectory 'd:\AI\LLM-Feynman-G' `
    -WindowStyle Hidden `
    -PassThru
Write-Host "[mix-lgf-go] launched PID=$($proc.Id) at $(Get-Date -Format 'HH:mm:ss')"
