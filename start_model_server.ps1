param(
    [int]$Port = 8001,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$pythonCandidates = @(
    (Join-Path $projectDir ".venv\Scripts\python.exe"),
    (Join-Path $projectDir "venv\Scripts\python.exe")
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Create .venv or install Python first."
    }
    $python = $pythonCommand.Source
}

$env:MODEL_SERVER_PORT = "$Port"
$dashboardUrl = "http://127.0.0.1:$Port/"
$browserJob = $null

if (-not $NoBrowser) {
    $browserJob = Start-Job -ScriptBlock {
        param($Url)
        for ($attempt = 0; $attempt -lt 90; $attempt++) {
            try {
                $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) {
                    Start-Process $Url
                    return
                }
            } catch {
                Start-Sleep -Seconds 1
            }
        }
    } -ArgumentList $dashboardUrl
}

Write-Host "Starting VirtuCEO model server on $dashboardUrl"
Write-Host "Press Ctrl+C to stop the server and any managed Cloudflare tunnel."

try {
    & $python -m uvicorn server:app --host 0.0.0.0 --port $Port
} finally {
    if ($browserJob) {
        Stop-Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job $browserJob -Force -ErrorAction SilentlyContinue
    }
}
