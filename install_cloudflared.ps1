param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolsDir = Join-Path $projectDir ".tools"
$destination = Join-Path $toolsDir "cloudflared.exe"
$downloadUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
if ((Test-Path -LiteralPath $destination) -and -not $Force) {
    Write-Host "cloudflared already exists at $destination"
    & $destination --version
    exit 0
}

Write-Host "Downloading the latest 64-bit cloudflared release from Cloudflare..."
Invoke-WebRequest -Uri $downloadUrl -OutFile $destination -UseBasicParsing
& $destination --version
Write-Host "Installed cloudflared at $destination"
