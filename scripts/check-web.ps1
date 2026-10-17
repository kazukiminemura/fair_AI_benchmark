param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Host "Node.js is required for web regression checks. Install Node.js, then rerun scripts/check-web.ps1." -ForegroundColor Red
    exit 1
}

& $node.Source (Join-Path $root 'scripts/check_precision_display.js')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $node.Source (Join-Path $root 'scripts/check_mobile.js')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $node.Source --check (Join-Path $root 'fairbench/web/app.js')
exit $LASTEXITCODE
